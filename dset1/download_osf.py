#!/usr/bin/env python3
"""Resumable, checksum-verified parallel downloader for a public OSF project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


USER_AGENT = "osf-project-downloader/1.0"


def open_url(url: str, timeout: int):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def listing_url(url: str, stable_sort: bool = True) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["page[size]"] = "100"
    if stable_sort:
        query["sort"] = "name"
    return urlunsplit(parts._replace(query=urlencode(query)))


def get_json(url: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with open_url(url, timeout=60) as response:
                return json.load(response)
        except (ValueError, HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to query {url}: {last_error}")


def list_folder(url: str, stable_sort: bool = True) -> list[dict]:
    records: list[dict] = []
    expected_total: int | None = None
    next_url: str | None = listing_url(url, stable_sort)
    while next_url:
        payload = get_json(next_url)
        expected_total = payload["links"]["meta"]["total"]
        records.extend(payload["data"])
        raw_next = payload["links"]["next"]
        next_url = listing_url(raw_next, stable_sort) if raw_next else None

    unique_ids = {record["id"] for record in records}
    if expected_total is not None and len(unique_ids) != expected_total:
        raise RuntimeError(
            f"incomplete OSF listing for {url}: "
            f"{len(unique_ids)} unique records != {expected_total} reported"
        )
    return records


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item: tuple[Path, str, int | None, str | None]) -> str:
    destination, url, expected_size, expected_md5 = item
    if destination.is_file():
        size_matches = expected_size is None or destination.stat().st_size == expected_size
        hash_matches = expected_md5 is None or md5sum(destination) == expected_md5
        if size_matches and hash_matches:
            return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None

    for attempt in range(5):
        try:
            with open_url(url, timeout=300) as response:
                with partial.open("wb") as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)

            if expected_size is not None and partial.stat().st_size != expected_size:
                raise OSError(
                    f"size mismatch for {destination}: "
                    f"{partial.stat().st_size} != {expected_size}"
                )
            if expected_md5 is not None and md5sum(partial) != expected_md5:
                raise OSError(f"MD5 mismatch for {destination}")

            os.replace(partial, destination)
            return "downloaded"
        except (OSError, HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(2**attempt)

    raise RuntimeError(f"failed to download {destination}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="OSF project ID, e.g. pcunw")
    parser.add_argument("output", nargs="?", default=".")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    items_by_path: dict[Path, tuple[Path, str, int | None, str | None]] = {}

    providers_url = f"https://api.osf.io/v2/nodes/{args.project}/files/"
    for storage in list_folder(providers_url, stable_sort=False):
        storage_name = storage["attributes"]["name"]
        storage_root = (output / storage_name).resolve()
        folders = [storage["relationships"]["files"]["links"]["related"]["href"]]

        while folders:
            for remote_file in list_folder(folders.pop()):
                attributes = remote_file["attributes"]
                if attributes["kind"] == "folder":
                    folders.append(
                        remote_file["relationships"]["files"]["links"]["related"]["href"]
                    )
                    continue

                relative = PurePosixPath(attributes["materialized_path"].lstrip("/"))
                destination = (storage_root / Path(*relative.parts)).resolve()
                if storage_root not in destination.parents:
                    raise ValueError(f"unsafe remote path: {attributes['materialized_path']}")
                item = (
                    destination,
                    remote_file["links"]["download"],
                    attributes["size"],
                    attributes["extra"]["hashes"].get("md5"),
                )
                previous = items_by_path.get(destination)
                if previous is not None and previous[2:] != item[2:]:
                    raise ValueError(f"conflicting OSF records for {destination}")
                items_by_path[destination] = item

    items = list(items_by_path.values())

    downloaded = skipped = failed = 0
    total = len(items)
    print(f"Found {total} files; downloading with {args.workers} workers")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, item): item[0] for item in items}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                if result == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as error:
                failed += 1
                print(error)

            if completed % 10 == 0 or completed == total:
                print(
                    f"{completed}/{total} checked "
                    f"({downloaded} downloaded, {skipped} reused, {failed} failed)",
                    flush=True,
                )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
