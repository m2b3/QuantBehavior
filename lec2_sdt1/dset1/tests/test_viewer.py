from other.viewer import app


def test_marimo_viewer_renders_default_state():
    outputs, definitions = app.run()

    assert outputs
    assert definitions["navigation"].value == "Overview"
    assert definitions["sdt_experiment"].value == "Exp1"
    assert definitions["sdt_metric"].value == "dprime"
    assert definitions["sdt_metrics"].shape == (360, 16)
    assert definitions["sdt_summary"].shape == (36, 7)
    assert len(definitions["criterion_figure"].data) > 0
    assert definitions["criterion_figure"].layout.title.text == "Criterion (c)"
    assert definitions["criterion_figure"].layout.shapes[0].y0 == 0
    assert definitions["criterion_figure"].layout.shapes[0].y1 == 0
    assert definitions["gallery_page"].text.count("data:image/png;base64,") == 6
