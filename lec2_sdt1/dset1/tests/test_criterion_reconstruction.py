from criterion_reconstruction import app


def test_criterion_reconstruction_notebook_runs_from_csvs():
    outputs, definitions = app.run()

    assert outputs
    assert len(definitions["trials"]) == 32_525
    assert len(definitions["analysis_trials"]) == 31_420
    assert definitions["cell_metrics"].shape == (616, 8)
    assert definitions["cell_metrics"]["experiment"].nunique() == 3
    assert len(definitions["reconstruction_figure"].axes) == 6
