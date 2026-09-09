from viewer import app


def test_marimo_viewer_renders_default_state():
    outputs, definitions = app.run()

    assert outputs
    assert definitions["navigation"].value == "Overview"
    assert definitions["sdt_experiment"].value == "Exp1"
    assert definitions["sdt_metric"].value == "dprime"
    assert definitions["sdt_metrics"].shape == (360, 16)
    assert definitions["sdt_summary"].shape == (36, 7)
