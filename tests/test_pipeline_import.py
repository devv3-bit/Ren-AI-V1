def test_pipeline_import():
    from src.pipeline.run_pipeline import run_pipeline
    assert callable(run_pipeline)
