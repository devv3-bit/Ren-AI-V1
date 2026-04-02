from pathlib import Path

def test_pipeline_runs_on_ckd_csv():
    from src.pipeline.run_pipeline import run_pipeline

    data_path = Path("data/raw/CKD.csv")
    assert data_path.exists(), "Expected data/raw/CKD.csv to exist"

    out = run_pipeline(str(data_path), require_target=True)

    # Basic sanity checks
    assert out.df_raw is not None
    assert out.df_clean is not None
    assert out.df_valid is not None

    # Feature matrices should exist by default
    assert out.X_base is not None
    assert out.X_base.shape[0] == out.df_valid.shape[0]
