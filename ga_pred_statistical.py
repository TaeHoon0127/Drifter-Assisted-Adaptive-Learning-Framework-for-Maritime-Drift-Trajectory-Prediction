from pathlib import Path

from ga_pred_statistical_common import run


if __name__ == "__main__":
    run(
        default_test_dir="data_path",
        default_add_original="data_path/hourly",
        default_output_dir="save_path",
        script_path=Path(__file__),
    )
