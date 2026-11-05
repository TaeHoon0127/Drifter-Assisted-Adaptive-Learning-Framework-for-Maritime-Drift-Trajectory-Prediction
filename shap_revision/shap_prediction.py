"""30-feature SHAP prediction; original source: feature_expansion/shap_prediction.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shap_runner import run_shap_experiments


if __name__ == '__main__':
    run_shap_experiments('prediction', slice(30, 32))
