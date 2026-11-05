"""30-feature SHAP correction; original source: feature_expansion/shap_correction.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shap_runner import run_shap_experiments


if __name__ == '__main__':
    run_shap_experiments('correction', slice(32, 34))
