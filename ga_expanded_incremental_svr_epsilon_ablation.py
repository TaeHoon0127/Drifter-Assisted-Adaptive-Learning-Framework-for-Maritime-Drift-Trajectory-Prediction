"""GA-only SVR rerun: the final selected 10 inputs, standardized target, epsilon=0.1."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    sys.argv.extend(['--test-hours', '24', '--result-filename', 'ga_expanded_incremental_svr_epsilon_ablation.txt'])
    runpy.run_path(str(Path(__file__).with_name('expanded_incremental_svr_epsilon_ablation.py')), run_name='__main__')
