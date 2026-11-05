"""Original GA prediction generation: the same 10-feature code, test_hours=24."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    sys.argv.extend(['--test-hours', '24', '--result-filename', 'ga_expanded_incremental_learning_correction.txt'])
    runpy.run_path(str(Path(__file__).with_name('expanded_incremental_learning_correction.py')), run_name='__main__')
