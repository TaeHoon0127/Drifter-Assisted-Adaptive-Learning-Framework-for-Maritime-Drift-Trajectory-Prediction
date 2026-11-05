import os

from revision_rbfn_common import run_experiment


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from publication_utils import arguments

args = arguments("rbfn")
DATASETS = (args.data_path,)
if args.java_bin:
    os.environ["JAVA_BIN"] = args.java_bin
if args.weka_classpath:
    os.environ["WEKA_CLASSPATH"] = args.weka_classpath
if args.decimals < 0:
    raise ValueError("decimals must be nonnegative")


def main():
    run_experiment(
        script_dir=SCRIPT_DIR,
        datasets=DATASETS,
        target_cols=(12, 13),
        output_filename=os.path.join(args.save_path, 'revision_rbfn_cor.txt'),
        apply_correction=True,
        decimal_places=args.decimals,
        distance=args.distance,
    )


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
