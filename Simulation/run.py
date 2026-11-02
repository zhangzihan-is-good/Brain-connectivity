"""One-command reproduction of the experiments and revised Fig. 1."""
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.workers < 1 or args.dpi < 72:
        parser.error("--workers must be positive and --dpi at least 72")
    output = args.output.resolve()
    import experiment
    previous = sys.argv
    try:
        sys.argv = ["experiment.py", "--workers", str(args.workers),
                    "--output", str(output / "results")]
        experiment.main()
    finally:
        sys.argv = previous
    from plot_figure import draw
    draw(output / "results", output / "figures", args.dpi)
    print("Figure and results: " + str(output), flush=True)


if __name__ == "__main__":
    main()
