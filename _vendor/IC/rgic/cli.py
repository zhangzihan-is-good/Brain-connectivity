"""Minimal command-line interface for reference fitting and connectivity."""
import argparse
from importlib import resources
import json
import zipfile

from .core import RGIC, RGICConfig
from .io import (PACKAGE_VERSION, inspect_file, load_config, load_model,
                 load_recordings, load_roi_names, save_model, save_results)


_INPUT_HELP = "ROI time series: .npy (T,R)/(N,T,R), or X/record_* arrays in .npz"


def _help(parser):
    parser.add_argument("-h", "--help", "--need", "---need", action="help",
                        help="show help and exit (--need and ---need are aliases)")
    return parser


def _common_output(parser):
    parser.add_argument("--out", required=True, metavar="FILE.npz", help="output .npz file")
    parser.add_argument("--roi-labels", metavar="FILE", help="JSON label list or one label per line, in input column order")
    parser.add_argument("--overwrite", action="store_true", help="allow an existing output file to be overwritten")


def build_parser():
    parser = _help(argparse.ArgumentParser(
        prog="rgic", add_help=False, allow_abbrev=False,
        description="RGIC reliability-gated connectivity",
        epilog="Use fit to estimate reference reliability, then transform to compute target connectivity. See rgic manual for the user guide."))
    parser.add_argument("--version", action="version", version="rgic " + PACKAGE_VERSION)
    sub = parser.add_subparsers(dest="command", title="commands", metavar="COMMAND")

    def command(name, description):
        return _help(sub.add_parser(name, help=description, description=description,
                                    add_help=False, allow_abbrev=False))

    p = command("fit", "estimate F, L, S and R from reference records and save the model")
    p.add_argument("--input", required=True, metavar="FILE", help=_INPUT_HELP)
    p.add_argument("--config", metavar="JSON", help="RGIC configuration JSON; defaults are used when omitted")
    _common_output(p)

    p = command("transform", "compute target connectivity using fitted reference reliability")
    p.add_argument("--input", required=True, metavar="FILE", help=_INPUT_HELP)
    p.add_argument("--model", required=True, metavar="MODEL.npz", help="reference model saved by fit")
    _common_output(p)

    p = command("inspect", "show metadata and array shapes for input, model or result files")
    p.add_argument("file", metavar="FILE", help=".npy or .npz file to inspect")
    command("manual", "print the user guide in the terminal")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "fit":
            config = load_config(args.config) if args.config else RGICConfig()
            model = RGIC(config).fit(load_recordings(args.input), roi_names=load_roi_names(args.roi_labels))
            path = save_model(model, args.out, source_path=args.input, overwrite=args.overwrite)
            print(f"Reference model saved to {path}: {model.n_reference_} reference records, {model.n_regions_} ROIs.")
        elif args.command == "transform":
            model = load_model(args.model)
            results = model.transform(load_recordings(args.input), roi_names=load_roi_names(args.roi_labels))
            path = save_results(results, args.out, source_path=args.input, model_path=args.model,
                                overwrite=args.overwrite)
            print(f"Connectivity saved to {path}: {len(results)} target records. Reference reliability was kept fixed.")
        elif args.command == "inspect":
            print(json.dumps(inspect_file(args.file), ensure_ascii=False, indent=2, allow_nan=False))
        elif args.command == "manual":
            print(resources.files("rgic").joinpath("resources/user_guide.md").read_text(encoding="utf-8"))
    except (ValueError, TypeError, OSError, KeyError, EOFError, zipfile.BadZipFile,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\nRun rgic {args.command} --help for usage.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
