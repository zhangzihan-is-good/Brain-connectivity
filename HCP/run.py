"""Recompute HCP main and appendix tables and figures from external inputs."""
from pathlib import Path
import argparse
import subprocess
import sys
import _shared
from shared_paths import workspace_root

ROOT = Path(__file__).resolve().parent
INPUTS = workspace_root() / 'dataset' / 'reproduction_inputs' / 'HCP'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=('all', 'main', 'appendix'), default='all')
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=3)
    parser.add_argument('--panels', action='store_true', help='Also export separate main panels.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compute-only', action='store_true')
    mode.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    output, inputs = args.output.resolve(), args.inputs.resolve()
    if not args.plot_only and output.exists() and any(output.iterdir()):
        parser.error('Choose an empty --output directory for a fresh computation.')

    def run(script, *arguments):
        subprocess.run([sys.executable, '-B', str(ROOT / script), *map(str, arguments)], check=True)

    if args.section in ('all', 'main'):
        main_output = output / 'main'
        if not args.plot_only:
            run('evaluate.py', '--data', inputs / 'inputs.npz', '--weights', inputs / 'sc_weights.npy',
                '--output', main_output, '--workers', args.workers)
        if not args.compute_only:
            arguments = ['--results', main_output, '--output', main_output / 'figures']
            if args.panels:
                arguments.append('--panels')
            run('plot.py', *arguments)
    if args.section in ('all', 'appendix'):
        appendix_output = output / 'appendix'
        data = inputs / 'appendix_inputs.npz'
        if not args.plot_only:
            run('describe_features.py', '--data', data, '--results', appendix_output)
            run('supplement_diagnostics.py', '--data', data, '--results', appendix_output,
                '--workers', args.workers)
        if not args.compute_only:
            run('render_figure.py', '--data', data, '--results', appendix_output,
                '--output', appendix_output / 'figures')


if __name__ == '__main__':
    main()
