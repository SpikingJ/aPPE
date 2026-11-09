import argparse
import importlib
import sys

import sb_eval


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('usage: python sb_run.py [--first-batch J] [--ebno-min X] <paper> [runner arguments ...]')
        sys.exit(0)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--first-batch', type=int, default=0)
    ap.add_argument('--ebno-min', type=float, default=None)
    ap.add_argument('paper')
    ap.add_argument('argv', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    paper, argv = args.paper, args.argv
    sb_eval.install(args.first_batch, args.ebno_min)
    module = importlib.import_module(f'sionna.bench.mimo_ofdm_chest.papers.{paper}.run_benchmark')
    sys.argv = [module.__file__] + argv
    module.main()


if __name__ == '__main__':
    main()
