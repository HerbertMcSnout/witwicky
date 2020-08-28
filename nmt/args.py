import sys
import argparse
import nmt.all_constants as ac

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'translate', 'extract'], default='train')
parser.add_argument('--proto', type=str, required=True,
                    help='Training config defined in configurations.py')
parser.add_argument('--num-preload', type=int, default=ac.DEFAULT_NUM_PRELOAD,
                    help="""
                         Number of samples prefetched to memory.
                         Small is slower but too big might make data
                         less randomized.""")
parser.add_argument('--input-file', type=str, 
                    help='Input file if mode == translate')
parser.add_argument('--model-file', type=str, required=False,
                    help='Path to saved checkpoint if mode == translate')
parser.add_argument('--var-list', nargs='+',
                    help='List of model vars to extracted')
parser.add_argument('--save-to', required='--var-list' in sys.argv,
                    help='Extract vars to this directory.')
parser.add_argument('--config-overrides', type=str,
                    help='Dict of k-v pairs to override config with')

