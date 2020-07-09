import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'translate', 'extract', 'interactive'], default='train')
parser.add_argument('--proto', type=str, required=True,
                    help='Training config function defined in configurations.py')
parser.add_argument('--num-preload', type=int, default=1000,
                    help="""Number of train samples prefetched to memory
Small is slower but too big might make training data
less randomized.""")
parser.add_argument('--input-file', type=str, 
                    help='Input file if mode == translate')
parser.add_argument('--model-file', type=str, required=False,
                    help='Path to saved checkpoint if mode == translate or interactive')
parser.add_argument('--src-formatter', type=str, required=False,
                    help='Optional path to executable that formats input from interactive mode')
parser.add_argument('--trg-formatter', type=str, required=False,
                    help='Optional path to executable that formats output from interactive mode')
parser.add_argument('--var-list', nargs='+',
                    help='List of model vars to extracted')
parser.add_argument('--save-to', required='--var-list' in sys.argv, help='Directory to save extracted vars to')
parser.add_argument('--config-overrides', type=str,
                    help='Dict of k-v pairs to override config with')

