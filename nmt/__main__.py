from __future__ import print_function
from __future__ import division

from nmt.train import Trainer
from nmt.translate import Translator
from nmt.extractor import Extractor
from nmt.interactive import InteractiveTranslator

from nmt.args import parser

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()
    elif args.mode == 'translate':
        translator = Translator(args)
    elif args.mode == 'extract':
        extractor = Extractor(args)
    elif args.mode == 'interactive':
        InteractiveTranslator(args)
