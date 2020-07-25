import os
import time
import sys
import numpy
import torch

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations


class Translator(object):
    def __init__(self, args):
        super(Translator, self).__init__()
        self.config = configurations.get_config(args.proto, getattr(configurations, args.proto), args.config_overrides)
        self.logger = ut.get_logger(self.config['log_file'])
        self.num_preload = args.num_preload

        self.model_file = args.model_file
        if self.model_file is None:
            self.model_file = os.path.join(self.config['save_to'], '{}.pth'.format(self.config['model_name']))

        self.input_file = args.input_file
        if not os.path.exists(self.input_file):
            raise ValueError('Input file does not exist: {}'.format(self.input_file))
        if not os.path.exists(self.model_file):
            raise ValueError('Model file does not exist: {}'.format(self.model_file))

        if self.input_file:
            save_fp = os.path.join(self.config['save_to'], os.path.basename(self.input_file))
            self.best_output_fp = save_fp + '.best_trans'
            self.beam_output_fp = save_fp + '.beam_trans'
            open(self.best_output_fp, 'w').close()
            open(self.beam_output_fp, 'w').close()
        else:
            self.best_output_fp = self.beam_output_fp = None

        self.data_manager = DataManager(self.config)
        self.model = Model(self.config).to(ut.get_device())
        self.logger.info('Restore model from {}'.format(self.model_file))
        self.model.load_state_dict(torch.load(self.model_file))
        self.translate()

    def translate(self):
        best_stream = open(self.best_output_fp, 'a') if self.best_output_fp else sys.stdout
        beam_stream = open(self.beam_output_fp, 'a') if self.beam_output_fp else None
        self.data_manager.translate(self.model,
                                    self.input_file or sys.stdin,
                                    best_stream,
                                    beam_stream,
                                    mode=ac.TESTING,
                                    to_ids=True,
                                    num_preload=self.num_preload)
        if self.best_output_fp: best_stream.close()
        if self.beam_output_fp: beam_stream.close()


    def plot_head_map(self, mma, target_labels, target_ids, source_labels, source_ids, filename):
        """https://github.com/EdinburghNLP/nematus/blob/master/utils/plot_heatmap.py
        Change the font in family param below. If the system font is not used, delete matplotlib
        font cache https://github.com/matplotlib/matplotlib/issues/3590
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
        ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

        # without this I get some extra columns rows
        # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
        ax.set_xlim(0, int(mma.shape[1]))
        ax.set_ylim(0, int(mma.shape[0]))

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # source words -> column labels
        ax.set_xticklabels(source_labels, minor=False, family='Source Code Pro')
        for xtick, idx in zip(ax.get_xticklabels(), source_ids):
            if idx == ac.UNK_ID:
                xtick.set_color('b')
        # target words -> row labels
        ax.set_yticklabels(target_labels, minor=False, family='Source Code Pro')
        for ytick, idx in zip(ax.get_yticklabels(), target_ids):
            if idx == ac.UNK_ID:
                ytick.set_color('b')

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
