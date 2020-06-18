import os
import time

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
        self.config = configurations.get_config(args.proto, getattr(configurations, args.proto))
        self.logger = ut.get_logger(self.config['log_file'])

        self.input_file = args.input_file
        self.model_file = args.model_file

        if self.input_file is None or self.model_file is None or not os.path.exists(self.input_file) or not os.path.exists(self.model_file):
            raise ValueError('Input file or model file does not exist')

        self.data_manager = DataManager(self.config)
        self.translate()

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

    def translate(self):
        model = Model(self.config).to(ut.get_device())
        self.logger.info('Restore model from {}'.format(self.model_file))
        model.load_state_dict(torch.load(self.model_file))
        self.data_manager.translate(model, self.input_file, self.config['save_to'], self.logger)
