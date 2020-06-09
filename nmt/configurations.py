from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac
import nmt.structs as struct

"""You can add your own configuration function to this file and select
it using `--proto function_name`."""


def get_config(name, opts):
    config = {k:(v.format(name=name) if isinstance(v, str) else v)
              for k, v in base_config.items()}
    config.update(opts)
    return config


base_config = {
    ### Locations of files
    'save_to': './nmt/saved_models/{name}',
    'data_dir': './nmt/data/{name}',
    'log_file': './nmt/saved_models/{name}/DEBUG.log',

    # The name of the model
    'model_name': '{name}',

    # Source and target languages
    # Input files should be named with these as extensions
    'src_lang': 'src_lang',
    'trg_lang': 'trg_lang',

    ### Model options

    # Filter out sentences longer than this (minus one for bos/eos)
    'max_train_length': 1000,

    # Vocabulary sizes
    'src_vocab_size': 0,
    'trg_vocab_size': 0,
    'joint_vocab_size': 0,
    'share_vocab': False,

    # Normalize word embeddings (Nguyen and Chiang, 2018)
    'fix_norm': False,

    # Tie word embeddings
    'tie_mode': ac.ALL_TIED,

    # Penalize position embeddings that are too big
    'pos_norm_penalty': 5e-3, # set to 0 if you want no penalty
    'pos_norm_scale': 16., # sqrt(embed_dim / 2)

    # Module
    'struct': struct.sequence,

    # Whether to learn position encodings
    'learned_pos': False,
    
    # Layer sizes
    'embed_dim': 512,
    'ff_dim': 512 * 4,
    'num_enc_layers': 6,
    'num_enc_heads': 8,
    'num_dec_layers': 6,
    'num_dec_heads': 8,

    # Whether residual connections should bypass layer normalization
    # if True, layer-norm->dropout->add
    # if False, dropout->add->layer-norm (as in original paper)
    'norm_in': True,

    ### Dropout/smoothing options

    'dropout': 0.3,
    'word_dropout': 0.1,
    'label_smoothing': 0.1,

    ### Training options

    'batch_sort_src': True,
    'batch_size': 4096,
    'weight_init_type': ac.XAVIER_NORMAL,
    'normalize_loss': ac.LOSS_TOK,

    # Hyperparameters for Adam optimizer
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,

    # Learning rate
    'warmup_steps': 24000,
    'warmup_style': ac.NO_WARMUP,
    'lr': 3e-4,
    'lr_decay': 0.8, # if this is set to > 0, we'll do annealing
    'start_lr': 1e-8,
    'min_lr': 1e-5,
    'lr_decay_patience': 3, # if no improvements for this many epochs, anneal learning rate
    'early_stop_patience': 10, # if no improvements for this many epochs, stop early

    'embed_scale_lr': 0.03,

    # Gradient clipping
    'grad_clip': 1.0, # if no clip, just set it to some big value like 1e9

    ### Validation/stopping options

    'max_epochs': 100,
    'validate_freq': 1.0, # eval every [this many] epochs
    'val_per_epoch': True, # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    'val_by_bleu': True,

    # Undo BPE segmentation when validating
    'restore_segments': True,

    # How many of the best models to save
    'n_best': 1,

    ### Length model

    # Choices are:
    # - gnmt: https://arxiv.org/abs/1609.08144 equation 14
    # - linear: constant reward per word
    # - none
    'length_model': ac.GNMT_LENGTH_MODEL,
    # For gnmt, this is the exponent; for linear, this is the strength of the reward
    'length_alpha': 0.6,

    ### Decoding options
    'beam_size': 4,   
}

fun2com = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.tree,
    'batch_size': 2048,
    'restore_segments': False,
#    'warmup_style': ac.ORG_WARMUP,
}

fun2com2 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.tree2,
    'batch_size': 2048,
    'restore_segments': False,
}

fun2coml = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.sequence,
    'pos_norm_penalty': 0,
    'batch_size': 2048,
    'restore_segments': False,
}

fun2com3 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.tree3,
    'pos_norm_penalty': 0,
    'batch_size': 2048,
    'restore_segments': False,
}

fun2com4 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.tree4,
    'pos_norm_penalty': 0,
    'batch_size': 2048,
    'restore_segments': False,
}

fun2com5 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.tree5,
    'pos_norm_penalty': 0,
    'batch_size': 2048,
    'restore_segments': False,
}
