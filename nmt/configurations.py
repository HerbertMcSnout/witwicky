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

    # Penalize position embeddings that are too big,
    # if their struct has a get_reg_penalty function
    'pos_norm_penalty': 5e-2,

    # Module
    'struct': struct.sequence,

    # Whether to learn target position encodings
    'learned_pos': False,

    'learn_pos_scale': False,
    'separate_embed_scales': False,
    
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

    # TODO: Phase out? After first lr annealing, will get set to same lr as everything else
    #'embed_scale_lr': 0.03,

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

fun2com18 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_epochs': 30,
    'struct': struct.tree18,
    'batch_size': 3072,
    'pos_norm_penalty': 5e-2,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com17 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_epochs': 30,
    'struct': struct.tree,
    'batch_size': 3072,
    'pos_norm_penalty': 5e-2,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}


fun2com16 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_epochs': 30,
    'struct': struct.tree2,
    'batch_size': 3072,
    'pos_norm_penalty': 5e-2,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com14 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_epochs': 30,
    'struct': struct.tree14,
    'batch_size': 3072,
    'pos_norm_penalty': 5e-2,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com15 = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'data_dir': './nmt/data/fun2com',
    'max_epochs': 30,
    'struct': struct.tree15,
    'batch_size': 3072,
    'pos_norm_penalty': 5e-2,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com_seq = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'max_epochs': 30,
    'struct': struct.sequence,
    'batch_size': 3072,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com_rdr = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'max_epochs': 30,
    'struct': struct.sequence,
    'batch_size': 3072,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com_src = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'max_epochs': 30,
    'struct': struct.sequence,
    'batch_size': 3072,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}

fun2com_sbt = {
    'src_lang': 'fun',
    'trg_lang': 'com',
    'max_train_length': 2000,
    'max_epochs': 30,
    'struct': struct.sequence,
    'batch_size': 3072,
    'restore_segments': False,
    'warmup_style': ac.ORG_WARMUP,
}











#fun2com = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com2 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree2,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com3 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree3,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com4 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree4,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com5 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree5,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com6 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree6,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com7 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree7,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com8 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree8,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com9 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com10 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree10,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com11 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree11,
#    'batch_size': 3072,
#    'restore_segments': False,
#    'learn_pos_scale': True,
#    'separate_embed_scales': True,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com12 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree10,
#    'batch_size': 3072,
#    'pos_norm_penalty': 1.0, #5e-3,
#    'restore_segments': False,
#    'warmup_style': ac.ORG_WARMUP,
#}
#
#fun2com13 = {
#    'src_lang': 'fun',
#    'trg_lang': 'com',
#    'data_dir': './nmt/data/fun2com',
#    'max_epochs': 30,
#    'struct': struct.tree11,
#    'batch_size': 3072,
#    'pos_norm_penalty': 1.0, #5e-3,
#    'restore_segments': False,
#    'warmup_style': ac.ORG_WARMUP,
#}
