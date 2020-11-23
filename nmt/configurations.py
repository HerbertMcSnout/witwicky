from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac
import nmt.structs as struct

'''You can add your own configuration variable to this file and select
it using `--proto variable_name`.'''

class Config(dict):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def copy(self):
        return self.__class__(**self)
    
    def adapt(self, **kwargs):
        warn = ('warn_new_option' in   self and   self['warn_new_option']) \
            or ('warn_new_option' in kwargs and kwargs['warn_new_option'])
        for k in kwargs:
            if warn and k not in self:
                raise KeyError(k)
        return self.__class__(**{k:v for k,v in list(self.items()) + list(kwargs.items())})

    def compute(self):
        computed = self.__class__()
        for k, v in self.items():
            computed[k] = v.format(**computed) if isinstance(v, str) else v
        return computed

def get_config(name, opts, overrides=None):
    '''
    Returns a dict of configurations, with default values taken from base_config.
    String options will be formatted with the values of the options defined before them,
    so "foo/{model_name}/baz" will be formatted to "foo/bar/baz" if the model name is "bar".
    The order of definition is that of base_config.
    '''
    overrides = eval(overrides or '{}', globals())
    overrides['model_name'] = name
    return opts.adapt(**overrides).compute()



base_config = Config(
    model_name = 'model_name',
    
    ### Locations of files
    save_to = 'nmt/saved_models/{model_name}',
    data_dir = 'nmt/data/{model_name}',
    log_file = '{save_to}/DEBUG.log',

    log_freq = 100,

    # Source and target languages
    # Input files should be named with these as extensions
    src_lang = 'src_lang',
    trg_lang = 'trg_lang',

    ### Model options

    # Vocabulary sizes
    src_vocab_size = 0,
    trg_vocab_size = 0,
    joint_vocab_size = 0,
    share_vocab = False,

    # Normalize word embeddings (Nguyen and Chiang, 2018)
    fix_norm = False,

    # Tie word embeddings
    tie_mode = ac.ALL_TIED,

    # Penalize position embeddings that are too big,
    # if their struct has a get_reg_penalty function
    pos_norm_penalty = 5e-2,

    # Module
    struct = struct.sequence,
    add_sinusoidal_pe_src = False,

    # Whether to learn position embeddings
    learned_pos_src = False,
    learned_pos_trg = False,

    learn_pos_scale = False,
    separate_embed_scales = False,
    
    # Layer sizes
    embed_dim = 512,
    ff_dim = 512 * 4,
    num_enc_layers = 6,
    num_enc_heads = 8,
    num_dec_layers = 6,
    num_dec_heads = 8,

    # Whether residual connections should bypass layer normalization
    # if True, layer-norm->dropout->add
    # if False, dropout->add->layer-norm (as in original paper)
    norm_in = True,

    ### Dropout/smoothing options

    dropout = 0.3,
    word_dropout = 0.1,
    label_smoothing = 0.1,

    ### Training options

    batch_sort_src = True,
    batch_size = 4096,
    weight_init_type = ac.XAVIER_NORMAL,
    normalize_loss = ac.LOSS_TOK,

    # Hyperparameters for Adam optimizer
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,

    # Learning rate
    warmup_steps = 24000,
    warmup_style = ac.NO_WARMUP,
    lr = 3e-4,
    lr_decay = 0, # if this is set to > 0, we'll do annealing
    start_lr = 1e-8,
    min_lr = 1e-5,
    lr_decay_patience = 3, # if no improvements for this many epochs, anneal learning rate
    early_stop_patience = 20, # if no improvements for this many epochs, stop early

    # Gradient clipping
    grad_clip = 1.0, # if no clip, just set it to some big value like 1e9
    grad_clamp = 0, # if not 0, clamp gradients to [-grad_clamp, +grad_clamp]. This happens *before* gradient clipping.
    grad_clip_pe = 0, # if 0, clip position embedding params along with all others; otherwise, clip them separately to this value

    ### Validation/stopping options

    max_epochs = 100,
    validate_freq = 1, # eval every [this many] epochs
    val_per_epoch = True, # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    val_by_bleu = True,
    write_val_trans = False,

    # Undo BPE segmentation when validating
    restore_segments = True,

    # How many of the best models to save
    n_best = 1,

    bleu_script = 'scripts/multi-bleu.perl',

    ### Length model

    # Choices are:
    # - gnmt: https://arxiv.org/abs/1609.08144 equation 14
    # - linear: constant reward per word
    # - none
    length_model = ac.GNMT_LENGTH_MODEL,
    # For gnmt, this is the exponent; for linear, this is the strength of the reward
    length_alpha = 0.6,

    # Filter out sentences longer than this (minus one for bos/eos)
    max_src_length = 1000,
    max_trg_length = 1000,

    ### Decoding options
    beam_size = 4,

    # Warn if an adaptation introduces a new option (as it may be a typo)
    warn_new_option = True,
)



#######################################################################

second_base = base_config.adapt(
    learned_pos_src = True,
    max_src_length = 1000,
    max_epochs = 200,
    early_stop_patience = 0,
    validate_freq = 1,
    #length_model = ac.LINEAR_LENGTH_MODEL,
    #length_alpha = 0.6,
    dropout = 0.2,
    lr = 1e-4,
    restore_segments = False,
)


##########################

# Untagging doesn't really work for this dataset
en2vi_base = base_config.adapt(src_lang = 'en', trg_lang = 'vi', early_stop_patience = 0, learned_pos_src = True)
en2vi17a2_base = en2vi_base.adapt(struct = struct.tree17a2, grad_clamp = 100.0)
en2vi_tree_nomask_base = en2vi_base.adapt(struct = struct.tree17c, grad_clamp = 100.0)
#en2vi_tree = en2vi_base.adapt(struct = struct.tree17c, grad_clamp = 100.0)
#en2vi17a_base = en2vi_base.adapt(struct = struct.tree17a, grad_clamp = 100.0)
#en2vi17a = en2vi17a_base.adapt(data_dir = 'nmt/data/en2vi_tree')

#en2vi17a_20k = en2vi17a.adapt(data_dir = 'nmt/data/en2vi_tree_20k')
#en2vi17_20k = en2vi_tree.adapt(data_dir = 'nmt/data/en2vi_tree_20k')

en2vi = en2vi_base # baseline
en2vi_10k = en2vi  # baseline
en2vi_20k = en2vi  # baseline
en2vi_50k = en2vi  # baseline

en2vi_att_sin = en2vi_base.adapt(data_dir = 'nmt/data/en2vi_tree', struct = struct.att_sin, add_sinusoidal_pe_src = True)
en2vi_att_sin_10k = en2vi_att_sin.adapt(data_dir = 'nmt/data/en2vi_tree_10k')
en2vi_att_sin_20k = en2vi_att_sin.adapt(data_dir = 'nmt/data/en2vi_tree_20k')
en2vi_att_sin_50k = en2vi_att_sin.adapt(data_dir = 'nmt/data/en2vi_tree_50k')

en2vi_att_sin2 = en2vi_att_sin

en2vi_seq     = en2vi.adapt(data_dir = 'nmt/data/en2vi_tree')
en2vi_seq_10k = en2vi.adapt(data_dir = 'nmt/data/en2vi_tree_10k')
en2vi_seq_20k = en2vi.adapt(data_dir = 'nmt/data/en2vi_tree_20k')
en2vi_seq_50k = en2vi.adapt(data_dir = 'nmt/data/en2vi_tree_50k')

en2vi17a2     = en2vi17a2_base.adapt(data_dir = 'nmt/data/en2vi_tree')
en2vi17a2_10k = en2vi17a2_base.adapt(data_dir = 'nmt/data/en2vi_tree_10k')
en2vi17a2_20k = en2vi17a2_base.adapt(data_dir = 'nmt/data/en2vi_tree_20k')
en2vi17a2_50k = en2vi17a2_base.adapt(data_dir = 'nmt/data/en2vi_tree_50k')

en2vi_tree_nomask     = en2vi_tree_nomask_base.adapt(data_dir = 'nmt/data/en2vi_tree')
en2vi_tree_nomask_10k = en2vi_tree_nomask_base.adapt(data_dir = 'nmt/data/en2vi_tree_10k')
en2vi_tree_nomask_20k = en2vi_tree_nomask_base.adapt(data_dir = 'nmt/data/en2vi_tree_20k')
en2vi_tree_nomask_50k = en2vi_tree_nomask_base.adapt(data_dir = 'nmt/data/en2vi_tree_50k')

en2viLR = en2vi.adapt(data_dir = 'nmt/data/en2vi', struct = struct.attLR)

##########################
en2tu_base = base_config.adapt(src_lang = 'en', trg_lang = 'tu', early_stop_patience = 0, learned_pos_src = True)
en2tu = en2tu_base
en2tu_seq = en2tu_base.adapt(data_dir = 'nmt/data/en2tu_tree')
en2tu17a2 = en2tu_base.adapt(data_dir = 'nmt/data/en2tu_tree', struct = struct.tree17a2, grad_clamp = 100.0)
en2tu_tree_nomask = en2tu_base.adapt(data_dir = 'nmt/data/en2tu_tree', struct = struct.tree17c, grad_clamp = 100.0)
en2tu_att_sin = en2tu_base.adapt(data_dir = 'nmt/data/en2tu_tree', struct = struct.att_sin, add_sinusoidal_pe_src = True)

##########################
en2ha_base = base_config.adapt(src_lang = 'en', trg_lang = 'ha', early_stop_patience = 0, learned_pos_src = True)
en2ha = en2ha_base
en2ha_seq = en2ha_base.adapt(data_dir = 'nmt/data/en2ha_tree')
en2ha17a2 = en2ha_base.adapt(data_dir = 'nmt/data/en2ha_tree', struct = struct.tree17a2, grad_clamp = 100.0)
en2ha_tree_nomask = en2ha_base.adapt(data_dir = 'nmt/data/en2ha_tree', struct = struct.tree17c, grad_clamp = 100.0)
en2ha_att_sin = en2ha_base.adapt(data_dir = 'nmt/data/en2ha_tree', struct = struct.att_sin, add_sinusoidal_pe_src = True)

##########################
en2ur_base = base_config.adapt(src_lang = 'en', trg_lang = 'ur', early_stop_patience = 0, learned_pos_src = True)
en2ur = en2ur_base
en2ur_seq = en2ur_base.adapt(data_dir = 'nmt/data/en2ur_tree')
en2ur17a2 = en2ur_base.adapt(data_dir = 'nmt/data/en2ur_tree', struct = struct.tree17a2, grad_clamp = 100.0)
en2ur_tree_nomask = en2ur_base.adapt(data_dir = 'nmt/data/en2ur_tree', struct = struct.tree17c, grad_clamp = 100.0)
en2ur_att_sin = en2ur_base.adapt(data_dir = 'nmt/data/en2ur_tree', struct = struct.att_sin, add_sinusoidal_pe_src = True)


##########################

de2en_base = base_config.adapt(src_lang = 'de', trg_lang = 'en', early_stop_patience = 10, learned_pos_src = True)
de2en = de2en_base
de2en_seq = de2en_base.adapt(data_dir = 'nmt/data/de2en_tree')
de2en17a2 = de2en_base.adapt(data_dir = 'nmt/data/de2en_tree', struct = struct.tree17a2, grad_clamp = 100.0)
de2en_tree_nomask = de2en_base.adapt(data_dir = 'nmt/data/de2en_tree', struct = struct.tree17c, grad_clamp = 100.0)
de2en_att_sin = de2en_base.adapt(data_dir = 'nmt/data/de2en_tree', struct = struct.att_sin, add_sinusoidal_pe_src = True)

de2en_100k = de2en_base
de2en_seq_100k = de2en_seq.adapt(data_dir = 'nmt/data/de2en_tree_100k')
de2en17a2_100k = de2en17a2.adapt(data_dir = 'nmt/data/de2en_tree_100k')
de2en_tree_nomask_100k = de2en_tree_nomask.adapt(data_dir = 'nmt/data/de2en_tree_100k')
de2en_att_sin_100k = de2en_att_sin.adapt(data_dir = 'nmt/data/de2en_tree_100k')

de2en_500k = de2en_base
de2en_seq_500k = de2en_seq.adapt(data_dir = 'nmt/data/de2en_tree_500k')
de2en17a2_500k = de2en17a2.adapt(data_dir = 'nmt/data/de2en_tree_500k')
de2en_tree_nomask_500k = de2en_tree_nomask.adapt(data_dir = 'nmt/data/de2en_tree_500k')
de2en_att_sin_500k = de2en_att_sin.adapt(data_dir = 'nmt/data/de2en_tree_500k')

de2en_20k = de2en_base
de2en_seq_20k = de2en_seq.adapt(data_dir = 'nmt/data/de2en_tree_20k')
de2en17a2_20k = de2en17a2.adapt(data_dir = 'nmt/data/de2en_tree_20k')
de2en_tree_nomask_20k = de2en_tree_nomask.adapt(data_dir = 'nmt/data/de2en_tree_20k')
de2en_att_sin_20k = de2en_att_sin.adapt(data_dir = 'nmt/data/de2en_tree_20k')

de2en_10k = de2en_base
de2en_seq_10k = de2en_seq.adapt(data_dir = 'nmt/data/de2en_tree_10k')
de2en17a2_10k = de2en17a2.adapt(data_dir = 'nmt/data/de2en_tree_10k')
de2en_tree_nomask_10k = de2en_tree_nomask.adapt(data_dir = 'nmt/data/de2en_tree_10k')
de2en_att_sin_10k = de2en_att_sin.adapt(data_dir = 'nmt/data/de2en_tree_10k')

de2en_50k = de2en_base
de2en_seq_50k = de2en_seq.adapt(data_dir = 'nmt/data/de2en_tree_50k')
de2en17a2_50k = de2en17a2.adapt(data_dir = 'nmt/data/de2en_tree_50k')
de2en_tree_nomask_50k = de2en_tree_nomask.adapt(data_dir = 'nmt/data/de2en_tree_50k')
de2en_att_sin_50k = de2en_att_sin.adapt(data_dir = 'nmt/data/de2en_tree_50k')

##########################

en2de_base = base_config.adapt(src_lang = 'en', trg_lang = 'de', early_stop_patience = 10, learned_pos_src = True)
en2de = en2de_base
en2de_seq = en2de_base.adapt(data_dir = 'nmt/data/en2de_tree')
en2de17a2 = en2de_base.adapt(data_dir = 'nmt/data/en2de_tree', struct = struct.tree17a2, grad_clamp = 100.0)
en2de_tree_nomask = en2de_base.adapt(data_dir = 'nmt/data/en2de_tree', struct = struct.tree17c, grad_clamp = 100.0)
en2de_att_sin = en2de_base.adapt(data_dir = 'nmt/data/en2de_tree', struct = struct.att_sin, add_sinusoidal_pe_src = True)

en2de_100k = en2de_base
en2de_seq_100k = en2de_seq.adapt(data_dir = 'nmt/data/en2de_tree_100k')
en2de17a2_100k = en2de17a2.adapt(data_dir = 'nmt/data/en2de_tree_100k')
en2de_tree_nomask_100k = en2de_tree_nomask.adapt(data_dir = 'nmt/data/en2de_tree_100k')
en2de_att_sin_100k = en2de_att_sin.adapt(data_dir = 'nmt/data/en2de_tree_100k')

en2de_500k = en2de_base
en2de_seq_500k = en2de_seq.adapt(data_dir = 'nmt/data/en2de_tree_500k')
en2de17a2_500k = en2de17a2.adapt(data_dir = 'nmt/data/en2de_tree_500k')
en2de_tree_nomask_500k = en2de_tree_nomask.adapt(data_dir = 'nmt/data/en2de_tree_500k')
en2de_att_sin_500k = en2de_att_sin.adapt(data_dir = 'nmt/data/en2de_tree_500k')

en2de_20k = en2de_base
en2de_seq_20k = en2de_seq.adapt(data_dir = 'nmt/data/en2de_tree_20k')
en2de17a2_20k = en2de17a2.adapt(data_dir = 'nmt/data/en2de_tree_20k')
en2de_tree_nomask_20k = en2de_tree_nomask.adapt(data_dir = 'nmt/data/en2de_tree_20k')
en2de_att_sin_20k = en2de_att_sin.adapt(data_dir = 'nmt/data/en2de_tree_20k')

en2de_10k = en2de_base
en2de_seq_10k = en2de_seq.adapt(data_dir = 'nmt/data/en2de_tree_10k')
en2de17a2_10k = en2de17a2.adapt(data_dir = 'nmt/data/en2de_tree_10k')
en2de_tree_nomask_10k = en2de_tree_nomask.adapt(data_dir = 'nmt/data/en2de_tree_10k')
en2de_att_sin_10k = en2de_att_sin.adapt(data_dir = 'nmt/data/en2de_tree_10k')

en2de_50k = en2de_base
en2de_seq_50k = en2de_seq.adapt(data_dir = 'nmt/data/en2de_tree_50k')
en2de17a2_50k = en2de17a2.adapt(data_dir = 'nmt/data/en2de_tree_50k')
en2de_tree_nomask_50k = en2de_tree_nomask.adapt(data_dir = 'nmt/data/en2de_tree_50k')
en2de_att_sin_50k = en2de_att_sin.adapt(data_dir = 'nmt/data/en2de_tree_50k')
