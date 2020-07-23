from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac
import nmt.structs as struct

'''You can add your own configuration variable to this file and select
it using `--proto variable_name`.'''

def adapt(base, **kwargs):
    'Returns a copy of base (dict), after updating it with kwargs'
    new = base.copy()
    for k in kwargs: assert k in base, 'Unknown config option "{}"'.format(k)
    new.update(kwargs)
    return new

def get_config(name, opts, overrides=None):
    '''
    Returns a dict of configurations, with default values taken from base_config.
    String options will be formatted with the values of the options defined before them,
    so "foo/{model_name}/baz" will be formatted to "foo/bar/baz" if the model name is "bar".
    The order of definition is that of base_config.
    '''
    overrides = eval(overrides or '{}', globals())
    overrides['model_name'] = name
    config = {}
    opts = adapt(opts, **overrides)
    for k, v in opts.items():
        config[k] = v.format(**config) if isinstance(v, str) else v
    return config


base_config = dict(
    model_name = 'model_name',
    
    ### Locations of files
    save_to = './nmt/saved_models/{model_name}',
    data_dir = './nmt/data/{model_name}',
    log_file = '{save_to}/DEBUG.log',

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

    # Whether to learn target position encodings
    learned_pos = False,

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
    lr_decay = 0.8, # if this is set to > 0, we'll do annealing
    start_lr = 1e-8,
    min_lr = 1e-5,
    lr_decay_patience = 3, # if no improvements for this many epochs, anneal learning rate
    early_stop_patience = 10, # if no improvements for this many epochs, stop early

    # Gradient clipping
    grad_clip = 1.0, # if no clip, just set it to some big value like 1e9

    ### Validation/stopping options

    max_epochs = 100,
    validate_freq = 1, # eval every [this many] epochs
    val_per_epoch = True, # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    val_by_bleu = True,

    # Undo BPE segmentation when validating
    restore_segments = True,

    # How many of the best models to save
    n_best = 1,

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
    write_val_trans = False,
)

fun2com_base = adapt(
    base_config,
    src_lang = 'fun',
    trg_lang = 'com',
    max_epochs = 30,
    batch_size = 3072,
    max_trg_length = 25,
    restore_segments = False,
    warmup_style = ac.ORG_WARMUP,
)

fun2com22 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree22,
)

fun2com21 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree21,
)

fun2com20 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree20,
)

fun2com19 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree19,
)

fun2com18 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree18,
)

fun2com17 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree,
)

fun2com173 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree173,
)

fun2com17_all = adapt(
    fun2com_base,
    struct = struct.tree,
)

fun2com172_all = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com17_all',
    struct = struct.tree172,
)

fun2com16 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree2,
)

fun2com15 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree15,
)

fun2com14 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree14,
)

fun2com142 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree142,
)

fun2com143 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree143,
)

fun2com144 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree144,
)

fun2com1442 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree1442,
)

fun2com1443 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree1443,
)

fun2com1444 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree1444,
)

fun2comM = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.treem,
)

fun2com1444i = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree1444i,
)

fun2com1444o = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree1444o,
)

fun2com1445 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree1445,
)

fun2com145 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree144,
    embed_dim = 256,
    ff_dim = 256 * 4,
)

fun2com146 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree144,
    embed_dim = 128,
    ff_dim = 128 * 4,
)

fun2com147 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree144,
    embed_dim = 64,
    ff_dim = 64 * 4,
)

fun2com148 = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree144,
    embed_dim = 32,
    ff_dim = 32 * 4,
)

fun2com_3d = adapt(
    fun2com_base,
    data_dir = './nmt/data/fun2com',
    struct = struct.tree14_3d,
    embed_dim = 64,
    ff_dim = 64 * 4,
)

fun2com_seq = adapt(
    fun2com_base,
    struct = struct.sequence,
)

fun2com_seq2 = adapt(
    fun2com_base,
    struct = struct.sequence2,
)

fun2com_rdr = adapt(
    fun2com_base,
    struct = struct.sequence,
)

fun2com_src = adapt(
    fun2com_base,
    struct = struct.sequence,
)

fun2com_sbt = adapt(
    fun2com_base,
    max_src_length = 2000,
    struct = struct.sequence,
)

#fun2com172 = dict(list(fun2com17.items()) + list(dict(learn_pos_scale = True, separate_embed_scales = True).items()))

fun2com_all = adapt(
    fun2com_base,
    max_src_length = 2000,
    struct = struct.sequence,
)

fun2com_rdr_all = adapt(
    fun2com_base,
    struct = struct.sequence,
)

fun2com_seq_all = adapt(
    fun2com_base,
    struct = struct.sequence,
)

fun2com_seq_all2 = adapt(
    fun2com_base,
    struct = struct.sequence2,
)


#######################################################################

second_base = adapt(
    base_config,
    max_src_length = 1000,
    max_epochs = 200,
    early_stop_patience = 20,
    validate_freq = 1,
    length_model = ac.LINEAR_LENGTH_MODEL,
    length_alpha = 0.2,
    dropout = 0.2,
    lr = 1e-4,
    restore_segments = False,
)


#######################################################################

java2doc_base = adapt(
    second_base,
    src_lang = 'java',
    trg_lang = 'doc',
    save_to = './nmt/java2doc_models/{model_name}',
    joint_vocab_size = 32000,
)

java2doc_tree_base = adapt(java2doc_base, data_dir = './nmt/data/java2doc')

java2doc14 = adapt(java2doc_tree_base, struct = struct.tree1444)
java2doc15 = adapt(java2doc_tree_base, struct = struct.tree14442)
java2doc17 = adapt(java2doc_tree_base, struct = struct.tree)
java2doc18 = adapt(java2doc_tree_base, struct = struct.tree172)
java2doc_seq = adapt(java2doc_base, struct = struct.sequence)
java2doc_rare = adapt(java2doc_base, struct = struct.sequence)
java2doc_raw2 = adapt(java2doc_base, struct = struct.sequence)
java2doc_raw = adapt(java2doc_base, struct = struct.sequence)




#######################################################################

py2doc_base = adapt(
    second_base,
    src_lang = 'py',
    trg_lang = 'doc',
    save_to = './nmt/py2doc_models/{model_name}',
    joint_vocab_size = 32000,
)

py2doc_tree_base = adapt(py2doc_base, data_dir = './nmt/data/py2doc')

py2doc14 = adapt(py2doc_tree_base, struct = struct.tree1444)
py2doc15 = adapt(py2doc_tree_base, struct = struct.tree14442)
py2doc17 = adapt(py2doc_tree_base, struct = struct.tree)
py2doc18 = adapt(py2doc_tree_base, struct = struct.tree172)
py2doc_seq = adapt(py2doc_base, struct = struct.sequence)
py2doc_rare = adapt(py2doc_base, struct = struct.sequence)
