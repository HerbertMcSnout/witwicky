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
        for k in kwargs:
            if k not in self:
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
#    config = {}
#    opts = opts.adapt(**overrides)
#    for k, v in opts.items():
#        config[k] = v.format(**config) if isinstance(v, str) else v
#    return config


base_config = Config(
    model_name = 'model_name',
    
    ### Locations of files
    save_to = 'nmt/saved_models/{model_name}',
    data_dir = 'nmt/data/{model_name}',
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
    lr_decay = 0, # if this is set to > 0, we'll do annealing
    start_lr = 1e-8,
    min_lr = 1e-5,
    lr_decay_patience = 3, # if no improvements for this many epochs, anneal learning rate
    early_stop_patience = 20, # if no improvements for this many epochs, stop early

    # Gradient clipping
    grad_clip = 1.0, # if no clip, just set it to some big value like 1e9
    grad_clamp = 0, # if not 0, clamp gradients to [-grad_clamp, +grad_clamp]. This happens *before* gradient clipping.

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
)

fun2com_base = base_config.adapt(
    src_lang = 'fun',
    trg_lang = 'com',
    max_epochs = 30,
    batch_size = 3072,
    max_trg_length = 25,
    restore_segments = False,
    warmup_style = ac.ORG_WARMUP,
)

fun2com_tree_base = fun2com_base.adapt(data_dir = 'nmt/data/fun2com')

fun2com22 = fun2com_tree_base.adapt(
    struct = struct.tree22,
)

fun2com21 = fun2com_tree_base.adapt(
    struct = struct.tree21,
)

fun2com20 = fun2com_tree_base.adapt(
    struct = struct.tree20,
)

fun2com19 = fun2com_tree_base.adapt(
    struct = struct.tree19,
)

fun2com18 = fun2com_tree_base.adapt(
    struct = struct.tree18,
)

fun2com17 = fun2com_tree_base.adapt(
    struct = struct.tree,
)

fun2com173 = fun2com_tree_base.adapt(
    struct = struct.tree173,
)

fun2com17_all = fun2com_base.adapt(
    struct = struct.tree,
)

fun2com172_all = fun2com_base.adapt(
    data_dir = 'nmt/data/fun2com17_all',
    struct = struct.tree172,
)

fun2com16 = fun2com_tree_base.adapt(
    struct = struct.tree2,
)

fun2com15 = fun2com_tree_base.adapt(
    struct = struct.tree15,
)

fun2com14 = fun2com_tree_base.adapt(
    struct = struct.tree14,
)

fun2com142 = fun2com_tree_base.adapt(
    struct = struct.tree142,
)

fun2com143 = fun2com_tree_base.adapt(
    struct = struct.tree143,
)

fun2com144 = fun2com_tree_base.adapt(
    struct = struct.tree144,
)

fun2com1442 = fun2com_tree_base.adapt(
    struct = struct.tree1442,
)

fun2com1443 = fun2com_tree_base.adapt(
    struct = struct.tree1443,
)

fun2com1444 = fun2com_tree_base.adapt(
    struct = struct.tree1444,
)

fun2comM = fun2com_tree_base.adapt(
    struct = struct.treem,
)

fun2com1444i = fun2com_tree_base.adapt(
    struct = struct.tree1444i,
)

fun2com1444o = fun2com_tree_base.adapt(
    struct = struct.tree1444o,
)

fun2com1445 = fun2com_tree_base.adapt(
    struct = struct.tree1445,
)

fun2com145 = fun2com_tree_base.adapt(
    struct = struct.tree144,
    embed_dim = 256,
    ff_dim = 256 * 4,
)

fun2com146 = fun2com_tree_base.adapt(
    struct = struct.tree144,
    embed_dim = 128,
    ff_dim = 128 * 4,
)

fun2com147 = fun2com_tree_base.adapt(
    struct = struct.tree144,
    embed_dim = 64,
    ff_dim = 64 * 4,
)

fun2com148 = fun2com_tree_base.adapt(
    struct = struct.tree144,
    embed_dim = 32,
    ff_dim = 32 * 4,
)

fun2com_3d = fun2com_tree_base.adapt(
    struct = struct.tree14_3d,
    embed_dim = 64,
    ff_dim = 64 * 4,
)

fun2com_seq = fun2com_base.adapt(
    struct = struct.sequence,
)

fun2com_seq2 = fun2com_base.adapt(
    struct = struct.sequence2,
)

fun2com_rdr = fun2com_base.adapt(
    struct = struct.sequence,
)

fun2com_src = fun2com_base.adapt(
    struct = struct.sequence,
)

fun2com_sbt = fun2com_base.adapt(
    max_src_length = 2000,
    struct = struct.sequence,
)

#fun2com172 = dict(list(fun2com17.items()) + list(dict(learn_pos_scale = True, separate_embed_scales = True).items()))

fun2com_all = fun2com_base.adapt(
    max_src_length = 2000,
    struct = struct.sequence,
)

fun2com_rdr_all = fun2com_base.adapt(
    struct = struct.sequence,
)

fun2com_seq_all = fun2com_base.adapt(
    struct = struct.sequence,
)

fun2com_seq_all2 = fun2com_base.adapt(
    struct = struct.sequence2,
)


#######################################################################

second_base = base_config.adapt(
    max_src_length = 1000,
    max_epochs = 200,
    early_stop_patience = 0,
    validate_freq = 1,
    length_model = ac.LINEAR_LENGTH_MODEL,
    length_alpha = 0.2,
    dropout = 0.2,
    lr = 1e-4,
    restore_segments = False,
)


#######################################################################

java2doc_base = second_base.adapt(
    src_lang = 'java',
    trg_lang = 'doc',
    save_to = 'nmt/java2doc_models/{model_name}',
    joint_vocab_size = 32000,
    lr_decay = 0.8,
)

java2doc_tree_base = java2doc_base.adapt(data_dir = 'nmt/data/java2doc')

java2doc14 = java2doc_tree_base.adapt(struct = struct.tree1444)
java2doc15 = java2doc_tree_base.adapt(struct = struct.tree14442)
java2doc17 = java2doc_tree_base.adapt(struct = struct.tree)
java2doc17f = java2doc_tree_base.adapt(struct = struct.tree17f, grad_clamp = 100.0, data_dir = 'nmt/data/java2doc2')
java2doc18 = java2doc_tree_base.adapt(struct = struct.tree172)
java2doc_seq = java2doc_base.adapt(struct = struct.sequence)
java2doc_rare = java2doc_base.adapt(struct = struct.sequence)
java2doc_raw2 = java2doc_base.adapt(struct = struct.sequence)
java2doc_raw = java2doc_base.adapt(struct = struct.sequence)

java2doc_bpe = java2doc_base.adapt(struct = struct.tree, joint_vocab_size = 0)
java2doc_sbpe = java2doc_bpe.adapt(struct = struct.sequence)
java2doc_bpe_16000 = java2doc_bpe.adapt()
java2doc_bpe_32000 = java2doc_bpe.adapt()
java2doc_sbpe_16000 = java2doc_sbpe.adapt()
java2doc_sbpe_32000 = java2doc_sbpe.adapt()




#######################################################################

py2doc_base = second_base.adapt(
    src_lang = 'py',
    trg_lang = 'doc',
    save_to = 'nmt/py2doc_models/{model_name}',
    joint_vocab_size = 32000,
)

py2doc_tree_base = py2doc_base.adapt(data_dir = 'nmt/data/py2doc')
py2doc2_tree_base = py2doc_base.adapt(data_dir = 'nmt/data/py2doc2')

py2doc14 = py2doc2_tree_base.adapt(struct = struct.tree1444, grad_clamp = 100.0)
py2doc15 = py2doc2_tree_base.adapt(struct = struct.tree14442, grad_clamp = 100.0)
py2doc16 = py2doc2_tree_base.adapt(struct = struct.tree1445, grad_clamp = 100.0)
py2doc17 = py2doc_tree_base.adapt(struct = struct.tree)
py2doc18 = py2doc2_tree_base.adapt(struct = struct.tree172, grad_clamp = 100.0)
py2doc17f = py2doc2_tree_base.adapt(struct = struct.tree17f, grad_clamp = 100.0)
py2doc17f2 = py2doc2_tree_base.adapt(struct = struct.tree17f, grad_clamp = 1.0)
py2doc_seq = py2doc_base.adapt(struct = struct.sequence)
py2doc_rare = py2doc_base.adapt(struct = struct.sequence)
py2doc_rare2 = py2doc_base.adapt(struct = struct.sequence, data_dir = 'nmt/data/py2doc_rare2')

py2doc_bpe = py2doc_base.adapt(struct = struct.tree, joint_vocab_size = 0, grad_clamp = 100.0)
py2doc_sbpe = py2doc_bpe.adapt(struct = struct.sequence)
py2doc_bpe_16000 = py2doc_bpe.adapt()
py2doc_bpe_32000 = py2doc_bpe.adapt()
py2doc_sbpe_16000 = py2doc_sbpe.adapt()
py2doc_sbpe_32000 = py2doc_sbpe.adapt()



##########################

en2vi = base_config.adapt(src_lang = 'en', trg_lang = 'vi')
en2vi2 = base_config.adapt(src_lang = 'en', trg_lang = 'vi', struct = struct.sequence2)
