import os
import logging
import numpy
import torch
import random

import nmt.all_constants as ac

random.seed(ac.SEED)
numpy.random.seed(ac.SEED)
torch.manual_seed(ac.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ac.SEED)


def ensure_dirs_exists(filepath):
    "Creates the directories containing filepath, if it does not yet exist. Returns if the path already existed."
    parent = os.path.dirname(filepath)
    if not os.path.exists(parent):
        os.makedirs(parent)
        return False
    return True


def get_logger(logfile='./DEBUG.log'):
    "Initializes (if necessary) logger, then returns it"
    if logfile: ensure_dirs_exists(logfile)
    # Performance recommendations per
    # https://docs.python.org/3/howto/logging.html#optimization
    logging.logThreads = 0
    logging.logProcesses = 0
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)16s:%(lineno)4s | %(message)s')

    if not logger.handlers:
        if logfile is not None:
            debug_handler = logging.FileHandler(logfile)
            debug_handler.setFormatter(formatter)
            debug_handler.setLevel(logging.DEBUG)
        else:
            debug_handler = logging.NullHandler()
        logger.addHandler(debug_handler)

    return logger

def format_table_row(cells, column_widths=None, alignments=None):
    column_widths = column_widths or [len(cell) for cell in cells]
    alignments = alignments or ['left'] * len(cells)
    def pad(cell, width, align):
        pads = ' ' * (width - len(cell))
        if align == 'left': return pads + cell
        elif align == 'right': return cell + pads
        elif align == 'center': return pads[0::2] + cell + pads[1::2]
        else: return cell
    return '  '.join(pad(c, w, a) for c, w, a in zip(cell, columns_widths, alignments))

def reorder(xs, indices):
    'Reorders the elements in xs according to indices'
    if isinstance(xs, numpy.ndarray) or torch.is_tensor(xs):
        return xs[indices]
    else:
        return [xs[i] for i in indices]

def shuffle_indices(iter_or_int):
    'Returns a numpy.ndarray of randomly shuffled indices corresponding to iter_or_int'
    if not isinstance(iter_or_int, int):
        iter_or_int = len(iter_or_int)
    indices = numpy.arange(iter_or_int)
    numpy.random.shuffle(indices)
    return indices

def shuffle_parallel(*arrays):
    'Shuffles arrays in parallel'
    arrays = list(arrays)
    randomized_indices = shuffle_indices(len(arrays[0]))
    return tuple(reorder(array, randomized_indices) for array in arrays)


def shuffle_file(input_file):
    with open(input_file, 'r') as fh:
        data = [(random.random(), line) for line in fh]
    data.sort()
    with open(input_file, 'w') as fh:
        for _, line in data:
            fh.write(line)

def format_time(secs):
    "Formats secs as a nice, human-readable time (in hrs, mins, secs, ms when significant)"
    secs_exact = secs
    mins_exact = secs_exact / 60
    hrs_exact = mins_exact / 60
    secs_rounded = int((mins_exact % 1) * 60 + 0.5)
    ms_rounded = int((secs_exact % 1) * 1000 + 0.5)
    secs = int((mins_exact % 1) * 60)
    mins = int((hrs_exact % 1) * 60)
    hrs = int(hrs_exact)
    if hrs_exact >= 1: return f"{hrs}:{mins:02}:{secs_rounded:02}"
    elif mins_exact >= 1: return f"{mins}:{secs_rounded:02}"
    else: return f"{secs}.{ms_rounded:03}s"


def get_vocab_masks(config, src_vocab_size, trg_vocab_size):
    "Computes and returns [src_vocab_mask, trg_vocab_mask]"
    masks = []
    device = get_device()
    for vocab_size, lang in [(src_vocab_size, config['src_lang']), (trg_vocab_size, config['trg_lang'])]:
        if config['tie_mode'] == ac.ALL_TIED:
            mask = numpy.load(os.path.join(config['data_dir'], 'joint_vocab_mask.{}.npy'.format(lang)))
        else:
            mask = numpy.ones([vocab_size], numpy.float32)

        mask[ac.PAD_ID] = 0.
        mask[ac.BOS_ID] = 0.
        masks.append(torch.from_numpy(mask).type(torch.bool).to(device)) # bool for torch versions >= 1.2.0; uint8 for versions < 1.2.0

    return masks


def get_vocab_sizes(config):
    "Returns sizes of src_vocab, trg_vocab"
    def _get_vocab_size(vocab_file):
        vocab_size = 0
        with open(vocab_file) as f:
            for line in f:
                if line.strip():
                    vocab_size += 1
        return vocab_size

    src_vocab_file = os.path.join(config['data_dir'], 'vocab-{}.{}'.format(config['src_vocab_size'], config['src_lang']))
    trg_vocab_file = os.path.join(config['data_dir'], 'vocab-{}.{}'.format(config['trg_vocab_size'], config['trg_lang']))

    return _get_vocab_size(src_vocab_file), _get_vocab_size(trg_vocab_file)

position_encoding_cached = None
position_encoding_len_cached = None

def get_position_encoding(dim, sentence_length):
    "Returns sequence-to-sequence position encoding [sentence_length, dim]"
    global position_encoding_cached, position_encoding_len_cached
    if position_encoding_cached is not None and sentence_length <= position_encoding_len_cached:
        return position_encoding_cached[:sentence_length, :]
    else:
        div_term = numpy.power(10000.0, - (numpy.arange(dim) // 2).astype(numpy.float32) * 2.0 / dim)
        div_term = div_term.reshape(1, -1)
        pos = numpy.arange(sentence_length, dtype=numpy.float32).reshape(-1, 1)
        encoded_vec = numpy.matmul(pos, div_term)
        encoded_vec[:, 0::2] = numpy.sin(encoded_vec[:, 0::2])
        encoded_vec[:, 1::2] = numpy.cos(encoded_vec[:, 1::2])

        dtype = get_float_type()
        position_encoding_cached = torch.from_numpy(encoded_vec.reshape([sentence_length, dim])).type(dtype)
        position_encoding_len_cached = sentence_length
        return position_encoding_cached


def normalize(x, scale=True):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True) + 1e-6
    if scale:
        std = std * x.size()[-1] ** 0.5
    return (x - mean) / std


def gnmt_length_model(alpha):
    def f(time_step, prob):
        return prob / ((5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha)
    return f

def get_device():
    "Returns cuda:0 if available, or otherwise cpu"
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_float_type():
    "Chooses between torch.cuda.FloatTensor and torch.FloatTensor"
    return torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def get_num_digits(x):
    "Returns the number of digits needed to print positive integer x in base 10"
    return int(numpy.log10(x) + 1) if x else 1
