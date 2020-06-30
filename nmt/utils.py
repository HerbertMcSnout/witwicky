import os, os.path
import logging
from datetime import timedelta
import subprocess

import numpy
import torch

import nmt.all_constants as ac

def ensure_dirs_exists(filepath):
    "Creates the directories containing filepath, if it does not yet exist. Returns if the path already existed."
    parent = os.path.dirname(filepath)
    if not os.path.exists(parent):
        os.makedirs(parent)
        return False
    return True


def get_logger(logfile=None):
    """Global logger for every logging"""
    _logfile = logfile if logfile else './DEBUG.log'
    ensure_dirs_exists(_logfile)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)16s:%(lineno)4s: %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def shuffle_file(input_file):
    shuffled_file = input_file + '.shuf'
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    commands = 'bash {}/../scripts/shuffle_file.sh {} {}'.format(scriptdir, input_file, shuffled_file)
    subprocess.check_call(commands, shell=True)
    subprocess.check_call('mv {} {}'.format(shuffled_file, input_file), shell=True)


def get_validation_frequency(train_length_file, val_frequency, batch_size):
    with open(train_length_file) as f:
        line = f.readline().strip()
        num_train_toks = int(line)

    return int(num_train_toks * val_frequency / batch_size)


def format_seconds(seconds):
    return str(timedelta(seconds=seconds))

def format_time(secs):
    ms_exact = secs * 1000
    secs_exact = secs
    mins_exact = secs_exact / 60
    hrs_exact = mins_exact / 60
    ms = int((secs_exact % 1) * 1000)
    secs = int((mins_exact % 1) * 60)
    mins = int((hrs_exact % 1) * 60)
    hrs = int(hrs_exact)
    if hrs_exact >= 1: return f"{hrs}:{mins}:{secs}"
    elif mins_exact >= 1: return f"{mins}:{secs}.{ms:.0f}"
    elif secs_exact >= 1: return f"{secs}.{ms:03.0f}s"
    else: return f"{ms}ms"


def get_vocab_masks(config, src_vocab_size, trg_vocab_size):
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

positional_encoding = None

def set_positional_encoding(dim, max_len):
    global positional_encoding
    if positional_encoding is None:
        positional_encoding = get_positional_encoding(dim, max_len)
        return positional_encoding

def get_positional_encoding(dim, sentence_length):
    if positional_encoding is not None:
        return positional_encoding[:sentence_length, :]
    else:
        div_term = numpy.power(10000.0, - (numpy.arange(dim) // 2).astype(numpy.float32) * 2.0 / dim)
        div_term = div_term.reshape(1, -1)
        pos = numpy.arange(sentence_length, dtype=numpy.float32).reshape(-1, 1)
        encoded_vec = numpy.matmul(pos, div_term)
        encoded_vec[:, 0::2] = numpy.sin(encoded_vec[:, 0::2])
        encoded_vec[:, 1::2] = numpy.cos(encoded_vec[:, 1::2])

        dtype = get_float_type()
        return torch.from_numpy(encoded_vec.reshape([sentence_length, dim])).type(dtype)


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
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_float_type():
    return torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
