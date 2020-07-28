import time
import os
from os.path import join, exists, basename
from itertools import islice
from collections import Counter
import numpy
import torch
import shutil
import io

import nmt.utils as ut
import nmt.all_constants as ac

class DataManager(object):

    def __init__(self, config):
        super(DataManager, self).__init__()
        self.logger = ut.get_logger(config['log_file'])

        self.src_lang = config['src_lang']
        self.trg_lang = config['trg_lang']
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.beam_size = config['beam_size']
        self.one_embedding = config['tie_mode'] == ac.ALL_TIED
        self.share_vocab = config['share_vocab']
        self.word_dropout = config['word_dropout']
        self.batch_sort_src = config['batch_sort_src']
        self.max_src_length = config['max_src_length']
        self.max_trg_length = config['max_trg_length']
        self.parse_struct = config['struct'].parse

        self.vocab_sizes = {
            self.src_lang: config['src_vocab_size'],
            self.trg_lang: config['trg_vocab_size'],
            'joint': config['joint_vocab_size']
        }

        self.data_files = {
            ac.TRAINING: {
                self.src_lang: join(self.data_dir, 'train.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'train.{}'.format(self.trg_lang))
            },
            ac.VALIDATING: {
                self.src_lang: join(self.data_dir, 'dev.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'dev.{}'.format(self.trg_lang))
            },
            ac.TESTING: {
                self.src_lang: join(self.data_dir, 'test.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'test.{}'.format(self.trg_lang))
            }
        }
        self.tok_count_files = {
            ac.TRAINING: join(self.data_dir, 'train.count'),
            ac.VALIDATING: join(self.data_dir, 'dev.count'),
            ac.TESTING: join(self.data_dir, 'test.count')
        }
        self.ids_files = {
            ac.TRAINING: join(self.data_dir, 'train.ids'),
            ac.VALIDATING: join(self.data_dir, 'dev.ids'),
            ac.TESTING: join(self.data_dir, 'test.ids')
        }
        self.vocab_files = {
            self.src_lang: join(self.data_dir, 'vocab-{}.{}'.format(self.vocab_sizes[self.src_lang], self.src_lang)),
            self.trg_lang: join(self.data_dir, 'vocab-{}.{}'.format(self.vocab_sizes[self.trg_lang], self.trg_lang))
        }

        train_ids_file = join(config['save_to'], 'train.ids')
        self.setup(train_ids_file)

    def setup(self, train_ids_file):
        self.create_all_vocabs()
        # it's important to check if we use one vocab before converting data to token ids
        self.parallel_data_to_token_ids(mode=ac.TRAINING)
        self.parallel_data_to_token_ids(mode=ac.VALIDATING)

        if exists(self.data_files[ac.TESTING][self.src_lang]) and exists(self.data_files[ac.TESTING][self.trg_lang]):
            self.parallel_data_to_token_ids(mode=ac.TESTING)

        # We don't want to modify anything in data_dir/ during
        # training, so copy data_dir/train.ids to save_to/train.ids
        # so we can shuffle it without modifying it in self.data_dir
        shutil.copy(self.ids_files[ac.TRAINING], train_ids_file)
        
        self.ids_files[ac.TRAINING] = train_ids_file


    def parse_line(self, line, is_src, max_len=None, to_ids=False):
        max_lang_len = self.max_src_length if is_src else self.max_trg_length
        max_len = (max_len or max_lang_len) - bool(to_ids) # append EOS if to_ids, so -1 to max len
        if is_src:
            s = self.parse_struct(line, clip=max_len)
            if to_ids:
                s = s.map(lambda w: self.src_vocab.get(w, ac.UNK_ID))
                s.maybe_add_eos(ac.EOS_ID)
        else:
            s = line.strip().split(maxsplit=max_len)[:max_len]
            if to_ids:
                s = [ac.BOS_ID] + [self.trg_vocab.get(w, ac.UNK_ID) for w in s]
        return s


    ############## Vocab Functions ##############

    def create_all_vocabs(self):
        self.create_vocabs()
        self.create_joint_vocab()
        self.src_vocab, self.src_ivocab, self.src_w2f = self.init_vocab(self.src_lang)
        self.trg_vocab, self.trg_ivocab, self.trg_w2f = self.init_vocab(self.trg_lang)

    def create_vocabs(self):
        def _write_vocab_file(vocab_dict, max_vocab_size, vocab_file):
            vocab_list = ac._START_VOCAB + sorted(vocab_dict, key=vocab_dict.get, reverse=True)
            if max_vocab_size == 0:
                pass
            elif len(vocab_list) < max_vocab_size:
                msg = '    The actual vocab size {} is < than your required {}\n'.format(
                    len(vocab_list), max_vocab_size)
                msg += '    Due to shameful reason, this cannot be handled automatically\n'
                msg += '    Please change the vocab size for {} to {}'.format(vocab_file, len(vocab_list))
                self.logger.info(msg)
                raise ValueError(msg)
            else:
                vocab_list = vocab_list[:max_vocab_size]

            open(vocab_file, 'w').close()
            with open(vocab_file, 'w') as fout:
                for idx, w in enumerate(vocab_list):
                    tok_freq = vocab_dict.get(w, 0)
                    fout.write(u'{} {} {}\n'.format(w, idx, tok_freq))

        src_file = self.data_files[ac.TRAINING][self.src_lang]
        src_vocab_file = self.vocab_files[self.src_lang]
        max_src_vocab_size = self.vocab_sizes[self.src_lang]

        trg_file = self.data_files[ac.TRAINING][self.trg_lang]
        trg_vocab_file = self.vocab_files[self.trg_lang]
        max_trg_vocab_size = self.vocab_sizes[self.trg_lang]

        self.logger.info('Create vocabulary of size {}/{}, {}/{}'.format(self.src_lang, max_src_vocab_size, self.trg_lang, max_trg_vocab_size))
        if exists(src_vocab_file) and exists(trg_vocab_file):
            self.logger.info('    Vocab files exist at {}/{}'.format(src_vocab_file, trg_vocab_file))
        else:
            src_vocab = Counter()
            trg_vocab = Counter()
            with open(src_file, 'r') as src_f, open(trg_file, 'r') as trg_f:
                count = 0
                for src_line, trg_line in zip(src_f, trg_f):
                    count += 1
                    if count % 10000 == 0:
                        self.logger.info('    processing line {}'.format(count))
                    src_line_parsed = self.parse_line(src_line, True)
                    src_line_words = src_line_parsed.flatten()
                    trg_line_words = self.parse_line(trg_line, False)
                    src_vocab.update(src_line_words)
                    trg_vocab.update(trg_line_words)

            _write_vocab_file(src_vocab, max_src_vocab_size, src_vocab_file)
            _write_vocab_file(trg_vocab, max_trg_vocab_size, trg_vocab_file)

    def create_joint_vocab(self):
        if not self.one_embedding:
            return

        self.logger.info('Using one embedding so use one joint vocab')

        joint_vocab_file = join(self.data_dir, 'joint_vocab.{}-{}'.format(self.src_lang, self.trg_lang))
        subjoint_src_vocab_file = join(self.data_dir, 'joint_vocab.{}'.format(self.src_lang))
        subjoint_trg_vocab_file = join(self.data_dir, 'joint_vocab.{}'.format(self.trg_lang))

        # If all vocab files exist, return
        if exists(joint_vocab_file) and exists(subjoint_src_vocab_file) and exists(subjoint_trg_vocab_file):
            # Make sure to replace the vocab_files with the subjoint_vocab_files
            self.vocab_files[self.src_lang] = subjoint_src_vocab_file
            self.vocab_files[self.trg_lang] = subjoint_trg_vocab_file
            self.logger.info("Joint vocab files already exist")
            return

        # Else, first combine the two word2freq from src + trg
        _, _, src_word2freq = self.init_vocab(self.src_lang)
        _, _, trg_word2freq = self.init_vocab(self.trg_lang)
        for special_tok in ac._START_VOCAB:
            del src_word2freq[special_tok]
            del trg_word2freq[special_tok]

        joint_word2freq = dict(Counter(src_word2freq) + Counter(trg_word2freq))

        # Save the joint vocab files
        joint_vocab_list = ac._START_VOCAB + sorted(joint_word2freq, key=joint_word2freq.get, reverse=True)
        if self.vocab_sizes['joint'] != 0 and len(joint_vocab_list) > self.vocab_sizes['joint']:
            self.logger.info('Cut off joint vocab size from {} to {}'.format(len(joint_vocab_list), self.vocab_sizes['joint']))
            joint_vocab_list = joint_vocab_list[:self.vocab_sizes['joint']]

        open(joint_vocab_file, 'w').close()
        with open(joint_vocab_file, 'w') as fout:
            for idx, w in enumerate(joint_vocab_list):
                tok_freq = joint_word2freq.get(w, 0)
                fout.write(u'{} {} {}\n'.format(w, idx, tok_freq))

        joint_vocab, _, _ = self.init_vocab('', joint_vocab_file)
        self.logger.info('Joint vocab has {} keys'.format(len(joint_vocab)))
        self.logger.info('Joint vocab saved in {}'.format(joint_vocab_file))

        if self.share_vocab:
            for lang in [self.src_lang, self.trg_lang]:
                self.vocab_files[lang] = joint_vocab_file
                vocab_mask = numpy.ones([len(joint_vocab)], dtype=numpy.float32)
                vocab_mask_file = join(self.data_dir, 'joint_vocab_mask.{}.npy'.format(lang))
                numpy.save(vocab_mask_file, vocab_mask)

            self.logger.info('Use the same vocab for both')
            return

        # Now generate the separate subjoint vocab, i.e. words that appear in both corresponding language's training
        # data and joint vocab
        # Also generate vocab mask
        for (lang, org_vocab, vocab_file) in [(self.src_lang, src_word2freq, subjoint_src_vocab_file), (self.trg_lang, trg_word2freq, subjoint_trg_vocab_file)]:
            self.logger.info('Creating subjoint vocab file for {} at {}'.format(lang, vocab_file))
            new_vocab_list = ac._START_VOCAB[::] + [word for word in joint_vocab_list if word in org_vocab]

            open(vocab_file, 'w').close()
            with open(vocab_file, 'w') as fout:
                for word in new_vocab_list:
                    fout.write(u'{} {} {}\n'.format(word, joint_vocab[word], joint_word2freq.get(word, 0)))

            self.vocab_files[lang] = vocab_file
            vocab_mask = numpy.zeros([len(joint_vocab)], dtype=numpy.float32)
            for word in new_vocab_list:
                vocab_mask[joint_vocab[word]] = 1.0
            vocab_mask_file = join(self.data_dir, 'joint_vocab_mask.{}.npy'.format(lang))
            numpy.save(vocab_mask_file, vocab_mask)
            self.logger.info('Save joint vocab mask for {} to {}'.format(lang, vocab_mask_file))

    def init_vocab(self, lang=None, fromfile=None):
        if not lang and not fromfile:
            raise ValueError('Must provide either src/trg lang or fromfile')
        elif fromfile:
            vocab_file = fromfile
        else:
            vocab_file = self.vocab_files[lang]

        self.logger.info('Initialize {} vocab from {}'.format(lang, vocab_file))

        if not exists(vocab_file):
            raise ValueError('    Vocab file {} not found'.format(vocab_file))

        counter = 0 # fall back to old vocab format "word freq\n"
        vocab, ivocab, word2freq = {}, {}, {}
        with open(vocab_file, 'r') as f:
            for line in f:
                if line.strip():
                    wif = line.strip().split()
                    if len(wif) == 2:
                        word = wif[0]
                        freq = int(wif[1])
                        idx = counter
                        counter += 1
                    elif len(wif) == 3:
                        word = wif[0]
                        idx = int(wif[1])
                        freq = int(wif[2])
                    else:
                        raise ValueError('Something wrong with this vocab format')

                    vocab[word] = idx
                    ivocab[idx] = word
                    word2freq[word] = freq

        return vocab, ivocab, word2freq

    def parallel_data_to_token_ids(self, mode=ac.TRAINING):
        src_file = self.data_files[mode][self.src_lang]
        trg_file = self.data_files[mode][self.trg_lang]

        joint_file = self.ids_files[mode]
        joint_tok_count = self.tok_count_files[mode]

        if exists(joint_file) and exists(joint_tok_count):
            self.logger.info('    Token-id-ed data exists at {}'.format(joint_file))
            return

        self.logger.info('Parallel convert tokens from {} & {} to ids and save to {}'.format(src_file, trg_file, joint_file))
        self.logger.info('Also save the approx tok count to {}'.format(joint_tok_count))

        open(joint_file, 'w').close()
        open(joint_tok_count, 'w').close()
        num_lines = 0
        src_tok_count = 0
        trg_tok_count = 0
        with open(src_file, 'r') as src_f, \
                open(trg_file, 'r') as trg_f, \
                open(joint_file, 'w') as tokens_f:

            for src_line, trg_line in zip(src_f, trg_f):
                src_prsd = self.parse_line(src_line, True, to_ids=True)
                trg_ids = self.parse_line(trg_line, False, to_ids=True)

                if 0 < src_prsd.size() and 1 < len(trg_ids):
                    num_lines += 1
                    if num_lines % 10000 == 0:
                        self.logger.info('    converting line {}'.format(num_lines))
                    src_ids = src_prsd.flatten()
                    src_tok_count += len(src_ids)
                    trg_tok_count += len(trg_ids)
                    data = u'{}|||{}\n'.format(str(src_prsd), u' '.join(map(str, trg_ids)))
                    tokens_f.write(data)

        with open(joint_tok_count, 'w') as f:
            f.write('{} {}\n'.format(str(src_tok_count), str(trg_tok_count)))
        




    ############## Batch Functions ##############

    def replace_with_unk(self, data):
        drop_mask = numpy.random.choice([True, False], data.shape, p=[self.word_dropout, 1.0 - self.word_dropout])
        drop_mask = numpy.logical_and(drop_mask, data != ac.PAD_ID)
        data[drop_mask] = ac.UNK_ID

    def _prepare_one_batch(self, b_src_input, b_src_seq_length, b_src_structs, b_trg_input, b_trg_seq_length, with_trg=True):

        batch_size = len(b_src_input)
        max_src_length = max(b_src_seq_length)
        max_trg_length = max(b_trg_seq_length) if with_trg else 0

        src_input_batch = numpy.zeros([batch_size, max_src_length], dtype=numpy.int32)
        trg_input_batch = numpy.zeros([batch_size, max_trg_length], dtype=numpy.int32)
        trg_target_batch = numpy.zeros([batch_size, max_trg_length], dtype=numpy.int32)

        for i in range(batch_size):
            src_input_batch[i] = list(b_src_input[i]) + (max_src_length - b_src_seq_length[i]) * [ac.PAD_ID]
            if with_trg:
                trg_input_batch[i] = list(b_trg_input[i]) + (max_trg_length - b_trg_seq_length[i]) * [ac.PAD_ID]
                trg_target_batch[i] = list(b_trg_input[i][1:]) + [ac.EOS_ID] + (max_trg_length - b_trg_seq_length[i]) * [ac.PAD_ID]
            else:
                trg_input_batch[i] = []
                trg_target_batch[i] = []

        return src_input_batch, b_src_structs, trg_input_batch, trg_target_batch

    def prepare_batches(self, src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths, mode=ac.TRAINING, with_trg=True):

        # Sorting by src lengths
        # https://www.aclweb.org/anthology/W17-3203
        sorted_idxs = numpy.argsort(src_seq_lengths if self.batch_sort_src or not trg_inputs else trg_seq_lengths)
        src_inputs = src_inputs[sorted_idxs]
        src_seq_lengths = src_seq_lengths[sorted_idxs]
        src_structs = src_structs[sorted_idxs]
        trg_inputs = trg_inputs[sorted_idxs] if with_trg else []
        trg_seq_lengths = trg_seq_lengths[sorted_idxs] if with_trg else []

        src_input_batches = []
        src_structs_batches = []
        trg_input_batches = []
        trg_target_batches = []
        idxs_batches = []

        est_src_trg_ratio = (1, 1) if with_trg else self.read_tok_count()
        est_src_trg_ratio = est_src_trg_ratio[0] / sum(est_src_trg_ratio)

        s_idx = 0
        while s_idx < len(src_inputs):
            e_idx = s_idx + 1
            max_src_in_batch = src_seq_lengths[s_idx]
            max_trg_in_batch = with_trg and trg_seq_lengths[s_idx]
            while e_idx < len(src_inputs):
                max_src_in_batch = max(max_src_in_batch, src_seq_lengths[e_idx])
                if with_trg: max_trg_in_batch = max(max_trg_in_batch, trg_seq_lengths[e_idx])
                else: max_trg_in_batch = round(max_src_in_batch * est_src_trg_ratio)
                count = (e_idx - s_idx + 1) * max_src_in_batch #(max_src_in_batch + max_trg_in_batch)
                if count > self.batch_size: break
                else: e_idx += 1

            idxs_batch = sorted_idxs[s_idx:e_idx]
            batch_values = self._prepare_one_batch(
                src_inputs[s_idx:e_idx],
                src_seq_lengths[s_idx:e_idx],
                src_structs[s_idx:e_idx],
                trg_inputs[s_idx:e_idx],
                trg_seq_lengths[s_idx:e_idx],
                with_trg=with_trg)
            src_input_batch, src_structs_batch, trg_input_batch, trg_target_batch = batch_values

            if mode == ac.TRAINING:
                self.replace_with_unk(src_input_batch)
                if with_trg: self.replace_with_unk(trg_input_batch)

            # Make sure only the structure of src is used
            # (some toks may have been replaced with UNK)
            src_structs_batch = [struct.forget() for struct in src_structs_batch]
            
            s_idx = e_idx
            idxs_batches.append(idxs_batch)
            src_input_batches.append(src_input_batch)
            src_structs_batches.append(src_structs_batch)
            trg_input_batches.append(trg_input_batch)
            trg_target_batches.append(trg_target_batch)

        batches = idxs_batches, src_input_batches, src_structs_batches, trg_input_batches, trg_target_batches
        
        if mode == ac.TRAINING:
            batches = ut.shuffle_parallel(*batches)

        return batches

    def process_n_batches(self, n_batches_string_list, to_ids=False, with_trg=True):
        src_inputs = []
        src_seq_lengths = []
        src_structs = []
        trg_inputs = []
        trg_seq_lengths = []

        for line in n_batches_string_list:
            data = line.strip()
            if data:
                _src_data, _trg_data = data.split('|||') if with_trg else [data, None]
                _src_struct = self.parse_line(_src_data, True, to_ids=to_ids)
                if not to_ids: _src_struct = _src_struct.map(int)
                _src_toks = _src_struct.flatten()
                _src_len = len(_src_toks)
                src_inputs.append(_src_toks)
                src_seq_lengths.append(_src_len)
                src_structs.append(_src_struct)

                if with_trg:
                    _trg_toks = self.parse_line(_trg_data, False, to_ids=to_ids)
                    if not to_ids: _trg_toks = [int(x) for x in _trg_toks]
                else:
                    _trg_toks = []
                _trg_len = len(_trg_toks)
                trg_inputs.append(_trg_toks)
                trg_seq_lengths.append(_trg_len)

        # convert to numpy array for sorting & reindexing
        src_inputs = numpy.array(src_inputs)
        src_seq_lengths = numpy.array(src_seq_lengths)
        src_structs = numpy.array(src_structs)
        trg_inputs = numpy.array(trg_inputs) if trg_inputs else None
        trg_seq_lengths = numpy.array(trg_seq_lengths) if trg_seq_lengths else None

        return src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths

    def read_batches(self, read_handler, mode=ac.TRAINING, num_preload=1000, to_ids=False, with_trg=True):
        device = ut.get_device()
        while True:
            next_n_lines = list(islice(read_handler, num_preload))
            if not next_n_lines: break
            src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths = self.process_n_batches(next_n_lines, to_ids=to_ids, with_trg=with_trg)
            batches = self.prepare_batches(src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths, mode=mode, with_trg=with_trg)
            for original_idxs, src_inputs, src_structs, trg_inputs, trg_target in zip(*batches):
                yield (original_idxs,
                       torch.from_numpy(src_inputs).type(torch.long).to(device),
                       src_structs,
                       torch.from_numpy(trg_inputs).type(torch.long).to(device),
                       torch.from_numpy(trg_target).type(torch.long).to(device))

    def get_batches(self, mode=ac.TRAINING, num_preload=1000):
        ids_file = self.ids_files[mode]
        if mode == ac.TRAINING:
            # Shuffle training dataset
            start = time.time()
            ut.shuffle_file(ids_file)
            self.logger.info('Shuffling {} took {}'.format(ids_file, ut.format_time(time.time() - start)))

        with open(ids_file, 'r') as f:
            yield from self.read_batches(f, mode, num_preload, to_ids=False, with_trg=True)

    def _ids_to_trans(self, trans_ids):
        words = []
        for idx in trans_ids:
            if idx == ac.EOS_ID: break
            words.append(self.trg_ivocab[idx])
        return u' '.join(words)

    def detach_outputs(self, rets):
        for ret in rets:
            yield (ret['probs'].detach().cpu().numpy().reshape([-1]),
                   ret['scores'].detach().cpu().numpy().reshape([-1]),
                   ret['symbols'].detach().cpu().numpy())

    def get_trans(self, probs, scores, symbols):
        sorted_rows = numpy.argsort(scores)[::-1]
        best_trans = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            trans_out = self._ids_to_trans(symbols[r])
            beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r], probs[r]))
            if i == 0: # highest prob trans
                best_trans = trans_out
        return best_trans, u'\n'.join(beam_trans)

    def translate(self, model, input_file_or_stream, best_output_stream, beam_output_stream, mode=ac.VALIDATING, num_preload=1000, to_ids=False):
        model.eval()

        input_file = not isinstance(input_file_or_stream, io.IOBase) and input_file_or_stream
        input_stream = open(input_file_or_stream, 'r') if input_file else input_file_or_stream
        
        if input_file:
            with open(input_file, 'r') as f:
                num_sents = sum(bool(line.strip()) for line in f)
            num_sents_digits = (4 * ut.get_num_digits(num_sents) - 1) // 3 # factor in thousands-separator
        
        with torch.no_grad():
            if input_file:
                self.logger.info('Start translating {}'.format(input_file))
                start = last = time.time()
                notify_every = 1000
            count = 0
            best_trans_cache = [None] * num_preload
            beam_trans_cache = [None] * num_preload
            for idxs, src_toks, src_structs, _, _ in self.read_batches(input_stream, mode, num_preload, to_ids, with_trg=False):
                rets = self.detach_outputs(model.beam_decode(src_toks, src_structs))
                for i, ret in enumerate(rets):
                    i2 = idxs[i]
                    best, beam = self.get_trans(*ret)
                    if best_output_stream: best_trans_cache[i2] = best
                    if beam_output_stream: beam_trans_cache[i2] = beam
                    count += 1
                    if count % num_preload == 0:
                        if best_output_stream:
                            best_output_stream.write('\n'.join(best_trans_cache) + '\n')
                            best_trans_cache = [None] * num_preload
                        if beam_output_stream:
                            beam_output_stream.write('\n\n'.join(beam_trans_cache) + '\n\n')
                            beam_trans_cache = [None] * num_preload
                    if input_file and count % notify_every == 0:
                        now = time.time()
                        self.logger.info('  Line {:>{},} / {:,}, {:.4f} sec/line'.format(count, num_sents_digits, num_sents, (time.time() - last) / notify_every))
                        last = now

            remaining = (count - 1) % num_preload
            if best_output_stream:
                best_output_stream.write('\n'.join(best_trans_cache[:remaining]))
            if beam_output_stream:
                beam_output_stream.write('\n\n'.join(beam_trans_cache[:remaining]))

        if input_file:
            input_stream.close()
            self.logger.info('Finished translating {}, took {}'.format(input_file, ut.format_time(time.time() - start)))
        model.train()
        
    #def translate_line(self, model, line):
    #    "Translates a single line of text, with no file I/O"
    #    struct = self.parse_line(line, True)
    #    struct = struct.map(lambda w: self.src_vocab.get(w, ac.UNK_ID))
    #    toks = torch.tensor(struct.flatten()).type(torch.long).to(ut.get_device()).unsqueeze(0)
    #    trans, = self.translate_batch(model, toks, [struct])
    #    return trans
    #
    #def translate_batch(self, model, toks, structs):
    #    for x in self.detach_outputs(model.beam_decode(toks, structs)):
    #        yield self.get_trans(*x)[0]

    def read_tok_count(self, mode=ac.TRAINING):
        fp = self.tok_count_files[mode]
        src, trg = -1, -1
        try:
            with open(fp, 'r') as fh:
                src, trg = fh.read().strip().split()
                src, trg = int(src), int(trg)
        except:
            self.logger.error('Error reading token count from {}'.format(fp))
        finally:
            return src, trg
