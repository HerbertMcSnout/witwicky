import time
import os
import itertools
from collections import Counter
import numpy
import torch
import shutil
import io

import nmt.utils as ut
import nmt.all_constants as ac

class DataManager(object):

    def __init__(self, config, init_vocab=True):
        super(DataManager, self).__init__()
        self.logger = ut.get_logger(config['log_file'])
        self.src_lang = config['src_lang']
        self.trg_lang = config['trg_lang']
        self.data_dir = config['data_dir']
        self.save_to = config['save_to']
        self.batch_size = config['batch_size']
        self.one_embedding = config['tie_mode'] == ac.ALL_TIED
        self.share_vocab = config['share_vocab']
        self.word_dropout = config['word_dropout']
        self.batch_sort_src = config['batch_sort_src']
        self.max_src_length = config['max_src_length']
        self.max_trg_length = config['max_trg_length']
        self.parse_struct = config['struct'].parse
        self.training_tok_counts = (-1, -1)
        self.vocab_masks = {}

        self.vocab_sizes = {
            self.src_lang: config['src_vocab_size'],
            self.trg_lang: config['trg_vocab_size'],
            'joint': config['joint_vocab_size']
        }
        
        if init_vocab:
            self.setup()
        else:
            self.data_files = None
            self.ids_files = None

    ############## Vocab Functions ##############

    def setup(self):
        self.data_files = {
            mode: {lang: os.path.join(self.data_dir, '{}.{}'.format(ut.get_mode_name(mode), lang))
                   for lang in [self.src_lang, self.trg_lang]}
            for mode in [ac.TRAINING, ac.VALIDATING, ac.TESTING]
        }
        self.ids_files = {
            mode: os.path.join(self.save_to, '{}.ids'.format(ut.get_mode_name(mode)))
            for mode in [ac.TRAINING, ac.VALIDATING, ac.TESTING]
        }
        self.create_vocabs()
        self.parallel_data_to_token_ids(mode=ac.TRAINING)
        self.parallel_data_to_token_ids(mode=ac.VALIDATING)
        if os.path.exists(self.data_files[ac.TESTING][self.src_lang]) and os.path.exists(self.data_files[ac.TESTING][self.trg_lang]):
            self.parallel_data_to_token_ids(mode=ac.TESTING)

    def create_vocabs(self):
        src_file = self.data_files[ac.TRAINING][self.src_lang]
        max_src_vocab_size = self.vocab_sizes[self.src_lang]
        src_tok_count = 0

        trg_file = self.data_files[ac.TRAINING][self.trg_lang]
        max_trg_vocab_size = self.vocab_sizes[self.trg_lang]
        trg_tok_count = 0

        self.logger.info('Computing vocab from training data')
        src_vocab = Counter()
        trg_vocab = Counter()
        with open(src_file, 'r') as src_f, open(trg_file, 'r') as trg_f:
            count = 0
            for src_line, trg_line in zip(src_f, trg_f):
                count += 1
                if count % 10000 == 0:
                    self.logger.info('  processing line {}'.format(count))
                src_line_parsed = self.parse_line(src_line, is_src=True)
                src_line_words = src_line_parsed.flatten()
                trg_line_words = self.parse_line(trg_line, is_src=False)
                src_vocab.update(src_line_words)
                trg_vocab.update(trg_line_words)
                src_tok_count += len(src_line_words)
                trg_tok_count += len(trg_line_words)

        self.training_tok_counts = (src_tok_count, trg_tok_count)
        if self.one_embedding:
            self.logger.info('Using one embedding so use joint vocab')
            joint_vocab = src_vocab + trg_vocab
            size = self.vocab_sizes['joint'] or None
            joint_vocab, joint_ivocab = self.clip_vocab(joint_vocab, size)
            self.src_vocab = joint_vocab
            self.trg_vocab = joint_vocab
            self.src_ivocab = joint_ivocab
            self.trg_ivocab = joint_ivocab
            self.vocab_masks[self.src_lang] = ut.process_mask(self.get_mask(self.src_vocab, joint_vocab))
            self.vocab_masks[self.trg_lang] = ut.process_mask(self.get_mask(self.trg_vocab, joint_vocab))
        else:
            self.src_vocab, self.src_ivocab = self.clip_vocab(src_vocab, self.vocab_sizes[self.src_lang])
            self.trg_vocab, self.trg_ivocab = self.clip_vocab(trg_vocab, self.vocab_sizes[self.trg_lang])
            self.vocab_masks[self.src_lang] = ut.process_mask(numpy.ones([len(self.src_vocab)], dtype=numpy.float32))
            self.vocab_masks[self.trg_lang] = ut.process_mask(numpy.ones([len(self.trg_vocab)], dtype=numpy.float32))        

    def clip_vocab(self, vocab, size):
        size = size - len(ac._START_VOCAB) if size else None
        words = [k for k in ac._START_VOCAB] + [k for k, v in vocab.most_common(size) if v]
        return {k:v for v, k in enumerate(words)}, words

    def get_mask(self, vocab, joint_vocab):
        if self.share_vocab:
            return numpy.ones([len(joint_vocab)], dtype=numpy.float32)
        else:
            mask = numpy.zeros([len(joint_vocab)], dtype=numpy.float32)
            for word in vocab:
                mask[joint_vocab[word]] = 1.0
            return mask
        

    def parallel_data_to_token_ids(self, mode=ac.TRAINING):
        src_file = self.data_files[mode][self.src_lang]
        trg_file = self.data_files[mode][self.trg_lang]
        joint_file = self.ids_files[mode]

        self.logger.info('Converting {} data to ids'.format(ut.get_mode_name(mode)))
        open(joint_file, 'w').close()
        num_lines = 0
        src_tok_count = 0
        trg_tok_count = 0
        with open(src_file, 'r') as src_f, \
             open(trg_file, 'r') as trg_f, \
             open(joint_file, 'w') as tokens_f:

            for src_line, trg_line in zip(src_f, trg_f):
                src_prsd = self.parse_line(src_line, is_src=True, to_ids=True)
                trg_ids = self.parse_line(trg_line, is_src=False, to_ids=True)

                if 0 < src_prsd.size() and 1 < len(trg_ids):
                    num_lines += 1
                    if num_lines % 10000 == 0:
                        self.logger.info('  converting line {}'.format(num_lines))
                    src_ids = src_prsd.flatten()
                    data = u'{}|||{}\n'.format(str(src_prsd), u' '.join(map(str, trg_ids)))
                    tokens_f.write(data)



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

    def prepare_batches(self, src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths, is_training=True, with_trg=True):

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

        est_src_trg_ratio = self.training_tok_counts[0] / sum(self.training_tok_counts)

        s_idx = 0
        while s_idx < len(src_inputs):
            e_idx = s_idx + 1
            max_src_in_batch = src_seq_lengths[s_idx]
            max_trg_in_batch = with_trg and trg_seq_lengths[s_idx]
            while e_idx < len(src_inputs):
                max_src_in_batch = max(max_src_in_batch, src_seq_lengths[e_idx])
                if with_trg: max_trg_in_batch = max(max_trg_in_batch, trg_seq_lengths[e_idx])
                else: max_trg_in_batch = round(max_src_in_batch * est_src_trg_ratio)
                count = (e_idx - s_idx + 1) * max(max_src_in_batch, max_trg_in_batch)
                #count = (e_idx - s_idx + 1) * (max_src_in_batch + max_trg_in_batch)
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

            if is_training:
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
        
        if is_training:
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
                _src_struct = self.parse_line(_src_data, is_src=True, to_ids=to_ids)
                if not to_ids: _src_struct = _src_struct.map(int)
                _src_toks = _src_struct.flatten()
                _src_len = len(_src_toks)
                src_inputs.append(_src_toks)
                src_seq_lengths.append(_src_len)
                src_structs.append(_src_struct)

                if with_trg:
                    _trg_toks = self.parse_line(_trg_data, is_src=False, to_ids=to_ids)
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

    def read_batches(self, read_handler, is_training=True, num_preload=ac.DEFAULT_NUM_PRELOAD, to_ids=False, with_trg=True):
        device = ut.get_device()
        while True:
            next_n_lines = list(itertools.islice(read_handler, num_preload))
            if not next_n_lines: break
            src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths = self.process_n_batches(next_n_lines, to_ids=to_ids, with_trg=with_trg)
            batches = self.prepare_batches(src_inputs, src_seq_lengths, src_structs, trg_inputs, trg_seq_lengths, is_training=is_training, with_trg=with_trg)
            for original_idxs, src_inputs, src_structs, trg_inputs, trg_target in zip(*batches):
                yield (original_idxs,
                       torch.from_numpy(src_inputs).type(torch.long).to(device),
                       src_structs,
                       torch.from_numpy(trg_inputs).type(torch.long).to(device),
                       torch.from_numpy(trg_target).type(torch.long).to(device))

    def get_batches(self, mode=ac.TRAINING, num_preload=ac.DEFAULT_NUM_PRELOAD):
        ids_file = self.ids_files[mode]
        is_training = mode == ac.TRAINING
        if is_training:
            # Shuffle training dataset
            start = time.time()
            ut.shuffle_file(ids_file)
            self.logger.info('Shuffling {} took {}'.format(ids_file, ut.format_time(time.time() - start)))

        with open(ids_file, 'r') as f:
            yield from self.read_batches(f, is_training, num_preload, to_ids=False, with_trg=True)

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

    def translate(self, model, input_file_or_stream, best_output_stream, beam_output_stream, num_preload=ac.DEFAULT_NUM_PRELOAD, to_ids=False):
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
            for idxs, src_toks, src_structs, _, _ in self.read_batches(input_stream, False, num_preload, to_ids, with_trg=False):
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


    def load_state_dict(self, state_dict):
        self.src_vocab = state_dict['src_vocab']
        self.src_ivocab = state_dict['src_ivocab']
        self.trg_vocab = state_dict['trg_vocab']
        self.trg_ivocab = state_dict['trg_ivocab']
        self.vocab_masks = state_dict['masks']
        self.training_tok_counts = state_dict['training_tok_counts']

    def state_dict(self):
        return {
            'src_vocab':self.src_vocab,
            'src_ivocab':self.src_ivocab,
            'trg_vocab':self.trg_vocab,
            'trg_ivocab':self.trg_ivocab,
            'masks':self.vocab_masks,
            'training_tok_counts':self.training_tok_counts,
        }
        
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
