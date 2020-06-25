import os, os.path
import re
import time
import shutil
from subprocess import Popen, PIPE
from os.path import join
from os.path import exists

import numpy
import torch

import nmt.utils as ut
import nmt.all_constants as ac


class Validator(object):
    def __init__(self, config, data_manager):
        super(Validator, self).__init__()
        self.logger = ut.get_logger(config['log_file'])
        self.logger.info('Initializing validator')

        self.data_manager = data_manager
        self.restore_segments = config['restore_segments']
        self.val_by_bleu = config['val_by_bleu']
        self.save_to = config['save_to']

        self.get_cpkt_path = lambda score: join(self.save_to, '{}-{}.pth'.format(config['model_name'], score))
        self.n_best = config['n_best']

        scriptdir = os.path.dirname(os.path.abspath(__file__))
        self.bleu_script = '{}/../scripts/multi-bleu.perl'.format(scriptdir)
        assert exists(self.bleu_script)

        if not exists(self.save_to):
            os.makedirs(self.save_to)

        self.val_trans_out = join(self.save_to, 'val_trans.txt')
        self.val_beam_out = join(self.save_to, 'val_beam_trans.txt')

        # I'll leave test alone for now since this version of the code doesn't automatically
        # report BLEU on test anw. The reason is it's up to the dataset to use multi-bleu
        # or NIST bleu. I'll include it in the future
        self.dev_ref = self.data_manager.data_files[ac.VALIDATING][self.data_manager.trg_lang]
        if self.restore_segments:
            self.dev_ref = self.remove_bpe(self.dev_ref)

        self.perp_curve_path = join(self.save_to, 'dev_perps.npy')
        self.best_perps_path = join(self.save_to, 'best_perp_scores.npy')
        self.perp_curve = numpy.array([], dtype=numpy.float32)
        self.best_perps = numpy.array([], dtype=numpy.float32)
        if exists(self.perp_curve_path):
            self.perp_curve = numpy.load(self.perp_curve_path)
        if exists(self.best_perps_path):
            self.best_perps = numpy.load(self.best_perps_path)

        if self.val_by_bleu:
            self.bleu_curve_path = join(self.save_to, 'bleu_scores.npy')
            self.best_bleus_path = join(self.save_to, 'best_bleu_scores.npy')
            self.bleu_curve = numpy.array([], dtype=numpy.float32)
            self.best_bleus = numpy.array([], dtype=numpy.float32)
            if exists(self.bleu_curve_path):
                self.bleu_curve = numpy.load(self.bleu_curve_path)
            if exists(self.best_bleus_path):
                self.best_bleus = numpy.load(self.best_bleus_path)

    def evaluate_perp(self, model):
        model.eval()
        start_time = time.time()
        total_loss = []
        total_smoothed_loss = []
        total_weight = []

        with torch.no_grad():
            for src_toks, src_structs, trg_toks, targets in self.data_manager.get_batch(mode=ac.VALIDATING):
                # get loss
                ret = model(src_toks, src_structs, trg_toks, targets)
                total_loss.append(ret['nll_loss'].detach())
                total_smoothed_loss.append(ret['loss'].detach())
                total_weight.append((targets != ac.PAD_ID).detach().sum())

        total_loss = sum(x.item() for x in total_loss)
        total_smoothed_loss = sum(x.item() for x in total_smoothed_loss)
        total_weight = sum(x.item() for x in total_weight())

        perp = total_loss / total_weight
        perp = numpy.exp(perp) if perp < 300 else float('inf')
        perp = round(perp, ndigits=3)

        smoothed_perp = total_smoothed_loss / total_weight
        smoothed_perp = numpy.exp(smoothed_perp) if smoothed_perp < 300 else float('inf')
        smoothed_perp = round(smoothed_perp, ndigits=3)

        self.perp_curve = numpy.append(self.perp_curve, smoothed_perp)
        numpy.save(self.perp_curve_path, self.perp_curve)

        model.train()
        self.logger.info('dev perp: {}'.format(perp))
        self.logger.info('smoothed dev perp: {}'.format(smoothed_perp))
        self.logger.info('Calculating dev perp took: {:.2f} minutes'.format(float(time.time() - start_time) / 60.0))

    def evaluate_bleu(self, model):
        model.eval()

        val_trans_out = self.val_trans_out
        val_beam_out = self.val_beam_out
        ref_file = self.dev_ref

        if self.restore_segments: # using bpe
            val_trans_out = val_trans_out + '.bpe'
            val_beam_out = val_beam_out + '.bpe'

        start = time.time()
        src_file = self.data_manager.data_files[ac.VALIDATING][self.data_manager.src_lang]
        self.data_manager.translate(model, src_file, (val_trans_out, val_beam_out), self.logger)

        # Remove BPE
        if self.restore_segments:
            val_trans_out = self.remove_bpe(val_trans_out, self.val_trans_out)
            val_beam_out = self.remove_bpe(val_beam_out, self.val_beam_out)

        multibleu_cmd = ['perl', self.bleu_script, ref_file, '<', val_trans_out]
        p = Popen(' '.join(multibleu_cmd), shell=True, stdout=PIPE)
        output, _ = p.communicate()
        output = output.decode('utf-8').strip('\n')
        out_parse = re.match(r'BLEU = [-.0-9]+', output)
        self.logger.info(output)
        self.logger.info('Validation took: {:.2f} minutes'.format(float(time.time() - start) / 60.0))

        bleu = float('-inf')
        if out_parse is None:
            msg = '\n    Error extracting BLEU score, out_parse is None'
            msg += '\n    It is highly likely that your model just produces garbage.'
            msg += '\n    Be patient yo, it will get better.'
            self.logger.info(msg)
        else:
            bleu = float(out_parse.group()[6:])

        validation_file = "{}-{}".format(val_trans_out, bleu)
        shutil.copyfile(val_trans_out, validation_file)

        beam_file = "{}-{}".format(val_beam_out, bleu)
        shutil.copyfile(val_beam_out, beam_file)

        # add summaries
        self.bleu_curve = numpy.append(self.bleu_curve, bleu)
        numpy.save(self.bleu_curve_path, self.bleu_curve)

    def evaluate(self, model):
        self.evaluate_perp(model)
        if self.val_by_bleu:
            self.evaluate_bleu(model)

    def _is_valid_to_save(self):
        best_scores = self.best_bleus if self.val_by_bleu else self.best_perps
        curve = self.bleu_curve if self.val_by_bleu else self.perp_curve
        if len(best_scores) < self.n_best:
            return None, True
        else:
            m_idx = (numpy.argmin if self.val_by_bleu else numpy.argmax)(best_scores)
            m_score = best_scores[m_idx]
            if (m_score > curve[-1]) == self.val_by_bleu:
                return None, False
            else:
                return m_idx, True

    def maybe_save(self, model):
        remove_idx, save_please = self._is_valid_to_save()

        if self.val_by_bleu:
            metric = 'bleu'
            path = self.best_bleus_path
            score = self.bleu_curve[-1]
            scores = self.best_bleus
            asc = False # descending
        else:
            metric = 'perp'
            path = self.best_perps_path
            score = self.perp_curve[-1]
            scores = self.best_perps
            asc = True # ascending

        if remove_idx is not None:
            worst = scores[remove_idx]
            scores_sorted = numpy.sort(scores)
            if not asc: scores_sorted = scores_sorted[::-1]
            self.logger.info('Current best {} scores: {}'.format(metric, ', '.join(["{:.2f}".format(float(x)) for x in scores_sorted])))
            self.logger.info('Delete {:.2f}, use {:.2f} instead'.format(float(worst), float(score)))
            scores = numpy.delete(scores, remove_idx)

            # Delete the right checkpoint
            cpkt_path = self.get_cpkt_path(worst)

            if exists(cpkt_path):
                os.remove(cpkt_path)

        if save_please:
            scores = numpy.append(scores, score)
            cpkt_path = self.get_cpkt_path(score)
            torch.save(model.state_dict(), cpkt_path)
            self.logger.info('Best {} scores so far: {}'.format(metric, ', '.join(["{:.2f}".format(float(x)) for x in numpy.sort(scores)])))

        numpy.save(path, scores)
        if self.val_by_bleu: self.best_bleus = scores
        else: self.best_perps = scores

    def validate_and_save(self, model):
        self.logger.info('Start validation')
        self.evaluate(model)
        self.maybe_save(model)

    def remove_bpe(self, infile, outfile=None):
        if not outfile:
            outfile = infile + '.nobpe'

        open(outfile, 'w').close()
        Popen("sed -r 's/(@@ )|(@@ ?$)//g' < {} > {}".format(infile, outfile), shell=True, stdout=PIPE).communicate()
        return outfile

    def translate(self, model, input_file):
        self.data_manager.translate(model, input_file, self.save_to, self.logger)



# from evaluate_bleu(), replaced by self.data_manager.translate():
#
#        get_trans_input = self.data_manager.get_trans_input
#        best_out = val_trans_out
#        beam_out = val_beam_out
#        src_file = src_file
#        get_trans = self.get_trans
#        logger = self.logger
#
#        num_sents = 0
#        with open(src_file, 'r') as f:
#            for line in f:
#                if line.strip():
#                    num_sents += 1
#        all_best_trans = [''] * num_sents
#        all_beam_trans = [''] * num_sents
#
#        with torch.no_grad():
#            logger.info('Start translating {}'.format(src_file))
#            start = time.time()
#            count = 0
#            for (src_toks, original_idxs, src_structs) in get_trans_input(src_file):
#                src_toks_cuda = src_toks.to(device)
#                rets = model.beam_decode(src_toks_cuda, src_structs)
#
#                for i, ret in enumerate(rets):
#                    probs = ret['probs'].cpu().detach().numpy().reshape([-1])
#                    scores = ret['scores'].cpu().detach().numpy().reshape([-1])
#                    symbols = ret['symbols'].cpu().detach().numpy()
#
#                    best_trans, beam_trans = get_trans(probs, scores, symbols)
#                    all_best_trans[original_idxs[i]] = best_trans
#                    all_beam_trans[original_idxs[i]] = beam_trans
#
#                    count += 1
#                    if count % 1000 == 0:
#                        logger.info('  Line {}, avg {:.4f} sec/line'.format(count, (time.time() - start) / count))
#
#        model.train()
#
#        with open(best_out, 'w') as ftrans, open(beam_out, 'w') as btrans:
#            ftrans.write('\n'.join(all_best_trans))
#            btrans.write('\n\n'.join(all_beam_trans))
#
#        logger.info('Finished translating {}, took {} minutes'.format(src_file, float(time.time() - start) / 60.0))
