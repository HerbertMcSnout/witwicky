import os
import re
import time
import shutil
from subprocess import Popen, PIPE

import numpy
import torch

import nmt.utils as ut
import nmt.all_constants as ac


class Validator(object):
    def __init__(self, config, model):
        super(Validator, self).__init__()
        self.logger = ut.get_logger(config['log_file'])
        self.logger.info('Initializing validator')

        self.model = model
        self.model_name = config['model_name']
        self.restore_segments = config['restore_segments']
        self.val_by_bleu = config['val_by_bleu']
        self.save_to = config['save_to']
        self.grad_clamp = bool(config['grad_clamp'])

        self.get_cpkt_path = lambda score: os.path.join(self.save_to, f'{self.model_name}-{score}.pth')
        self.n_best = config['n_best']

        scriptdir = os.path.dirname(os.path.abspath(__file__))
        self.bleu_script = f'{scriptdir}/../scripts/multi-bleu.perl'
        if not os.path.exists(self.bleu_script):
            raise FileNotFoundError(self.bleu_script)

        if not os.path.exists(self.save_to):
            os.makedirs(self.save_to)

        self.val_trans_out = os.path.join(self.save_to, 'val_trans.txt')
        self.val_beam_out = os.path.join(self.save_to, 'val_beam_trans.txt')

        self.write_val_trans = config['write_val_trans']

        # I'll leave test alone for now since this version of the code doesn't automatically
        # report BLEU on test anw. The reason is it's up to the dataset to use multi-bleu
        # or NIST bleu. I'll include it in the future
        self.dev_ref = self.model.data_manager.data_files[ac.VALIDATING][self.model.data_manager.trg_lang]
        if self.restore_segments:
            self.dev_ref = self.remove_bpe(self.dev_ref, outfile=os.path.join(self.save_to, f'dev.{self.model.data_manager.trg_lang}.nobpe'))

        self.perp_curve_path = os.path.join(self.save_to, 'dev_perps.npy')
        self.best_perps_path = os.path.join(self.save_to, 'best_perp_scores.npy')
        self.perp_curve = numpy.array([], dtype=numpy.float32)
        self.best_perps = numpy.array([], dtype=numpy.float32)
        if os.path.exists(self.perp_curve_path):
            self.perp_curve = numpy.load(self.perp_curve_path)
        if os.path.exists(self.best_perps_path):
            self.best_perps = numpy.load(self.best_perps_path)

        if self.val_by_bleu:
            self.bleu_curve_path = os.path.join(self.save_to, 'bleu_scores.npy')
            self.best_bleus_path = os.path.join(self.save_to, 'best_bleu_scores.npy')
            self.bleu_curve = numpy.array([], dtype=numpy.float32)
            self.best_bleus = numpy.array([], dtype=numpy.float32)
            if os.path.exists(self.bleu_curve_path):
                self.bleu_curve = numpy.load(self.bleu_curve_path)
            if os.path.exists(self.best_bleus_path):
                self.best_bleus = numpy.load(self.best_bleus_path)

    def evaluate_perp(self):
        self.model.eval()
        start_time = time.time()
        acc_loss = []
        acc_smooth_loss = []
        acc_weight = []

        with torch.no_grad():
            for _, src_toks, src_structs, trg_toks, targets in self.model.data_manager.get_batches(mode=ac.VALIDATING):
                # get loss
                ret = self.model(src_toks, src_structs, trg_toks, targets)
                acc_loss.append(ret['nll_loss'].detach())
                acc_smooth_loss.append(ret['loss'].detach())
                acc_weight.append((targets != ac.PAD_ID).detach().sum())

        #total_loss = sum(x.item() for x in acc_loss)
        #total_smooth_loss = sum(x.item() for x in acc_smooth_loss)
        #total_weight = sum(x.item() for x in acc_weight)
        total_loss = total_smooth_loss = total_weight = finite = infinite = 0
        for nll, smooth, weight in zip(acc_loss, acc_smooth_loss, acc_weight):
            if not self.grad_clamp or (torch.isfinite(smooth) and torch.isfinite(nll)):
                total_loss += nll.cpu().numpy()
                total_smooth_loss += smooth.cpu().numpy()
                total_weight += weight.cpu().numpy()
                finite += 1
            else:
                infinite += 1

        perp = total_loss / total_weight if total_weight else float('nan')
        perp = numpy.exp(perp) if perp < 300 else float('inf')
        perp = round(perp, ndigits=3)

        smooth_perp = total_smooth_loss / total_weight if total_weight else float('nan')
        smooth_perp = numpy.exp(smooth_perp) if smooth_perp < 300 else float('inf')
        smooth_perp = round(smooth_perp, ndigits=3)

        self.perp_curve = numpy.append(self.perp_curve, smooth_perp)
        numpy.save(self.perp_curve_path, self.perp_curve)

        self.model.train()
        self.logger.info(f'smooth, true dev perp: {smooth_perp}, {perp}')
        if self.grad_clamp: self.logger.info(f'{finite} finite, {infinite} infinite perp batches')
        self.logger.info('Calculating dev perp took ' + ut.format_time(time.time() - start_time))

    def evaluate_bleu(self):
        self.model.eval()

        start = time.time()
        src_file = self.model.data_manager.data_files[ac.VALIDATING][self.model.data_manager.src_lang]
        best_out, beam_out = self.translate(src_file, to_ids=True)

        p = Popen(f'perl {self.bleu_script} {self.dev_ref} < {best_out}', shell=True, stdout=PIPE)
        output, _ = p.communicate()
        output = output.decode('utf-8').strip('\n')
        out_parse = re.match(r'BLEU = [-.0-9]+', output)
        self.logger.info(output)

        if out_parse is None:
            self.logger.info('Error extracting BLEU score: out_parse is None')
            bleu = float('-inf')
        else:
            bleu = float(out_parse.group()[6:])

        if self.write_val_trans:
            best_file = f'{best_out}-{bleu:.2f}'
            shutil.copyfile(best_out, best_file)

            beam_file = f'{beam_out}-{bleu:.2f}'
            shutil.copyfile(beam_out, beam_file)

        # add summaries
        self.bleu_curve = numpy.append(self.bleu_curve, bleu)
        numpy.save(self.bleu_curve_path, self.bleu_curve)

    def evaluate(self):
        self.evaluate_perp()
        if self.val_by_bleu:
            self.evaluate_bleu()

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

    def maybe_save(self):
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
            best_scores_str = ', '.join([f'{float(x):.2f}' for x in scores_sorted])
            self.logger.info(f'Current best {metric} scores: {best_scores_str}')
            self.logger.info(f'Delete {float(worst):.2f}, use {float(score):.2f} instead')
            scores = numpy.delete(scores, remove_idx)

            # Delete the worst checkpoint
            cpkt_path = self.get_cpkt_path(worst)
            if os.path.exists(cpkt_path):
                os.remove(cpkt_path)

        if save_please:
            scores = numpy.append(scores, score)
            cpkt_path = self.get_cpkt_path(score)
            self.model.save(fp=cpkt_path)
            best_scores_str = ', '.join([f'{float(x):.2f}' for x in numpy.sort(scores)])
            self.logger.info(f'Best {metric} scores so far: {best_scores_str}')

        numpy.save(path, scores)
        if self.val_by_bleu: self.best_bleus = scores
        else: self.best_perps = scores

    def validate_and_save(self):
        self.logger.info('Start validation')
        self.evaluate()
        self.maybe_save()

    def remove_bpe(self, infile, outfile=None):
        outfile = outfile or infile + '.nobpe'
        Popen(f'sed -r \'s/(@@ )|(@@ ?$)//g\' < {infile} > {outfile}', shell=True, stdout=PIPE).communicate()
        return outfile

    def translate(self, input_file, to_ids=False):
        bpe_suffix = '.bpe' if self.restore_segments else ''

        basename = os.path.basename(input_file)
        basename = basename.rstrip(self.model.data_manager.src_lang) # remove '.src_lang' suffix
        basename += self.model.data_manager.trg_lang # add '.trg_lang' suffix

        base_fp = os.path.join(self.save_to, basename)
        best_fp_base = base_fp + '.best_trans'
        beam_fp_base = base_fp + '.beam_trans'
        best_fp = best_fp_base + bpe_suffix
        beam_fp = beam_fp_base + bpe_suffix

        open(best_fp, 'w').close()
        open(beam_fp, 'w').close()

        with open(best_fp, 'a') as best_stream, open(beam_fp, 'a') as beam_stream:
            self.model.translate(input_file,
                                 best_stream,
                                 beam_stream,
                                 num_preload=ac.DEFAULT_VALIDATION_NUM_PRELOAD,
                                 to_ids=to_ids
            )
        if self.restore_segments:
            self.remove_bpe(best_fp, best_fp_base)
            self.remove_bpe(beam_fp, beam_fp_base)
        return best_fp_base, beam_fp_base
