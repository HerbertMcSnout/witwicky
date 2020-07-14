import time
import numpy
from os.path import join
from os.path import exists

import torch

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations
from nmt.validator import Validator

from copy import deepcopy

if torch.cuda.is_available():
    torch.cuda.manual_seed(ac.SEED)
else:
    torch.manual_seed(ac.SEED)

numpy.random.seed(ac.SEED)


class Trainer(object):
    """Trainer"""
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.config = configurations.get_config(args.proto, getattr(configurations, args.proto), args.config_overrides)
        self.num_preload = args.num_preload

        self.logger = ut.get_logger(self.config['log_file'])

        self.lr = self.config['lr']

        self.train_smooth_perps = []
        self.train_true_perps = []

        self.data_manager = DataManager(self.config)
        self.validator = Validator(self.config, self.data_manager)

        self.validate_freq = self.config['validate_freq']
        if self.validate_freq == 1:
            self.logger.info('Evaluate every {}'.format('epoch' if self.config['val_per_epoch'] else 'batch'))
        else:
            self.logger.info('Evaluate every {:,} {}'.format(self.validate_freq, 'epochs' if self.config['val_per_epoch'] else 'batches'))

        # For logging
        self.log_freq = 100  # log train stat every this-many batches
        self.log_train_loss = []
        self.log_nll_loss = []
        self.log_train_weights = []
        self.log_grad_norms = []
        self.total_batches = 0 # number of batches done for the whole training
        self.epoch_loss = 0. # total train loss for whole epoch
        self.epoch_nll_loss = 0. # total train loss for whole epoch
        self.epoch_weights = 0. # total train weights (# target words) for whole epoch
        self.epoch_time = 0. # total exec time for whole epoch, sounds like that tabloid

        # Estimated number of batches per epoch
        self.est_batches = self.data_manager.read_tok_count() // self.config['batch_size']
        self.logger.info('Guessing around {:,} batches per epoch'.format(self.est_batches))
        
        # get model
        self.model = Model(self.config).to(ut.get_device())

        param_count = sum([numpy.prod(p.size()) for p in self.model.parameters()])
        self.logger.info('Model has {:,} parameters'.format(param_count))

        # Set up parameter-specific options
        params = []
        for p in self.model.parameters():
            ptr = p.data_ptr()
            d = {'params': [p]}
            if ptr in self.model.parameter_attrs:
                attrs = self.model.parameter_attrs[ptr]
                for k in attrs:
                    d[k] = attrs[k]
            params.append(d)
        
        self.optimizer = torch.optim.Adam(params, lr=self.lr, betas=(self.config['beta1'], self.config['beta2']), eps=self.config['epsilon'])

        # Set up debug_stats
        self.debug_path = join(self.config['save_to'], 'debug_stats.pth')
        torch.save({}, self.debug_path)
        self.initial_debug_stats = deepcopy(self.model.debug_stats)

    def report_epoch(self, epoch, batches):

        self.logger.info('Finished epoch {}'.format(epoch))
        self.logger.info('    Took {}'.format(ut.format_time(self.epoch_time)))
        self.logger.info('    avg words/sec {:.2f}'.format(self.epoch_weights / self.epoch_time))
        self.logger.info('    avg sec/batch {:.2f}'.format(self.epoch_time / batches))
        self.logger.info('    {} batches'.format(batches))

        train_smooth_perp = self.epoch_loss / self.epoch_weights
        train_true_perp = self.epoch_nll_loss / self.epoch_weights

        self.est_batches = batches
        self.epoch_time = 0.
        self.epoch_nll_loss = 0.
        self.epoch_loss = 0.
        self.epoch_weights = 0.
        self.log_train_loss = []
        self.log_nll_loss = []
        self.log_train_weights = []
        self.log_grad_norms = []

        train_smooth_perp = numpy.exp(train_smooth_perp) if train_smooth_perp < 300 else float('inf')
        self.train_smooth_perps.append(train_smooth_perp)
        train_true_perp = numpy.exp(train_true_perp) if train_true_perp < 300 else float('inf')
        self.train_true_perps.append(train_true_perp)

        self.logger.info('    smooth, true perp: {:.2f}, {:.2f}'.format(float(train_smooth_perp), float(train_true_perp)))

        # Save debug_stats
        debug_stats = torch.load(self.debug_path)
        debug_stats[epoch] = self.model.debug_stats
        torch.save(debug_stats, self.debug_path)
        self.model.debug_stats = deepcopy(self.initial_debug_stats)

    def run_log(self, batch, epoch, batch_data):
      #with torch.autograd.detect_anomaly(): # throws exception when any forward computation produces nan
        start = time.time()
        src_toks, src_structs, trg_toks, targets = batch_data

        # zero grad
        self.optimizer.zero_grad()

        # get loss
        ret = self.model(src_toks, src_structs, trg_toks, targets, batch, epoch)
        loss = ret['loss']
        nll_loss = ret['nll_loss']

        if self.config['normalize_loss'] == ac.LOSS_TOK:
            opt_loss = loss / (targets != ac.PAD_ID).sum()
        elif self.config['normalize_loss'] == ac.LOSS_BATCH:
            opt_loss = loss / targets.size()[0]
        else:
            opt_loss = loss

        opt_loss.backward()
        # clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip']).detach()

        # update
        self.adjust_lr()
        self.optimizer.step()

        # update training stats
        num_words = (targets != ac.PAD_ID).detach().sum()

        loss = loss.detach()
        nll_loss = nll_loss.detach()
        self.total_batches += 1
        self.log_train_loss.append(loss)
        self.log_nll_loss.append(nll_loss)
        self.log_train_weights.append(num_words)
        self.log_grad_norms.append(grad_norm)
        self.epoch_time += time.time() - start

        if self.total_batches % self.log_freq == 0:
            log_train_loss = sum(x.item() for x in self.log_train_loss)
            log_nll_loss = sum(x.item() for x in self.log_nll_loss)
            log_train_weights = sum(x.item() for x in self.log_train_weights)

            self.epoch_loss += log_train_loss
            self.epoch_nll_loss += log_nll_loss
            self.epoch_weights += log_train_weights
            
            acc_speed_word = self.epoch_weights / self.epoch_time
            acc_speed_time = self.epoch_time / batch

            avg_smooth_perp = log_train_loss / log_train_weights
            avg_smooth_perp = numpy.exp(avg_smooth_perp) if avg_smooth_perp < 300 else float('inf')
            avg_true_perp = log_nll_loss / log_train_weights
            avg_true_perp = numpy.exp(avg_true_perp) if avg_true_perp < 300 else float('inf')

            avg_grad_norm = sum(self.log_grad_norms) / len(self.log_grad_norms)
            #median_grad_norm = sorted(self.log_grad_norms)[len(self.log_grad_norms)//2]

            est_percent = int(100 * batch / self.est_batches)
            epoch_len = max(5, ut.get_num_digits(self.config['max_epochs']))
            batch_len = max(5, ut.get_num_digits(self.est_batches))
            if batch > self.est_batches: remaining = '?'
            else: remaining = ut.format_time(acc_speed_time * (self.est_batches - batch))

            self.log_train_loss = []
            self.log_nll_loss = []
            self.log_train_weights = []
            self.log_grad_norms = []
            cells = [f'{epoch:{epoch_len}}',
                     f'{batch:{batch_len}}',
                     f'{est_percent:3}%',
                     f'{remaining:>9}',
                     f'{acc_speed_word:#7.4g}',
                     f'{acc_speed_time:#6.4g}s',
                     f'{avg_smooth_perp:#11.4g}',
                     f'{avg_true_perp:#9.4g}',
                     f'{avg_grad_norm:#9.4g}']
            self.logger.info('  '.join(cells))

    def adjust_lr(self):
        if self.config['warmup_style'] == ac.ORG_WARMUP:
            step = self.total_batches + 1.0
            if step < self.config['warmup_steps']:
                lr = self.config['embed_dim'] ** (-0.5) * step * self.config['warmup_steps'] ** (-1.5)
            else:
                lr = max(self.config['embed_dim'] ** (-0.5) * step ** (-0.5), self.config['min_lr'])
            for p in self.optimizer.param_groups:
                p['lr'] = lr
        elif self.config['warmup_style'] == ac.FIXED_WARMUP:
            warmup_steps = self.config['warmup_steps']
            step = self.total_batches + 1.0
            start_lr = self.config['start_lr']
            peak_lr = self.config['lr']
            min_lr = self.config['min_lr']
            if step < warmup_steps:
                lr = start_lr + (peak_lr - start_lr) * step / warmup_steps
            else:
                lr = max(min_lr, peak_lr * warmup_steps ** (0.5) * step ** (-0.5))
            for p in self.optimizer.param_groups:
                p['lr'] = lr
        elif self.config['warmup_style'] == ac.UPFLAT_WARMUP:
            warmup_steps = self.config['warmup_steps']
            step = self.total_batches + 1.0
            start_lr = self.config['start_lr']
            peak_lr = self.config['lr']
            min_lr = self.config['min_lr']
            if step < warmup_steps:
                lr = start_lr + (peak_lr - start_lr) * step / warmup_steps
                for p in self.optimizer.param_groups:
                    p['lr'] = lr
        else:
            pass

    def train(self):
        self.model.train()
        for epoch in range(1, self.config['max_epochs'] + 1):
            batch = 0
            for batch_data in self.data_manager.get_batch(mode=ac.TRAINING, num_preload=self.num_preload):
                if batch == 0:
                    self.logger.info('Begin epoch {}'.format(epoch))
                    epoch_str = ' ' * max(0, ut.get_num_digits(self.config['max_epochs']) - 5) + 'epoch'
                    batch_str = ' ' * max(0, ut.get_num_digits(self.est_batches) - 5) + 'batch'
                    self.logger.info('  '.join([epoch_str, batch_str, 'est%', 'remaining', 'trg w/s', 's/batch', 'smooth perp', 'true perp', 'grad norm']))
                batch += 1
                self.run_log(batch, epoch, batch_data)
                if not self.config['val_per_epoch']:
                    self.maybe_validate()

            self.report_epoch(epoch, batch)
            if self.config['val_per_epoch'] and epoch % self.validate_freq == 0:
                self.maybe_validate(just_validate=True)
            
            if self.is_patience_exhausted(self.config['early_stop_patience'] // self.validate_freq):
                self.logger.info('No improvement for last {} epochs; stopping early!'.format(self.config['early_stop_patience'] * self.validate_freq))
                break

        # validate 1 last time
        if not self.config['val_by_bleu']:
            self.maybe_validate(just_validate=True)

        self.logger.info('Training finished')
        self.logger.info('Train smooth perps:')
        self.logger.info(', '.join(['{:.2f}'.format(x) for x in self.train_smooth_perps]))
        self.logger.info('Train true perps:')
        self.logger.info(', '.join(['{:.2f}'.format(x) for x in self.train_true_perps]))
        numpy.save(join(self.config['save_to'], 'train_smooth_perps.npy'), self.train_smooth_perps)
        numpy.save(join(self.config['save_to'], 'train_true_perps.npy'), self.train_true_perps)

        self.logger.info('Save final checkpoint')
        self.save_checkpoint()

        # Evaluate on test
        test_file = self.data_manager.data_files[ac.TESTING][self.data_manager.src_lang]
        if exists(test_file):
            self.logger.info('Evaluate test')
            self.restart_to_best_checkpoint()
            self.validator.translate(self.model, test_file)
            self.logger.info('Translate dev set')
            self.validator.translate(self.model, self.data_manager.data_files[ac.VALIDATING][self.data_manager.src_lang])

    def save_checkpoint(self):
        cpkt_path = join(self.config['save_to'], '{}.pth'.format(self.config['model_name']))
        torch.save(self.model.state_dict(), cpkt_path)

    def restart_to_best_checkpoint(self):
        if self.config['val_by_bleu']:
            best_bleu = numpy.max(self.validator.best_bleus)
            best_cpkt_path = self.validator.get_cpkt_path(best_bleu)
        else:
            best_perp = numpy.min(self.validator.best_perps)
            best_cpkt_path = self.validator.get_cpkt_path(best_perp)

        self.logger.info('Restore best cpkt from {}'.format(best_cpkt_path))
        self.model.load_state_dict(torch.load(best_cpkt_path))

    def is_patience_exhausted(self, patience):
        val_bleu = self.config['val_by_bleu']
        curve = self.validator.bleu_curve if val_bleu else self.validator.perp_curve
        minmax = min if val_bleu else max
        return patience and len(curve) > patience and (curve[-1] < minmax(curve[-1-patience:-1])) == val_bleu

    def maybe_validate(self, just_validate=False):
        if self.total_batches % self.validate_freq == 0 or just_validate:
            self.save_checkpoint()
            self.validator.validate_and_save(self.model)

            # if doing annealing
            step = self.total_batches + 1.0
            warmup_steps = self.config['warmup_steps']

            if self.config['warmup_style'] == ac.NO_WARMUP \
               or (self.config['warmup_style'] == ac.UPFLAT_WARMUP and step >= warmup_steps) \
               and self.config['lr_decay'] > 0:

                if self.is_patience_exhausted(self.config['lr_decay_patience'] // self.validate_freq):
                    if self.config['val_by_bleu']:
                        metric = 'bleu'
                        scores = self.validator.bleu_curve
                    else:
                        metric = 'perp'
                        scores = self.validator.perp_curve
                    scores = ', '.join([str(x) for x in scores[-1 - self.config['lr_decay_patience']:]])

                    self.logger.info('Past {} scores are {}'.format(metric, scores))
                    # when don't use warmup, decay lr if dev not improve
                    if self.lr * self.config['lr_decay'] >= self.config['min_lr']:
                        self.logger.info('Anneal the learning rate from {} to {}'.format(self.lr, self.lr * self.config['lr_decay']))
                        self.lr = self.lr * self.config['lr_decay']
                        for p in self.optimizer.param_groups:
                            p['lr'] = self.lr
