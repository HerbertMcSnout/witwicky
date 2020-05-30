import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import Encoder, Decoder
import nmt.all_constants as ac
import nmt.utils as ut
import nmt.tree as tr

#def check_tensor_h(x, name, f):
#    with open(f, "a") as fh:
#        if not torch.isfinite(x).all():
#            fh.write("{} are nan/inf! size: {}; sum: {}; norm: {}; min: {}; max: {}; infs at: {}; nans at: {}\n" \
#                     .format(name, x.size(), x.sum(), x.norm(), x.min(), x.max(), torch.isinf(x).nonzero(), torch.isnan(x).nonzero()))
#        elif x.dtype in [torch.double, torch.float, torch.half]:
#            fh.write("{} okay; size: {}; sum: {}; norm: {}; min: {}; max: {};\n".format(name, x.size(), x.sum(), x.norm(), x.min(), x.max()))
#        else:
#            fh.write("{} okay; size: {}; sum: {}; min: {}; max: {};\n".format(name, x.size(), x.sum(), x.min(), x.max()))
#
#def check_tensor(x, name, f, check_grad=False):
#    if f is not None:
#        check_tensor_h(x, name, f)
#        if x.requires_grad and check_grad:
#            x.register_hook(lambda x: check_tensor_h(x, name + " backward", f))


class Model(nn.Module):
    """Model"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.init_embeddings()
        self.init_model()

    def init_embeddings(self):
        embed_dim = self.config['embed_dim']
        tie_mode = self.config['tie_mode']
        fix_norm = self.config['fix_norm']
        max_pos_length = self.config['max_pos_length']
        learned_pos = self.config['learned_pos']
        
        self.pos_embedding_mu_l = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.pos_embedding_mu_r = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.pos_embedding_lambda = Parameter(torch.Tensor(embed_dim))
        nn.init.orthogonal_(self.pos_embedding_mu_l)
        nn.init.orthogonal_(self.pos_embedding_mu_r)
        nn.init.normal_(self.pos_embedding_lambda, mean=0, std=embed_dim ** -0.5)

        # get positonal embedding
        if not learned_pos:
            self.pos_embedding_linear = ut.get_positional_encoding(embed_dim, max_pos_length)
        else:
            self.pos_embedding_linear = Parameter(torch.Tensor(max_pos_length, embed_dim))
            nn.init.normal_(self.pos_embedding_linear, mean=0, std=embed_dim ** -0.5)


        # get word embeddings
        # TODO: src_vocab_mask is assigned but never used
        src_vocab_size, trg_vocab_size = ut.get_vocab_sizes(self.config)
        ut.get_logger().info("src_vocab_size: {}, trg_vocab_size: {}".format(src_vocab_size, trg_vocab_size))
        self.src_vocab_mask, self.trg_vocab_mask = ut.get_vocab_masks(self.config, src_vocab_size, trg_vocab_size)
        if tie_mode == ac.ALL_TIED:
            src_vocab_size = trg_vocab_size = self.trg_vocab_mask.shape[0]

        self.out_bias = Parameter(torch.Tensor(trg_vocab_size))
        nn.init.constant_(self.out_bias, 0.)

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.out_embedding = self.trg_embedding.weight
        self.embed_scale = Parameter(torch.tensor([embed_dim ** 0.25])) # embed_dim ** 0.5

        if tie_mode == ac.ALL_TIED:
            self.src_embedding.weight = self.trg_embedding.weight

        if not fix_norm:
            nn.init.normal_(self.src_embedding.weight, mean=0, std=embed_dim ** -0.5)
            nn.init.normal_(self.trg_embedding.weight, mean=0, std=embed_dim ** -0.5)
        else:
            d = 0.01 # pure magic
            nn.init.uniform_(self.src_embedding.weight, a=-d, b=d)
            nn.init.uniform_(self.trg_embedding.weight, a=-d, b=d)

        
        # dict where keys are data_ptrs to dicts of parameter options
        # see https://pytorch.org/docs/stable/optim.html#per-parameter-options
        self.parameter_attrs = {self.embed_scale.data_ptr():{'lr':self.config['embed_scale_lr']}}

        # Debugging
        self.debug_stats = {'sats':[], 'embed_scales':[], 'word_embeds':[], 'pos_embeds':[]}

    def init_model(self):
        num_enc_layers = self.config['num_enc_layers']
        num_enc_heads = self.config['num_enc_heads']
        num_dec_layers = self.config['num_dec_layers']
        num_dec_heads = self.config['num_dec_heads']

        embed_dim = self.config['embed_dim']
        ff_dim = self.config['ff_dim']
        dropout = self.config['dropout']
        norm_in = self.config['norm_in']

        # get encoder, decoder
        self.encoder = Encoder(num_enc_layers, num_enc_heads, embed_dim, ff_dim, dropout=dropout, norm_in=norm_in)
        self.decoder = Decoder(num_dec_layers, num_dec_heads, embed_dim, ff_dim, dropout=dropout, norm_in=norm_in)

        # leave layer norm alone
        init_func = nn.init.xavier_normal_ if self.config['weight_init_type'] == ac.XAVIER_NORMAL else nn.init.xavier_uniform_
        for m in [self.encoder.self_atts, self.encoder.pos_ffs, self.decoder.self_atts, self.decoder.pos_ffs, self.decoder.enc_dec_atts]:
            for p in m.parameters():
                if p.dim() > 1:
                    init_func(p)
                else:
                    nn.init.constant_(p, 0.)

    def get_input(self, toks, trees=None, training=False):
        # max_len = toks.size()[-1]
        embeds = self.src_embedding if trees is not None else self.trg_embedding
        word_embeds = embeds(toks) # [bsz, max_len, embed_dim]

        if self.config['fix_norm']:
            word_embeds = ut.normalize(word_embeds, scale=False)
        else:
            word_embeds = word_embeds * self.embed_scale

        if toks.size()[-1] > self.config['max_pos_length']:
            ut.get_logger().error("Sentence length ({}) is longer than max_pos_length ({}); please increase max_pos_length".format(toks.size()[-1], self.config['max_pos_length']))

        if trees is not None:
            pos_embeds = torch.stack([tree.get_pos_embedding2(self.pos_embedding_mu_l, self.pos_embedding_mu_r, self.pos_embedding_lambda, toks.size()[-1]) for tree in trees]) # [bsz, max_len, embed_dim]
        else:
            pos_embeds = self.pos_embedding_linear[:toks.size()[-1], :].unsqueeze(0) # [1, max_len, embed_dim]
        #check_tensor(word_embeds, "word_embeds", f)
        with torch.no_grad():
            pos_sat = (pos_embeds ==  tr.get_clamp_bound(pos_embeds)).sum()
            neg_sat = (pos_embeds == -tr.get_clamp_bound(pos_embeds)).sum()
            avg_sat = (pos_sat + neg_sat) / float(pos_embeds.size()[0] * pos_embeds.size()[1] * pos_embeds.size()[2])
            #check_tensor(pos_embeds, "pos_embeds (avg sat: {:.2f}%)".format(avg_sat*100), f)
            if trees is not None and training:
                self.debug_stats['sats'].append(avg_sat.item())
                self.debug_stats['word_embeds'].append((word_embeds.norm(dim=2).sum() / float(self.embed_scale.item() * pos_embeds.size()[0] * pos_embeds.size()[1])).item())
                self.debug_stats['pos_embeds'].append((pos_embeds.norm(dim=2).sum() / float(pos_embeds.size()[0] * pos_embeds.size()[1])).item())
        return word_embeds + pos_embeds

    def forward(self, src_toks, src_trees, trg_toks, targets, b=None, e=None):

        #f = "batch-logs/epoch-{}-batch-{}.log".format(e, b) if e is not None and b is not None else None
        #check_tensor(self.embed_scale, "embed_scale", f)
        #check_tensor(self.pos_embedding_linear, "pos_embedding_linear", f)
        #check_tensor(self.pos_embedding_lambda, "pos_embedding_lambda", f)
        #check_tensor(self.pos_embedding_mu_l, "pos_embedding_mu_l", f)
        #check_tensor(self.pos_embedding_mu_r, "pos_embedding_mu_r", f)
        self.debug_stats['embed_scales'].append(self.embed_scale.item())
        
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        decoder_mask = torch.triu(torch.ones((trg_toks.size()[-1], trg_toks.size()[-1])), diagonal=1).type(trg_toks.type()) == 1
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)

        encoder_inputs = self.get_input(src_toks, src_trees, training=True)
        
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)

        decoder_inputs = self.get_input(trg_toks, training=True)
        decoder_outputs = self.decoder(decoder_inputs, decoder_mask, encoder_outputs, encoder_mask)

        logits = self.logit_fn(decoder_outputs)
        neglprobs = F.log_softmax(logits, -1)
        neglprobs = neglprobs * self.trg_vocab_mask.type(neglprobs.type()).reshape(1, -1)
        targets = targets.reshape(-1, 1)
        non_pad_mask = targets != ac.PAD_ID
        nll_loss = -neglprobs.gather(dim=-1, index=targets)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = -neglprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        label_smoothing = self.config['label_smoothing']

        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / self.trg_vocab_mask.type(smooth_loss.type()).sum()
        else:
            loss = nll_loss

        return {
            'loss': loss,
            'nll_loss': nll_loss
        }

    def logit_fn(self, decoder_output):
        softmax_weight = self.out_embedding if not self.config['fix_norm'] else ut.normalize(self.out_embedding, scale=True)
        logits = F.linear(decoder_output, softmax_weight, bias=self.out_bias)
        logits = logits.reshape(-1, logits.size()[-1])
        logits[:, ~self.trg_vocab_mask] = -1e9
        return logits

    def beam_decode(self, src_toks, src_trees):
        """Translate a minibatch of sentences. 

        Arguments: src_toks[i,j] is the jth word of sentence i.

        Return: See encoders.Decoder.beam_decode
        """
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        encoder_inputs = self.get_input(src_toks, src_trees)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        #assert False, "src_toks type: {}, dtype: {}".format(src_toks.type(), src_toks.dtype)
        max_lengths = torch.sum(src_toks != ac.PAD_ID, dim=-1).type(src_toks.type()) + 50

        def get_trg_inp(ids, time_step):
            ids = ids.type(src_toks.type())
            word_embeds = self.trg_embedding(ids)
            if self.config['fix_norm']:
                word_embeds = ut.normalize(word_embeds, scale=False)
            else:
                word_embeds = word_embeds * self.embed_scale

            pos_embeds = self.pos_embedding_linear[time_step, :].reshape(1, 1, -1)
            return word_embeds + pos_embeds

        def logprob(decoder_output):
            return F.log_softmax(self.logit_fn(decoder_output), dim=-1)

        if self.config['length_model'] == 'gnmt':
            length_model = ut.gnmt_length_model(self.config['length_alpha'])
        elif self.config['length_model'] == 'linear':
            length_model = lambda t, p: p + self.config['length_alpha'] * t
        elif self.config['length_model'] == 'none':
            length_model = lambda t, p: p
        else:
            raise ValueError("invalid length_model '{}'".format(self.config['length_model']))

        return self.decoder.beam_decode(encoder_outputs, encoder_mask, get_trg_inp, logprob, length_model, ac.BOS_ID, ac.EOS_ID, max_lengths, beam_size=self.config['beam_size'])
