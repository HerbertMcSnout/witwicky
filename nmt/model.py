import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import Encoder, Decoder
import nmt.all_constants as ac
import nmt.utils as ut
#import nmt.tree as tr


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
        max_len = self.config['max_train_length']
        learned_pos = self.config['learned_pos']
        
        self.struct = self.config['struct']
        params = [(name, Parameter(x)) for name, x in self.struct.get_params(self.config).items()]
        self.struct_params = [x for _, x in params]
        for name, x in params:
            self.register_parameter(name, x)
        #self.pos_embedding_mu_l = Parameter(torch.Tensor(embed_dim, embed_dim))
        #self.pos_embedding_mu_r = Parameter(torch.Tensor(embed_dim, embed_dim))
        #self.pos_embedding_lambda = Parameter(torch.Tensor(embed_dim))
        #nn.init.orthogonal_(self.pos_embedding_mu_l)
        #nn.init.orthogonal_(self.pos_embedding_mu_r)
        #nn.init.normal_(self.pos_embedding_lambda, mean=0, std=embed_dim ** -0.5)

        # get positonal embedding
        if not learned_pos:
            self.pos_embedding_trg = ut.get_positional_encoding(embed_dim, max_len)
        else:
            self.pos_embedding_trg = Parameter(torch.Tensor(max_len, embed_dim))
            nn.init.normal_(self.pos_embedding_trg, mean=0, std=embed_dim ** -0.5)


        # get word embeddings
        # TODO: src_vocab_mask is assigned but never used
        src_vocab_size, trg_vocab_size = ut.get_vocab_sizes(self.config)
        self.src_vocab_mask, self.trg_vocab_mask = ut.get_vocab_masks(self.config, src_vocab_size, trg_vocab_size)
        if tie_mode == ac.ALL_TIED:
            src_vocab_size = trg_vocab_size = self.trg_vocab_mask.shape[0]

        self.out_bias = Parameter(torch.Tensor(trg_vocab_size))
        nn.init.constant_(self.out_bias, 0.)

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.out_embedding = self.trg_embedding.weight
        self.src_embed_scale = Parameter(torch.tensor([embed_dim ** 0.5]))
        self.trg_embed_scale = Parameter(torch.tensor([embed_dim ** 0.5]))

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
        self.parameter_attrs = {
            self.src_embed_scale.data_ptr():{'lr':self.config['embed_scale_lr']},
            self.trg_embed_scale.data_ptr():{'lr':self.config['embed_scale_lr']}
        }

        # Debugging
        self.debug_stats = {'src_embed_scales':[], 'trg_embed_scales':[], 'word_embeds':[], 'pos_embeds':[]}

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

    def get_input(self, toks, structs=None, training=False):
        max_len = toks.size()[-1]
        embed_dim = self.config['embed_dim']
        embeds = self.src_embedding if structs is not None else self.trg_embedding
        word_embeds = embeds(toks) # [bsz, max_len, embed_dim]
        dtype = word_embeds.type()
        embed_scale = self.trg_embed_scale if structs is None else self.src_embed_scale

        if self.config['fix_norm']:
            word_embeds = ut.normalize(word_embeds, scale=False)
        else:
            word_embeds = word_embeds * embed_scale

        if structs is not None:
            pos_embeds = [x.get_pos_embedding(embed_dim, self.struct_params).flatten() for x in structs]
            pos_embeds = [[x.type(dtype) for x in xs] for xs in pos_embeds]
            pos_embeds = [x + [torch.zeros(embed_dim).type(dtype)] * (max_len - len(x)) for x in pos_embeds]
            pos_embeds = [torch.stack(x) for x in pos_embeds]
            pos_embeds = torch.stack(pos_embeds) # [bsz, max_len, embed_dim]
#            pos_embeds = torch.stack([struct.get_pos_embedding(self.pos_embedding_mu_l, self.pos_embedding_mu_r, self.pos_embedding_lambda, toks.size()[-1]) for struct in structs]) # [bsz, max_len, embed_dim]
        else:
            pos_embeds = self.pos_embedding_trg[:toks.size()[-1], :].unsqueeze(0) # [1, max_len, embed_dim]
        with torch.no_grad():
            if structs is not None and training:
                self.debug_stats['word_embeds'].append((word_embeds.norm(dim=2).sum() / float(self.src_embed_scale.item() * pos_embeds.size()[0] * pos_embeds.size()[1])).item())
                self.debug_stats['pos_embeds'].append((pos_embeds.norm(dim=2).sum() / float(pos_embeds.size()[0] * pos_embeds.size()[1])).item())
        if structs is not None and training:
            return word_embeds, pos_embeds.type(word_embeds.type())
        return word_embeds + pos_embeds

    def forward(self, src_toks, src_structs, trg_toks, targets, b=None, e=None):
        self.debug_stats['src_embed_scales'].append(self.src_embed_scale.item())
        self.debug_stats['trg_embed_scales'].append(self.trg_embed_scale.item())
        
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        decoder_mask = torch.triu(torch.ones((trg_toks.size()[-1], trg_toks.size()[-1])), diagonal=1).type(trg_toks.type()) == 1
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)

        word_embeds, pos_embeds = self.get_input(src_toks, src_structs, training=True)
        encoder_inputs = word_embeds + pos_embeds * self.config['pos_norm_scale'](self.config)
        
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

        pos_embeds = pos_embeds.type(loss.type())
        # Penalize position embeddings that have (pre-scaled) norms greater than 1
        pe_norms = pos_embeds.norm(dim=2)
        pe_errs = torch.clamp(pe_norms - 1, min=0)
        loss += pe_errs.sum(dim=[0,1]) * self.config['pos_norm_penalty'] #.type(loss.type())

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

    def beam_decode(self, src_toks, src_structs):
        """Translate a minibatch of sentences. 

        Arguments: src_toks[i,j] is the jth word of sentence i.

        Return: See encoders.Decoder.beam_decode
        """
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        encoder_inputs = self.get_input(src_toks, src_structs)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        max_lengths = torch.sum(src_toks != ac.PAD_ID, dim=-1).type(src_toks.type()) + 50

        def get_trg_inp(ids, time_step):
            ids = ids.type(src_toks.type())
            word_embeds = self.trg_embedding(ids)
            if self.config['fix_norm']:
                word_embeds = ut.normalize(word_embeds, scale=False)
            else:
                word_embeds = word_embeds * self.trg_embed_scale

            pos_embeds = self.pos_embedding_trg[time_step, :].reshape(1, 1, -1)
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
