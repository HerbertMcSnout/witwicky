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
        learn_pos_scale = self.config['learn_pos_scale']

        device = ut.get_device()

        # get trg positonal embedding
        if not learned_pos:
            self.pos_embedding_trg = ut.set_position_encoding(embed_dim, max_len)
        else:
            self.pos_embedding_trg = Parameter(torch.empty(max_len, embed_dim, dtype=torch.float, device=device))
            nn.init.normal_(self.pos_embedding_trg, mean=0, std=embed_dim ** -0.5)

        self.struct = self.config['struct']
        params = [(name, Parameter(x)) for name, x in self.struct.get_params(self.config).items()]
        self.struct_params = [x for _, x in params]
        for name, x in params:
            self.register_parameter(name, x)

        # get word embeddings
        # TODO: src_vocab_mask is assigned but never used (?)
        src_vocab_size, trg_vocab_size = ut.get_vocab_sizes(self.config)
        self.src_vocab_mask, self.trg_vocab_mask = ut.get_vocab_masks(self.config, src_vocab_size, trg_vocab_size)
        if tie_mode == ac.ALL_TIED:
            src_vocab_size = trg_vocab_size = self.trg_vocab_mask.shape[0]

        self.out_bias = Parameter(torch.empty(trg_vocab_size, dtype=torch.float, device=device))
        nn.init.constant_(self.out_bias, 0.)

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.out_embedding = self.trg_embedding.weight
        if self.config['separate_embed_scales']:
            self.src_embed_scale = Parameter(torch.tensor([embed_dim ** 0.5], device=device))
            self.trg_embed_scale = Parameter(torch.tensor([embed_dim ** 0.5], device=device))
        else:
            self.src_embed_scale = self.trg_embed_scale = torch.tensor([embed_dim ** 0.5], device=device)

        self.src_pos_embed_scale = torch.tensor([(embed_dim / 2) ** 0.5], device=device)
        self.trg_pos_embed_scale = torch.tensor([1.], device=device) # trg pos embedding already returns vector of norm sqrt(embed_dim/2)
        if learn_pos_scale:
            self.src_pos_embed_scale = Parameter(self.src_pos_embed_scale)
            self.trg_pos_embed_scale = Parameter(self.trg_pos_embed_scale)

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
            #self.src_embed_scale.data_ptr():{'lr':self.config['embed_scale_lr']},
            #self.trg_embed_scale.data_ptr():{'lr':self.config['embed_scale_lr']}
        }

        # Debugging
        self.debug_stats = {'loss':[], 'reg':[]}

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
        
        self.decoder_mask = None

    def get_decoder_mask(self, size):
        if self.decoder_mask is None or self.decoder_mask.size()[-1] < size:
            self.decoder_mask = torch.triu(torch.ones((1, 1, size, size), dtype=torch.bool, device=ut.get_device()), diagonal=1)
            return self.decoder_mask
        else:
            return self.decoder_mask[:, :, :size, :size]

    def process_pos_embedding(self, max_len, struct):
        pad = 0, 0, 0, max_len - struct.size()
        embed_dim = self.config['embed_dim']
        pe = struct.get_pos_embedding(embed_dim, self.struct_params).flatten()
        if not torch.is_tensor(pe): pe = torch.stack(pe)
        return F.pad(pe, pad)
    
    def get_pos_embedding(self, max_len, structs=None):
        if structs is not None:
            # [bsz, max_len, embed_dim]
            return torch.stack([self.process_pos_embedding(max_len, x) for x in structs])
        else:
            # [1, max_len, embed_dim]
            return self.pos_embedding_trg[:max_len, :].unsqueeze(0)

    def get_input(self, toks, structs=None, calc_reg=False):
        max_len = toks.size()[-1]
        embed_dim = self.config['embed_dim']
        embeds = self.src_embedding if structs is not None else self.trg_embedding
        word_embeds = embeds(toks) # [bsz, max_len, embed_dim]
        embed_scale = self.trg_embed_scale if structs is None else self.src_embed_scale

        if self.config['fix_norm']:
            word_embeds = ut.normalize(word_embeds, scale=False)
        else:
            word_embeds = word_embeds * embed_scale

        pos_embeds = self.get_pos_embedding(max_len, structs)
        pe_scale = self.src_pos_embed_scale if structs is not None else self.trg_pos_embed_scale
        reg_penalty = 0.0
        if calc_reg: # Penalize pos embeddings with (pre-scaled) norms other than 1:
            #norms = pos_embeds.norm(dim=-1) + (toks == ac.PAD_ID) # set all padding values to 1 so they get no penalty
            #reg_penalty = self.struct.get_reg_penalty(norms).sum(dim=[0,1]) * self.config['pos_norm_penalty']
            reg_penalty = self.struct.get_reg_penalty(pos_embeds, toks != ac.PAD_ID) * self.config['pos_norm_penalty']
        return word_embeds + pos_embeds * pe_scale, reg_penalty

    def forward(self, src_toks, src_structs, trg_toks, targets, b=None, e=None):
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        #decoder_mask = torch.triu(torch.ones((trg_toks.size()[-1], trg_toks.size()[-1]), dtype=torch.bool, device=ut.get_device()), diagonal=1)
        #decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)
        decoder_mask = self.get_decoder_mask(trg_toks.size()[-1])

        encoder_inputs, reg_penalty = self.get_input(src_toks, src_structs, calc_reg=hasattr(self.struct, "get_reg_penalty"))
        
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)

        decoder_inputs, _ = self.get_input(trg_toks)
        decoder_outputs = self.decoder(decoder_inputs, decoder_mask, encoder_outputs, encoder_mask)

        logits = self.logit_fn(decoder_outputs)
        neglprobs = F.log_softmax(logits, -1)
        neglprobs = neglprobs * self.trg_vocab_mask.reshape(1, -1)
        targets = targets.reshape(-1, 1)
        non_pad_mask = targets != ac.PAD_ID
        nll_loss = -neglprobs.gather(dim=-1, index=targets)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = -neglprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        label_smoothing = self.config['label_smoothing']

        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / self.trg_vocab_mask.sum()
        else:
            loss = nll_loss
        
        self.debug_stats['loss'].append(loss.detach().item())
        self.debug_stats['reg'].append(float(reg_penalty))
        loss += reg_penalty

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
        encoder_inputs, _ = self.get_input(src_toks, src_structs)
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
            return word_embeds + pos_embeds * self.trg_pos_embed_scale

        def logprob(decoder_output):
            return F.log_softmax(self.logit_fn(decoder_output), dim=-1)

        if self.config['length_model'] == ac.GNMT_LENGTH_MODEL:
            length_model = ut.gnmt_length_model(self.config['length_alpha'])
        elif self.config['length_model'] == ac.LINEAR_LENGTH_MODEL:
            length_model = lambda t, p: p + self.config['length_alpha'] * t
        elif self.config['length_model'] == ac.NO_LENGTH_MODEL:
            length_model = lambda t, p: p
        else:
            raise ValueError("invalid length_model '{}'".format(self.config['length_model']))

        return self.decoder.beam_decode(encoder_outputs, encoder_mask, get_trg_inp, logprob, length_model, ac.BOS_ID, ac.EOS_ID, max_lengths, beam_size=self.config['beam_size'])
