import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, self_attn_weights):
    return torch.zeros(self.size(), embed_dim, device=self_attn_weights.device)

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  return dict(
    self_attn_weights = torch.zeros(len(tree_utils.HEAD_IDS[1:]), config['num_enc_heads'], device=ut.get_device()),
  )

def get_enc_mask(toks, # [bsz, src_len]
                 structs, # list of structs
                 num_heads, # integer
                 self_attn_weights, # [NUM_HEAD_IDS, num_heads]
                 ):
  bsz, src_len = toks.size()
  masks = tree_utils.get_enc_mask(toks, structs, 1).unsqueeze(1)  # [bsz, 1, src_len, src_len]
  heads = torch.zeros(bsz, num_heads, src_len, src_len, dtype=torch.float, device=toks.device) # [bsz, num_heads, src_len, src_len]
  self_attn = self_attn_weights.view(self_attn_weights.size()[0], 1, -1, 1, 1)
  for i, hid in enumerate(tree_utils.HEAD_IDS[1:]):
    heads += self_attn[i] * masks.bitwise_and(hid).type(torch.bool)
  return heads, masks.type(torch.bool)

