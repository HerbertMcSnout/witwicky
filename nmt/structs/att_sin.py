import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, self_attn_weights):
    return torch.zeros(self.size(), embed_dim, device=ut.get_device())

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  return dict(
    self_attn_weights = torch.zeros(len(tree_utils.HEAD_IDS[1:]), config['num_enc_heads'], device=ut.get_device()),
  )

def get_enc_mask(toks, structs, num_heads, self_attn_weights):
  masks = tree_utils.get_enc_mask(toks, structs, 1)  # [bsz, 1, src_len, src_len]
  heads = torch.zeros_like(masks, dtype=torch.float) # [bsz, 1, src_len, src_len]
  mask = masks.squeeze(1) # [bsz, src_len, src_len]
  heads = heads.expand(-1, num_heads, -1, -1).clone() # [bsz, num_heads, src_len, src_len]

  for head_num in range(num_heads):
    for i, hid in enumerate(tree_utils.HEAD_IDS[1:]):
      heads[:, head_num] += self_attn_weights[i, head_num] * mask.bitwise_and(hid).type(torch.bool)
  return heads, masks.type(torch.bool)
