import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam, c_l, c_r, self_attn_weights):
    cmu_l = c_l * mu_l
    cmu_r = c_r * mu_r
    def f(_, p, is_left): return (cmu_l if is_left else cmu_r) @ p
    return self.fold_down_tree(f, lam)

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam  = tree_utils.init_tensor(embed_dim),
    c_l = tree_utils.init_tensor(),
    c_r = tree_utils.init_tensor(),
    self_attn_weights = torch.zeros(len(tree_utils.HEAD_IDS[1:]), config['num_enc_heads'], dtype=torch.float, device=ut.get_device()),
  )

def get_enc_mask(toks, # [bsz, src_len]
                 structs, # list of structs
                 num_heads, # integer
                 mu_l, # [embed_dim, embed_dim]
                 mu_r, # [embed_dim, embed_dim]
                 lam, # [embed_dim]
                 c_l, # scalar
                 c_r, # scalar
                 self_attn_weights, # [NUM_HEAD_IDS, num_heads]
                 ):
  bsz, src_len = toks.size()
  masks = tree_utils.get_enc_mask(toks, structs, 1).unsqueeze(1)  # [bsz, 1, src_len, src_len]
  heads = torch.zeros(bsz, num_heads, src_len, src_len, dtype=torch.float, device=toks.device) # [bsz, num_heads, src_len, src_len]
  self_attn = self_attn_weights.view(self_attn_weights.size()[0], 1, -1, 1, 1)
  for i, hid in enumerate(tree_utils.HEAD_IDS[1:]):
    heads += self_attn[i] * masks.bitwise_and(hid).type(torch.bool)
  return heads, masks.type(torch.bool)
