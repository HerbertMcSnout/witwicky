import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam, self_attn_weights):
    def f(_, p, is_left): return (mu_l if is_left else mu_r) @ p
    return self.fold_down_tree(f, lam)

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam  = tree_utils.init_tensor(embed_dim),
    self_attn_weights = 0.1 * torch.ones(4, device=ut.get_device()),
  )

def get_enc_mask(toks, structs, num_heads, mu_l, mu_r, lam, self_attn_weights):
  # 17a
  #heads = torch.zeros(num_heads, dtype=torch.uint16)
  #heads[ : num_heads//2]                 = tree_utils.HEAD_PARENT_ID | tree_utils.HEAD_CHILD_ID | tree_utils.HEAD_SELF_ID
  #heads[num_heads//2 : (3*num_heads)//4] = tree_utils.HEAD_PARENT_ID | tree_utils.HEAD_SELF_ID
  #heads[(3*num_heads)//4 : ]             = tree_utils.HEAD_SELF_ID | tree_utils.HEAD_CHILD_ID | tree_utils.HEAD_OTHER_ID | tree_utils.HEAD_PARENT_ID
  
  # 17b
  #heads = torch.zeros(1, dtype=torch.uint16)
  #heads[:] = tree_utils.HEAD_SELF_ID | tree_utils.HEAD_CHILD_ID | tree_utils.HEAD_PARENT_ID
  #return tree_utils.get_enc_mask(toks, structs, heads)
  
  masks = tree_utils.get_enc_mask(toks, structs, 1)
  heads = torch.zeros_like(masks, dtype=torch.float)

  for i, hid in enumerate(tree_utils.HEAD_IDS[1:]):
    heads += self_attn_weights[i] * masks.bitwise_and(hid)
  return heads
