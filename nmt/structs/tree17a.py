import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam):
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
  )

def get_enc_mask(toks, structs, num_heads, mu_l, mu_r, lam):
  heads = torch.zeros(num_heads, dtype=torch.int, device=ut.get_device()) # [num_heads]
  heads[:num_heads//4] =               tree_utils.HEAD_BASE_IDS | tree_utils.HEAD_CHILD_ID
  heads[num_heads//4:num_heads//2] =   tree_utils.HEAD_BASE_IDS | tree_utils.HEAD_PARENT_ID
  heads[num_heads//2:3*num_heads//4] = tree_utils.HEAD_BASE_IDS | tree_utils.HEAD_SIB_ID
  heads[3*num_heads//4:] =             tree_utils.HEAD_BASE_IDS | tree_utils.HEAD_DESC_ID
  masks = tree_utils.get_enc_mask(toks, structs, num_heads) # [bsz, num_heads, src_len, src_len]
  masks.bitwise_and_(heads.unsqueeze(0).unsqueeze(2).unsqueeze(3))
  return torch.logical_not(masks)#.transpose(2, 3)
