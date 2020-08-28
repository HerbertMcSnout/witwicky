import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam, mu_c):
    cmu_l = mu_l * mu_c
    cmu_r = mu_r * mu_c
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
    mu_c = tree_utils.init_tensor(),
  )

