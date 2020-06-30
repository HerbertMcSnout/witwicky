import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, params):
    mu_l, mu_r, lam = params
    def f(_, p, is_left):
      return (mu_l if is_left else mu_r) @ p # regularize(mu_... @ p)
    return self.fold_down_tree(f, lam)

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam  = tree_utils.init_tensor(embed_dim),
  )

def get_reg_penalty(x):
  return torch.max(x, 1/x) - 1
