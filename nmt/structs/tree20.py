import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, params):
    mu_l, mu_r, lam, lam_l, lam_r = params
    step_scale = embed_dim ** 0.5
    def f_down(_, p, is_left):
      return (mu_l if is_left else mu_r) @ p
    def f_up(_, l, r):
      lv = (mu_l @ l) if l is not None else lam_l
      rv = (mu_r @ r) if r is not None else lam_r
      return lv * rv * step_scale
    d = self.fold_down_tree(f_down, lam)
    u = self.fold_up_tree(f_up)
    return d.zip(u).map(lambda x: x[0] * x[1] * step_scale)

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  mu_l = tree_utils.init_tensor(embed_dim, embed_dim)
  mu_r = tree_utils.init_tensor(embed_dim, embed_dim)
  lam = tree_utils.init_tensor(embed_dim)
  lam_l = tree_utils.init_tensor(embed_dim)
  lam_r = tree_utils.init_tensor(embed_dim)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam":lam, "lam_l":lam_l, "lam_r":lam_r}

def get_reg_penalty(x):
  return torch.abs(torch.log(x))
