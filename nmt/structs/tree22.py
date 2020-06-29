import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, params):
    mu_l_up, mu_r_up, mu_l_dn, mu_r_dn, lam, lam_l, lam_r = params
    add_scale = 2.0 ** -0.5
    def f_down(_, p, is_left):
      return (mu_l_dn if is_left else mu_r_dn) @ p
    def f_up(_, l, r):
      lv = (mu_l_up @ l) if l is not None else lam_l
      rv = (mu_r_up @ r) if r is not None else lam_r
      return (lv + rv) * add_scale
    d = self.fold_down_tree(f_down, lam)
    u = self.fold_up_tree(f_up)
    return d.zip(u).map(lambda x: sum(x) * add_scale)

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  mu_l_up = tree_utils.init_tensor(embed_dim, embed_dim)
  mu_r_up = tree_utils.init_tensor(embed_dim, embed_dim)
  mu_l_dn = tree_utils.init_tensor(embed_dim, embed_dim)
  mu_r_dn = tree_utils.init_tensor(embed_dim, embed_dim)
  lam = tree_utils.init_tensor(embed_dim)
  lam_l = tree_utils.init_tensor(embed_dim)
  lam_r = tree_utils.init_tensor(embed_dim)
  return {"mu_l_up":mu_l_up, "mu_r_up":mu_r_up, "mu_l_dn":mu_l_dn, "mu_r_dn":mu_r_dn, "lam":lam, "lam_l":lam_l, "lam_r":lam_r}

def get_reg_penalty(x):
  return torch.max(x, 1/x) - 1
