import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, params):
    mu_l_up, mu_r_up, mu_l_dn, mu_r_dn, lam, lam_l, lam_r = params
    def f_down(_, p, is_left):
      return (mu_l_dn if is_left else mu_r_dn) @ p
    def f_up(_, l, r):
      lv = (mu_l_up @ l) if l is not None else lam_l
      rv = (mu_r_up @ r) if r is not None else lam_r
      return lv * rv * (embed_dim ** 0.5)
    d = self.fold_down_tree(f_down, lam)
    u = self.fold_up_tree(f_up)
    return d.zip(u).map(lambda x: sum(x) * (2**-0.5))

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l_up = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r_up = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_l_dn = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r_dn = tree_utils.init_tensor(embed_dim, embed_dim),
    lam = tree_utils.init_tensor(embed_dim),
    lam_l = tree_utils.init_tensor(embed_dim),
    lam_r = tree_utils.init_tensor(embed_dim),
  )

def get_reg_penalty(x, mask):
  norms = x.norm(dim=-1) + ~mask # set all padding values to 1 so they get no penalty
  return (torch.max(norms, 1/norms) - 1).sum()
