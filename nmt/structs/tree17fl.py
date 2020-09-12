import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def add_omega(self, omega):
    v, l, r = self.v, self.l, self.r
    if l: l = l.add_omega(omega)
    else: v = self.v + omega # right-child, left-sibling
    if r: r = r.add_omega(omega)
    return self.new(v, l, r)

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, c_l, c_r, lam, omega):
    cmu_l = mu_l * c_l
    cmu_r = mu_r * c_r
    def f(_, p, is_left): return (cmu_l if is_left else cmu_r) @ p
    t = self.fold_down_tree(f, lam)
    t.add_omega(omega)
    return t

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l  = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r  = tree_utils.init_tensor(embed_dim, embed_dim),
    c_l   = tree_utils.init_tensor(),
    c_r   = tree_utils.init_tensor(),
    lam   = tree_utils.init_tensor(embed_dim),
    omega = tree_utils.init_tensor(embed_dim),
  )

