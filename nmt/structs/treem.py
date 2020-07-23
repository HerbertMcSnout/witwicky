import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(x):
  return (x - x.mean()) / x.std()

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    mu_l, mu_r, lam_root, lam_leaf_l, lam_leaf_r = params

    def f_up(_, l, r):
      l2 = (mu_l @ l) if l is not None else lam_leaf_l
      r2 = (mu_r @ r) if r is not None else lam_leaf_r
      return l2 * r2

    def f_down(_, p, is_left):
      mu = mu_l if is_left else mu_r
      return mu @ p

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(x): return x[0] * x[1] * (embed_dim ** -0.5)

    pe_up = self.fold_up_tree(f_up)
    pe_down = self.fold_down_tree(f_down, lam_root)
    pe = pe_up.zip(pe_down).map(f_mult)
    return pe

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  from nmt.utils import get_device
  device = get_device()
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam_root   = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)),
    lam_leaf_l = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)),
    lam_leaf_r = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)),
  )

def get_reg_penalty(x, mask):
  x = x.norm(dim=-1) + ~mask # set all padding values to 1 so they get no penalty
  x = x + 1/x - 2
  x = x.sum()
  return x
