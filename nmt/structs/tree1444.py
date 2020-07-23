import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(x):
  return (x - x.mean()) / x.std()

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    mu_l, mu_r, lam_leaf, lam_root, lam_leaf_l, lam_leaf_r = params

    def f_in(_, l, r):
      l2 = (mu_l @ l) if l is not None else lam_leaf_l
      r2 = (mu_r @ r) if r is not None else lam_leaf_r
      return l2 * r2

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      in_p, out_p = p
      if is_left:
        r = (mu_r @ in_r) if in_r is not None else lam_leaf_r
        return in_l, torch.einsum("i,ij,i->j", out_p, mu_l, r)
        #return in_l, torch.einsum("i,ij,ik,k->j", out_p, mu_l, mu_r, in_r)
      else:
        l = (mu_l @ in_l) if in_l is not None else lam_leaf_l
        return in_r, torch.einsum("i,i,ij->j", out_p, l, mu_r)
        #return in_r, torch.einsum("i,ij,ik,j->k", out_p, mu_l, mu_r, in_l)

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(io): return io[0] * io[1] * (embed_dim ** -0.5)

    pe = self
    pe = pe.fold_up_tree(f_in)
    #pe = pe.fold_up_tree(f_in_aux, (lam_leaf_l, lam_leaf_r))
    pe = pe.fold_up_tree(f_in_aux, (None, None))
    pe = pe.fold_down_tree(f_out, (pe.v[0], lam_root))
    pe = pe.map(f_mult)
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
    lam_leaf   = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)), # inside
    lam_root   = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)), # outside
    lam_leaf_l = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)), # outside
    lam_leaf_r = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)), # outside
  )

def get_reg_penalty(x, mask):
  x = x.norm(dim=-1) + ~mask # set all padding values to 1 so they get no penalty
  x = x + 1/x - 2
  x = x.sum()
  return x
