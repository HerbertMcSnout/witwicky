import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(t, embed_dim):
  norm = 1 if len(t.size()) == 1 else embed_dim ** 0.5
  return t * norm / t.norm()

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    params = [normalize(x, embed_dim) for x in params[:-2]] + [x for x in params[-2:]]
    mu_l, mu_r, lam_leaf, lam_root, lam_leaf_l, lam_leaf_r, mu_l_scale, mu_r_scale = params
    mu_l *= mu_l_scale
    mu_r *= mu_r_scale
    step_scale = embed_dim ** 0.5

    def f_in(_, l, r): return normalize((mu_l @ l) * (mu_r @ r) * step_scale, embed_dim)

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      in_p, out_p = p
      if is_left:
        in_r = in_r if in_r is not None else lam_leaf_r
        return in_l, normalize(torch.einsum("i,ij,i->j", out_p, mu_l, mu_r @ in_r) * step_scale, embed_dim)
      else:
        in_l = in_l if in_l is not None else lam_leaf_l
        return in_r, normalize(torch.einsum("i,i,ij->j", out_p, mu_l @ in_l, mu_r) * step_scale, embed_dim)

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(io): return normalize(io[0] * io[1] * step_scale, embed_dim)

    pe = self
    pe = pe.fold_up_tree(f_in, lam_leaf)
    pe = pe.fold_up_tree(f_in_aux, (None, None))
    pe = pe.fold_down_tree(f_out, (pe.v[0], lam_root))
    pe = pe.map(f_mult)
    return pe

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam_leaf = tree_utils.init_tensor(embed_dim), # inside
    lam_root = tree_utils.init_tensor(embed_dim), # outside
    lam_leaf_l = tree_utils.init_tensor(embed_dim), # outside
    lam_leaf_r = tree_utils.init_tensor(embed_dim), # outside
    mu_l_scale = tree_utils.init_tensor(),
    mu_r_scale = tree_utils.init_tensor(),
  )
