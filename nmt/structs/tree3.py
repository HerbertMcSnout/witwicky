import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(t, embed_dim):
  norm = 1 if len(t.size()) == 1 else embed_dim ** 0.5
  t2 = torch.tanh(t * (embed_dim ** 0.5))
  return t2 * norm / t2.norm()

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, params):
    mu_l, mu_r, lam_root, lam_leaf_l, lam_leaf_r = params
    step_scale = embed_dim ** 0.5

    def reg(x): return normalize(x, embed_dim)

    def f_in(_, l, r):
      l = l if l is not None else lam_leaf_l
      r = r if r is not None else lam_leaf_r
      return reg((mu_l @ l) * (mu_r @ r) * step_scale)

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      in_p, out_p = p
      if is_left:
        in_r = in_r if in_r is not None else lam_leaf_r
        return in_l, reg(torch.einsum("i,ij,i->j", out_p, mu_l, mu_r @ in_r) * step_scale)
      else:
        in_l = in_l if in_l is not None else lam_leaf_l
        return in_r, reg(torch.einsum("i,i,ij->j", out_p, mu_l @ in_l, mu_r) * step_scale)

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(io): return io[0] * io[1] * step_scale

    pe = self
    pe = pe.fold_up_tree(f_in, None)
    pe = pe.fold_up_tree(f_in_aux, (None, None))
    pe = pe.fold_down_tree(f_out, (pe.v[0], lam_root))
    pe = pe.map(f_mult)
    pe = pe.map(reg)
    return pe

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(args):
  embed_dim = args['embed_dim']
  mu_l = tree_utils.init_tensor(embed_dim, embed_dim)
  mu_r = tree_utils.init_tensor(embed_dim, embed_dim)
  lam_root   = tree_utils.init_tensor(embed_dim) # outside
  lam_leaf_l = tree_utils.init_tensor(embed_dim) # outside
  lam_leaf_r = tree_utils.init_tensor(embed_dim) # outside
  
  #torch.nn.init.orthogonal_(mu_l)
  #torch.nn.init.orthogonal_(mu_r)
  #torch.nn.init.normal_(lam_leaf_l, mean=0, std=embed_dim ** -0.5)
  #torch.nn.init.normal_(lam_leaf_r, mean=0, std=embed_dim ** -0.5)
  #torch.nn.init.normal_(lam_root, mean=0, std=embed_dim ** -0.5)
  #torch.nn.init.normal_(lam_leaf_l, mean=0, std=embed_dim ** -0.5)
  #torch.nn.init.normal_(lam_leaf_r, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam_root":lam_root, "lam_leaf_l":lam_leaf_l, "lam_leaf_r":lam_leaf_r}
