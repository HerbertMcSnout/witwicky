import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    mu_l, mu_r, lam_leaf, lam_root, lam_leaf_l, lam_leaf_r = params

    def scale(x):
      return x
      #return x * (embed_dim ** 0.5)
      #return x / x.sum()

    def f_in(_, l, r): return scale((mu_l @ l) * (mu_r @ r))

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      in_p, out_p = p
      if is_left:
        #in_r = in_r if in_r is not None else lam_leaf_r
        #return in_l, scale(torch.einsum("i,ij,i->j", out_p, mu_l, (mu_r @ in_r)))
        return in_l, scale(torch.einsum("i,ij,ik,k->j", out_p, mu_l, mu_r, in_r))
      else:
        #in_l = in_l if in_l is not None else lam_leaf_l
        #return in_r, scale(torch.einsum("i,i,ij->j", out_p, (mu_l @ in_l), mu_r))
        return in_r, scale(torch.einsum("i,ij,ik,j->k", out_p, mu_l, mu_r, in_l))

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(io): return scale(io[0] * io[1])

    pe = self
    pe = pe.fold_up_tree(f_in, lam_leaf)
    pe = pe.fold_up_tree(f_in_aux, (lam_leaf_l, lam_leaf_r))
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
    lam_leaf   = tree_utils.init_tensor(embed_dim), # inside
    lam_root   = tree_utils.init_tensor(embed_dim), # outside
    lam_leaf_l = tree_utils.init_tensor(embed_dim), # outside
    lam_leaf_r = tree_utils.init_tensor(embed_dim), # outside
  )

#def get_reg_penalty(x):
#  return x * 0.0 # torch.max(x, 1/x) - 1.0
