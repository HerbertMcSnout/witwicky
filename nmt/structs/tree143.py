import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam_leaf, lam_root, lam_leaf_l, lam_leaf_r):
    lam_lf_s = lam_leaf.sum()
    lam_lf = lam_leaf / lam_lf_s, lam_lf_s
    lam_rt_s = lam_root.sum()
    lam_rt = lam_root / lam_rt_s, lam_rt_s
    lam_lf_l_s = lam_leaf_l.sum()
    lam_lf_r_s = lam_leaf_r.sum()
    lam_lf_l = lam_leaf_l / lam_lf_l_s, lam_lf_l_s
    lam_lf_r = lam_leaf_r / lam_lf_r_s, lam_lf_r_s

    def f_in(_, l, r):
      li, ls = l
      ri, rs = r
      i = (mu_l @ li) * (mu_r @ ri)
      z = i.sum()
      i /= z
      s = torch.log(z) + ls + rs
      return i, s

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      (in_p, in_p_s), (out_p, out_p_s) = p
      if is_left:
        in_r, in_r_s = in_r if in_r is not None else lam_lf_r
        o_l = torch.einsum("i,ij,i->j", out_p, mu_l, mu_r @ in_r)
        z = o_l.sum()
        o_l /= z
        o_l_s = out_p_s + in_r_s + torch.log(z)
        return in_l, (o_l, o_l_s)
      else:
        in_l, in_l_s = in_l if in_l is not None else lam_lf_l
        o_r = torch.einsum("i,i,ij->j", out_p, mu_l @ in_l, mu_r)
        z = o_r.sum()
        o_r /= z
        o_r_s = out_p_s + in_l_s + torch.log(z)
        return in_r, (o_r, o_r_s)

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(io):
      (in_n, in_n_s), (out_n, out_n_s) = io
      return torch.log(in_n * out_n) + in_n_s + out_n_s

    pe = self
    pe = pe.fold_up_tree(f_in, lam_lf)
    pe = pe.fold_up_tree(f_in_aux, (None, None))
    pe = pe.fold_down_tree(f_out, (pe.v[0], lam_rt))
    pe = pe.map(f_mult)
    return pe

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

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
