import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(t, embed_dim):
  norm = 1 if len(t.size()) == 1 else embed_dim ** 0.5
  return t * norm / t.norm()

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    params = [normalize(x.type(dtype), embed_dim) for x in params[:-2]] + [x.type(dtype) for x in params[-2:]]
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
  mu_l = torch.Tensor(embed_dim, embed_dim)
  mu_r = torch.Tensor(embed_dim, embed_dim)
  lam_leaf   = torch.Tensor(embed_dim) # inside
  lam_root   = torch.Tensor(embed_dim) # outside
  lam_leaf_l = torch.Tensor(embed_dim) # outside
  lam_leaf_r = torch.Tensor(embed_dim) # outside
  mu_l_scale = torch.tensor([1.])
  mu_r_scale = torch.tensor([1.])
  
  torch.nn.init.orthogonal_(mu_l)
  torch.nn.init.orthogonal_(mu_r)
  torch.nn.init.normal_(lam_leaf, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_root, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_leaf_l, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_leaf_r, mean=0, std=embed_dim ** -0.5)
  #self.pos_embedding_linear = Parameter(torch.Tensor(max_pos_length, embed_dim))
  #torch.nn.init.normal_(self.pos_embedding_linear, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam_leaf":lam_leaf, "lam_root":lam_root, "lam_leaf_l":lam_leaf_l, "lam_leaf_r":lam_leaf_r,
          "mu_l_scale":mu_l_scale, "mu_r_scale":mu_r_scale}
