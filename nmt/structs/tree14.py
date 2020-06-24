import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    params = [x.type(dtype) for x in params]
    mu_l, mu_r, lam_leaf, lam_root, lam_leaf_l, lam_leaf_r = params
    step_scale = embed_dim ** 0.5

    def f_in(_, l, r): return (mu_l @ l) * (mu_r @ r) * step_scale

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      in_p, out_p = p
      if is_left:
        in_r = in_r if in_r is not None else lam_leaf_r
        return in_l, torch.einsum("i,ij,i->j", out_p, mu_l, mu_r @ in_r) * step_scale
      else:
        in_l = in_l if in_l is not None else lam_leaf_l
        return in_r, torch.einsum("i,i,ij->j", out_p, mu_l @ in_l, mu_r) * step_scale

    def f_in_aux(v, l, r): return v, l[0], r[0]
    def f_mult(io): return io[0] * io[1] * step_scale

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
  
  torch.nn.init.orthogonal_(mu_l)
  torch.nn.init.orthogonal_(mu_r)
  torch.nn.init.normal_(lam_leaf, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_root, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_leaf_l, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_leaf_r, mean=0, std=embed_dim ** -0.5)
  #self.pos_embedding_linear = Parameter(torch.Tensor(max_pos_length, embed_dim))
  #torch.nn.init.normal_(self.pos_embedding_linear, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam_leaf":lam_leaf, "lam_root":lam_root, "lam_leaf_l":lam_leaf_l, "lam_leaf_r":lam_leaf_r}

def get_reg_penalty(x):
  # smooth min/max for some small, nonzero number as min and large number as max for batch_pe_norms
  #eps = 0.1
  #k = 0.25
  #return torch.max(torch.max(x - 1, (x + eps) ** -1 - (1 + eps) ** -1) - k, torch.tensor([0]).type(x.type()))

  #k = 0.01
  
  #t = x + k
  
  #return t + 1/t - 2

  #return 1 - torch.min(x, x ** -1)

  eps_b = 0.05
  eps_h = 0.01

  return torch.clamp((x + eps_h) + 1/(x + eps_h) - 2 - eps_b, min=0.0)
