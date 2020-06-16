import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(t, embed_dim):
  norm = 1 if len(t.size()) == 1 else embed_dim ** 0.5
  return t * norm / t.norm()

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    params = [x.type(dtype) for x in params]
    mu_l, mu_r, lam, mu_l_scale, mu_r_scale, lam_scale = params
    mu_l = normalize(mu_l, embed_dim) * mu_l_scale
    mu_r = normalize(mu_r, embed_dim) * mu_r_scale
    lam  = normalize( lam, embed_dim) *  lam_scale
    def f(_, p, is_left):
      return normalize((mu_l if is_left else mu_r) @ p, embed_dim)
    return self.fold_down_tree(f, normalize(lam, embed_dim))

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  mu_l = torch.Tensor(embed_dim, embed_dim)
  mu_r = torch.Tensor(embed_dim, embed_dim)
  lam  = torch.Tensor(embed_dim)
  mu_l_scale = torch.tensor([1.])
  mu_r_scale = torch.tensor([1.])
  lam_scale  = torch.tensor([1.])
  torch.nn.init.orthogonal_(mu_l)
  torch.nn.init.orthogonal_(mu_r)
  torch.nn.init.normal_(lam, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam":lam, "mu_l_scale":mu_l_scale, "mu_r_scale":mu_r_scale, "lam_scale":lam_scale}
