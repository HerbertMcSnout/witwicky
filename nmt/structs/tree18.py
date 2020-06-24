import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, params):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    params = [x.type(dtype) for x in params]
    mu_l, mu_r, lam, lam_l, lam_r = params
    def f_down(_, p, is_left):
      return (mu_l if is_left else mu_r) @ p
    def f_up(_, l, r):
      lv = (mu_l @ l) if l is not None else lam_l
      rv = (mu_r @ r) if r is not None else lam_r
      return lv * rv * (embed_dim ** 0.5)
    d = self.fold_down_tree(f_down, lam)
    u = self.fold_up_tree(f_up)
    return d.zip(u).map(lambda x: sum(x) * (2**-0.5))

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  mu_l = torch.Tensor(embed_dim, embed_dim)
  mu_r = torch.Tensor(embed_dim, embed_dim)
  lam = torch.Tensor(embed_dim)
  lam_l = torch.Tensor(embed_dim)
  lam_r = torch.Tensor(embed_dim)
  torch.nn.init.orthogonal_(mu_l)
  torch.nn.init.orthogonal_(mu_r)
  torch.nn.init.normal_(lam, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_l, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam_r, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam":lam, "lam_l":lam_l, "lam_r":lam_r}

def get_reg_penalty(x):
  return torch.max(x, 1/x) - 1
