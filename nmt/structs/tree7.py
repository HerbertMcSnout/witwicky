import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    params = [x.type(dtype) for x in params]
    mu_l, mu_r, lam = params
    def f(_, p, is_left): return (mu_l if is_left else mu_r) @ p
    return self.fold_down_tree(f, lam)

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  mu_l = torch.Tensor(embed_dim, embed_dim)
  mu_r = torch.Tensor(embed_dim, embed_dim)
  lam  = torch.Tensor(embed_dim)
  torch.nn.init.normal_(mu_l, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(mu_r, mean=0, std=embed_dim ** -0.5)
  torch.nn.init.normal_(lam, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam":lam}

def get_reg_penalty(batch_pe_norms):
  return tree_utils.reg_smooth(torch.exp(torch.abs(torch.log(batch_pe_norms))) - 1, 0.01)
