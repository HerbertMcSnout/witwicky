import torch
from .struct import Struct
import nmt.structs.tree_utils as tree_utils

def normalize(t, embed_dim):
  norm = 1 if len(t.size()) == 1 else embed_dim ** 0.5
  t2 = torch.tanh(t * (embed_dim ** 0.5))
  return t2 * norm / t2.norm()

class Tree(tree_utils.Tree):
  
  def get_pos_embedding(self, embed_dim, params):
    params = [normalize(x, embed_dim) for x in params] # vectors -> 1, matrices -> sqrt(d)
    mu_l, mu_r, lam = params
    def f(_, p, is_left):
      return normalize((mu_l if is_left else mu_r) @ p, embed_dim)
    return self.fold_down_tree(f, lam)

def parse(fun_str):
  return tree_utils.parse(fun_str, cls=Tree)

def get_params(config):
  embed_dim = config['embed_dim']
  mu_l = tree_utils.init_tensor(embed_dim, embed_dim)
  mu_r = tree_utils.init_tensor(embed_dim, embed_dim)
  lam  = tree_utils.init_tensor(embed_dim)
  #torch.nn.init.orthogonal_(mu_l)
  #torch.nn.init.orthogonal_(mu_r)
  #torch.nn.init.normal_(lam, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam":lam}
