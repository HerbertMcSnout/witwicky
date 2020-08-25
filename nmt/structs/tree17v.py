import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam):
    def f(_, p, is_left): return (mu_l if is_left else mu_r) * p
    return self.fold_down_tree(f, lam)

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = torch.nn.init.normal_(torch.empty(embed_dim), mean=0, std=1),
    mu_r = torch.nn.init.normal_(torch.empty(embed_dim), mean=0, std=1),
    lam  = torch.nn.init.normal_(torch.empty(embed_dim), mean=0, std=1) * (embed_dim ** -0.5),
  )

