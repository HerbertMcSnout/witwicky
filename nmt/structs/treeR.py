import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam):
    def f(_, p, is_left): return (mu_l if is_left else mu_r) @ p
    return self.fold_down_tree(f, lam)

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  state_dict = torch.load('nmt/saved_models/en2vi_tree/en2vi_tree.pth', map_location=ut.get_device())['model']
  return dict(
    mu_l = state_dict['mu_l'],
    mu_r = state_dict['mu_r'],
    lam  = state_dict['lam' ],
  )

