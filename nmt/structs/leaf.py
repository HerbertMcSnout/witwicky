import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class Tree(tree_utils.Tree):

  def map_(self, f):
    if self.l: self.l = self.l.map_(f)
    else: self.v = f(self.v) # left-child, right-sibling
    if self.r: self.r = self.r.map_(f)
    return self

  def map(self, f):
    v = f(self.v) if not self.l else self.v
    l = self.l.map(f) if self.l else None
    r = self.r.map(f) if self.r else None
    return self.new(v, l, r)

  def flatten(self):
    stack = [self]
    acc = []
    while stack:
      node = stack.pop()
      if node.r: stack.append(node.r)
      if node.l: stack.append(node.l)
      else: acc.append(node.v) # left-child, right-sibling
    return acc

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam):
    def f(_, p, is_left): return (mu_l if is_left else mu_r) @ p
    return self.fold_down_tree(f, lam)

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam  = tree_utils.init_tensor(embed_dim),
  )

