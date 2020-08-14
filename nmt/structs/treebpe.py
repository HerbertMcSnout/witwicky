import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

bpe_tok = '_'

class Tree(tree_utils.Tree):

  def map_(self, f):
    self.v = [f(v) for v in self.v]
    if self.l: self.l.map_(f)
    if self.r: self.r.map_(f)
    return self

  def map_array(self, f):
    v = f(self.v)
    l = self.l.map_array(f) if self.l else None
    r = self.r.map_array(f) if self.r else None
    return self.new(v, l, r)

  def map(self, f):
    v = [f(v) for v in self.v]
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
      acc.extend(node.v)
    return acc

  def __str__h(self, strs):
    if self.l:
      strs.append("(")
      strs.append(bpe_tok.join(str(v) for v in self.v))
      self.l.__str__h(strs)
      strs.append(")")
    else:
      strs.append(bpe_tok.join(str(v) for v in self.v))
    if self.r:
      self.r.__str__h(strs)

  def set_clip_length(self, clip):
    vs = 1 if isinstance(clip, str) or isinstance(clip, int) else len(self.v)
    if clip is None:
      return -1, self
    elif clip >= vs:
      clip -= vs
      l, r = None, None
      if clip > 0 and self.l:
        clip, l = self.l.set_clip_length(clip)
      if clip > 0 and self.r:
        clip, r = self.r.set_clip_length(clip)
      self.l, self.r = l, r
      return clip, self
    else:
      return clip, None

  def fold_up(self, f, leaf=None):
    return f(self.v, self.l.fold_up(f, leaf) if self.l else leaf, self.r.fold_up(f, leaf) if self.r else leaf)

  def fold_up_tree(self, f, leaf=None):
    return self.fold_up(lambda vs, l, r: self.new([f(v, (l.v if l else leaf), (r.v if r else leaf)) for v in vs], l, r))

  def fold_down_tree(self, f, root=None):
    l = self.l.fold_down_tree(f, f(self.v, root, True)) if self.l else None
    r = self.r.fold_down_tree(f, f(self.v, root, False)) if self.r else None
    return self.new([root for _ in self.v], l, r)

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam):
    def f(_, p, is_left): return (mu_l if is_left else mu_r) @ p
    return self.fold_down_tree(f, lam)


def parse(fun_str, clip=None):
  tree = tree_utils.parse(fun_str, cls=Tree, clip=clip)
  tree = tree.map_array(lambda x: x.split(bpe_tok))
  return tree_utils.maybe_clip(tree, clip)

def get_params(config):
  embed_dim = config['embed_dim']
  return dict(
    mu_l = tree_utils.init_tensor(embed_dim, embed_dim),
    mu_r = tree_utils.init_tensor(embed_dim, embed_dim),
    lam  = tree_utils.init_tensor(embed_dim),
  )

def get_reg_penalty(x, mask):
  x = x.norm(dim=-1) + ~mask # set all padding values to 1 so they get no penalty
  x = torch.max(x, 1/x) - 1
  x = x.sum()
  return x
