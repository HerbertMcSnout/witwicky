import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

max_src_length = -1

class Tree(tree_utils.Tree):

  def get_pe_h(self, abs_pe, i):
    li = 2*i + 1
    ri = 2*i + 2
    if ri >= max_src_length: ri = i
    if li >= max_src_length: li = i
    l = self.l.get_pe_h(abs_pe, li) if self.l else None
    r = self.r.get_pe_h(abs_pe, ri) if self.r else None
    return self.new(abs_pe[i, :], l, r)

  def get_pos_embedding(self, embed_dim, abs_pe):
    return self.get_pe_h(abs_pe, 0)


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


def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  global max_src_length
  max_src_length = config['max_src_length']
  embed_dim = config['embed_dim']
  abs_pe = torch.empty(max_src_length, embed_dim, dtype=torch.float, device=ut.get_device())
  torch.nn.init.normal_(abs_pe, mean=0, std=embed_dim ** -0.5)

  return dict(
    abs_pe = abs_pe,
  )
