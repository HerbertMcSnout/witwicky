import torch
import nmt.utils as ut
from .struct import Struct

class Record(object):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

class Tree(Struct):
  
  def __init__(self, v, l=None, r=None):
    self.l = l
    self.r = r
    self.v = v

  def new(self, *args, **kwargs):
    return self.__class__(*args, **kwargs)

  def __str__h(self, strs):
    if self.l:
      strs.append("(")
      strs.append(str(self.v))
      self.l.__str__h(strs)
      strs.append(")")
    else:
      strs.append(str(self.v))
    if self.r:
      self.r.__str__h(strs)
      
  def __str__(self):
    strs = []
    self.__str__h(strs)
    return " ".join(strs)

  def __repr__(self):
    return self.__str__()

  def map_(self, f):
    self.v = f(self.v)
    if self.l: self.l = self.l.map_(f)
    if self.r: self.r = self.r.map_(f)
    return self

  def map(self, f):
    v = f(self.v)
    l = self.l.map(f) if self.l else None
    r = self.r.map(f) if self.r else None
    return self.new(v, l, r)

  def flatten(self, acc=None, lefts=[]):
    if acc is None:
      acc = []
      self.flatten(acc)
      return acc
    else:
      acc.append(self.v)
      if self.l: lefts.append(self.l)
      if self.r: self.r.flatten(acc, lefts)
      elif len(lefts) > 0: lefts.pop().flatten(acc, lefts)

  def fold_up(self, f, leaf=None):
    return f(self.v, self.l.fold_up(f, leaf) if self.l else leaf, self.r.fold_up(f, leaf) if self.r else leaf)

  def fold_up_tree(self, f, leaf=None):
    l = self.l.fold_up_tree(f, leaf) if self.l else None
    r = self.r.fold_up_tree(f, leaf) if self.r else None
    lv = l.v if self.l else leaf
    rv = r.v if self.r else leaf
    v = f(self.v, lv, rv)
    return self.new(v, l, r)

  def fold_down_tree(self, f, root=None):
    l = self.l.fold_down_tree(f, f(self.v, root, True)) if self.l else None
    r = self.r.fold_down_tree(f, f(self.v, root, False)) if self.r else None
    return self.new(root, l, r)

  def zip(self, other):
    "Zips the node values of this tree with other's"
    assert ((self.l is None) == (other.l is None)) \
       and ((self.r is None) == (other.r is None)), \
       "Trying to zip two trees of different shape"
    v = self.v, other.v
    l = self.l.zip(other.l) if self.l else None
    r = self.r.zip(other.r) if self.r else None
    return self.new(v, l, r)

def parse_clean(fun_str, remove_parens=True):
  # fun_str\n -> fun_str
  if fun_str[-1] == "\n": fun_str = fun_str[:-1]

  fun_str = fun_str.strip()

  # (fun_str) -> fun_str
  if fun_str[0] == "(" and fun_str[-1] == ")" and remove_parens:
    fun_str = fun_str[1:-1].strip()

  return fun_str

def parse_lc_rs_h(fun_str):
  children = []
  start = 0
  pos = 0
  while pos < len(fun_str):
    c = fun_str[pos]
    pos += 1
    if c == " " or c == "(" or c == ")":
      if start != pos - 1:
        children.append(fun_str[start:pos - 1])
      if c == "(":
        child, end = parse_lc_rs_h(fun_str[pos:])
        children.append(child)
        start = pos = pos + end
      elif c == ")": return children, pos
      else: start = pos

  if start != pos:
    children.append(fun_str[start:pos])

  return children, pos

def construct_tree(children, siblings=[], cls=Tree):
  return cls(children if isinstance(children, str) else children[0],
             construct_tree(children[1], children[2:], cls=cls) if len(children) > 1 and not isinstance(children, str) else None,
             construct_tree(siblings[0], siblings[1:], cls=cls) if len(siblings) > 0 else None)

def parse(fun_str, cls=Tree):
  return construct_tree(parse_lc_rs_h(parse_clean(fun_str))[0], cls=cls)

def parse_no_binarization(fun_str):
  return parse_lc_rs_h(parse_clean(fun_str))[0]

def init_tensor(*size):
  device = ut.get_device()
  if len(size) == 0:
    t = torch.tensor([1.], device=device)
  if len(size) == 1:
    t = torch.empty(*size, device=device)
    torch.nn.init.normal_(t, mean=0, std=size[0]**-0.5)
  elif len(size) == 2:
    t = torch.empty(*size, device=device)
    torch.nn.init.orthogonal_(t)
  else:
    assert False, "nmt.structs.tree_utils.init_tensor(*size) only implemented for len(size) == 0, 1, and 2, but got len({}) = {}".format(size, len(size))
  return t

def reg_smooth(x, eps):
  "sqrt(x^2 + eps) - sqrt(eps)"
  return (x**2 + eps)**0.5 - eps**0.5

def reg_smooth2(x, eps):
  "x*tanh(eps*x)"
  return x * torch.tanh(eps * x)


#  def fold_up_iterative(self, f, leaf=None):
#    stack = [(self, True)]
#    resolved = []
#    while stack:
#      last, is_node = stack.pop()
#      if last is None:
#        resolved.append(leaf)
#      elif is_node:
#        stack.append((last.v, False))
#        stack.append((last.l, True))
#        stack.append((last.r, True))
#      else:
#        l = resolved.pop()
#        r = resolved.pop()
#        resolved.append(f(last, l, r))
#    return resolved[0]
