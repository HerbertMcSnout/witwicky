import torch
import nmt.utils as ut
from nmt.structs.struct import Struct

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
  
  def has_left(self):
    return bool(self.l)
  def has_right(self):
    return bool(self.r)
  def is_leaf(self):
    return not (self.l or self.r)

  def flatten(self):
    stack = [self]
    acc = []
    while stack:
      node = stack.pop()
      if node.r: stack.append(node.r)
      if node.l: stack.append(node.l)
      acc.append(node.v)
    return acc

  def set_clip_length(self, clip):
    if clip is None:
      return -1, self
    elif clip:
      clip -= 1
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
    return self.fold_up(lambda v, l, r: self.new(f(v, (l.v if l else leaf), (r.v if r else leaf)), l, r))

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


def construct_tree(tree, cls=Tree):
  stack = [tree]
  resolved = []
  while stack:
    node = stack.pop()
    if isinstance(node, str):
      resolved.append(cls(node))
    elif isinstance(node, list):
      value = node[0]
      children = node[1:]
      if children:
        stack.append((value, len(children)))
        stack.extend(reversed(children))
      else:
        resolved.append(cls(value))
    elif isinstance(node, tuple):
      value, num_children = node
      resolved, children = resolved[:-num_children], resolved[-num_children:]
      t = cls(value, children[0], None)
      t2 = t.l
      for child in children[1:]:
        t2.r = child
        t2 = t2.r
      resolved.append(t)
  return resolved.pop()

#def construct_tree(children, siblings=[], cls):
#  return cls(children if isinstance(children, str) else children[0],
#             construct_tree(children[1], children[2:], cls) if len(children) > 1 and not isinstance(children, str) else None,
#             construct_tree(siblings[0], siblings[1:], cls) if len(siblings) > 0 else None)

def maybe_clip(tree, clip):
  return tree.set_clip_length(clip)[1] if clip else tree

def parse(fun_str, cls=Tree, clip=None):
  return maybe_clip(construct_tree(parse_lc_rs_h(parse_clean(fun_str))[0], cls=cls), clip)

def parse_no_binarization(fun_str):
  return parse_lc_rs_h(parse_clean(fun_str))[0]

def init_tensor(*size):
  device = ut.get_device()
  if len(size) == 0:
    t = torch.tensor([1.], device=device)
  elif len(size) == 1:
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


#def flatten_mask_left(tree, i, size, acc):
#  nv = [1 << HEAD_OTHER_ID] * size
#  nv[i] = 1 << HEAD_SELF_ID
#  acc.append(nv)
#  i += 1
#  if tree.l:
#    j = flatten_mask_left(tree.l, i, size, acc)
#    acc[i - 1][i : j] = [1 << HEAD_CHILD_ID] * (j - i)
#    i = j
#  if tree.r:
#    i = flatten_mask_left(tree.r, i, size, acc)
#  return i

HEAD_IDS = [1 << x for x in range(9)]
HEAD_PAD_ID, HEAD_SELF_ID, HEAD_OTHER_ID, HEAD_CHILD_ID, HEAD_PARENT_ID, HEAD_SIB_ID, HEAD_ANCE_ID, HEAD_DESC_ID, HEAD_EXTRA_ID = HEAD_IDS

HEAD_BASE_IDS = HEAD_SELF_ID | HEAD_EXTRA_ID
HEAD_ALL_IDS = 0
for HEAD_ID in HEAD_IDS[1:]:
  HEAD_ALL_IDS |= HEAD_ID

def flatten_mask_left2(tree, i, mask):
  k = i
  i += 1
  mask[k, :] = HEAD_OTHER_ID
  mask[k, k] = HEAD_SELF_ID

  if tree.l:
    j = flatten_mask_left2(tree.l, i, mask)
    mask[k, i : j] = HEAD_DESC_ID
    mask[i : j, k] = HEAD_ANCE_ID
    children = (mask[i, :] & (HEAD_SELF_ID | HEAD_SIB_ID)).type(torch.bool)
    mask[k, :].masked_fill_(children, HEAD_CHILD_ID)
    mask[:, k].masked_fill_(children, HEAD_PARENT_ID)
    i = j

  if tree.r:
    j = flatten_mask_left2(tree.r, i, mask)
    siblings = (mask[i, :] & (HEAD_SELF_ID | HEAD_SIB_ID)).type(torch.bool)
    mask[k, :].masked_fill_(siblings, HEAD_SIB_ID)
    mask[:, k].masked_fill_(siblings, HEAD_SIB_ID)
    i = j
  return i


def get_enc_mask(toks, structs, num_heads):
  bsz, src_len = toks.size()
  masks = torch.full((bsz, src_len, src_len), HEAD_PAD_ID, dtype=torch.int, device=ut.get_device())
  
  for c in range(bsz):
    size = structs[c].size()
    flatten_mask_left2(structs[c], 0, masks[c, :size, :size])
    masks[c, size:, :] = HEAD_EXTRA_ID

  if num_heads == 1: return masks.unsqueeze(1)
  else: return masks.unsqueeze(1).expand(-1, num_heads, -1, -1).clone()

def test(tree_str, num_heads=1):
  tree = parse(tree_str)
  print(tree)
  toks = torch.Tensor(num_heads, tree.size())
  print(get_enc_mask(toks, [tree], num_heads))
