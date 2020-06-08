import torch
from .struct import Struct

class Tree(Struct):
  
  def __init__(self, v, l=None, r=None):
    self.l = l
    self.r = r
    self.v = v

  def has_left(self):
    return self.l is not None

  def has_right(self):
    return self.r is not None

  def __str__h(self, strs):
    if self.has_left():
      strs.append("(")
      strs.append(str(self.v))
      self.l.__str__h(strs)
      strs.append(")")
    else:
      strs.append(str(self.v))
    if self.has_right():
      self.r.__str__h(strs)
      
  def __str__(self):
    strs = []
    self.__str__h(strs)
    return " ".join(strs)

  def __repr__(self):
    return self.__str__()

  def map_(self, f):
    self.v = f(self.v)
    if self.has_left(): self.l = self.l.map_(f)
    if self.has_right(): self.r = self.r.map_(f)
    return self

  def map(self, f):
    v = f(self.v)
    l = self.l.map(f) if self.has_left() else None
    r = self.r.map(f) if self.has_right() else None
    return Tree(v, l, r)

  def flatten(self, acc=None, lefts=[]):
    if acc is None:
      acc = []
      self.flatten(acc)
      return acc
    else:
      acc.append(self.v)
      if self.has_left(): lefts.append(self.l)
      if self.has_right(): self.r.flatten(acc, lefts)
      elif len(lefts) > 0: lefts.pop().flatten(acc, lefts)

  def fold_up(self, f, leaf=None):
    return f(self.v, self.l.fold_up(f, leaf) if self.has_left() else leaf, self.r.fold_up(f, leaf) if self.has_right() else leaf)

  def fold_up_tree(self, f, leaf=None):
    l = self.l.fold_up_tree(f, leaf) if self.has_left() else None
    r = self.r.fold_up_tree(f, leaf) if self.has_right() else None
    lv = l.v if self.has_left() else leaf
    rv = r.v if self.has_right() else leaf
    v = f(self.v, lv, rv)
    return Tree(v, l, r)

  def fold_down_tree(self, f, root=None):
    l = self.l.fold_down_tree(f, f(self.v, root, True)) if self.has_left() else None
    r = self.r.fold_down_tree(f, f(self.v, root, False)) if self.has_right() else None
    return Tree(root, l, r)

  def zip(self, other):
    "Zips the node values of this tree with other's"
    assert (self.has_left() == other.has_left) \
       and (self.has_right() == other.has_right()), \
       "Trying to zip two trees of different shape"
    v = self.v, other.v
    l = self.l.zip(other.l) if self.has_left() else None
    r = self.r.zip(other.r) if self.has_right() else None
    return Tree(v, l, r)

  def get_pos_embedding(self, embed_dim, params):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    params = [x.type(dtype) for x in params]
    mu_l, mu_r, lam = params
    f = lambda _, p, is_left: (mu_l if is_left else mu_r) @ p # regularize(mu_... @ p)
    #pe = self.fold_down_tree(f, lam).flatten()
    #pe += [torch.zeros(embed_dim).type(dtype)] * (pad_len - len(pe))
    #return torch.stack(pe).type(dtype)
    return self.fold_down_tree(f, lam)



#def get_clamp_bound(t):
#  """
#  Makes sure that when you mv a square matrix with a vector,
#  even if repeated many times, no individual values can explode.
#  If d = t.size()[-1], then the max value should be d^-1, and
#  the max value of an element in the returned vector is
#    d*(d^-1)*(d^-1) = d^-1 = original max.
#  """
#  return t.size()[-1] ** -1 # -0.5

#def clamp(t):
#  bound = get_clamp_bound(t)
#  return torch.clamp(t, -bound, bound)

#def regularize(t): # apply softplus
#  return t
#  #return torch.log(1 + torch.exp(t)) * ((t.size()[-1] / 2) ** -0.5)


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

def construct_tree(children, siblings=[]):
  return Tree(children if isinstance(children, str) else children[0],
               construct_tree(children[1], children[2:]) if len(children) > 1 and not isinstance(children, str) else None,
               construct_tree(siblings[0], siblings[1:]) if len(siblings) > 0 else None)

def parse(fun_str):
  return construct_tree(parse_lc_rs_h(parse_clean(fun_str))[0])

def get_params(config):
  embed_dim = config['embed_dim']
  max_len = config['max_train_length']
  mu_l = torch.Tensor(embed_dim, embed_dim)
  mu_r = torch.Tensor(embed_dim, embed_dim)
  lam  = torch.Tensor(embed_dim)
  torch.nn.init.orthogonal_(mu_l)
  torch.nn.init.orthogonal_(mu_r)
  torch.nn.init.normal_(lam, mean=0, std=embed_dim ** -0.5)
  #self.pos_embedding_linear = Parameter(torch.Tensor(max_pos_length, embed_dim))
  #torch.nn.init.normal_(self.pos_embedding_linear, mean=0, std=embed_dim ** -0.5)
  return {"mu_l":mu_l, "mu_r":mu_r, "lam":lam}
