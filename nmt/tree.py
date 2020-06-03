import torch

def get_clamp_bound(t):
  """
  Makes sure that when you mv a square matrix with a vector,
  even if repeated many times, no individual values can explode.
  If d = t.size()[-1], then the max value should be d^-1, and
  the max value of an element in the returned vector is
    d*(d^-1)*(d^-1) = d^-1 = original max.
  """
  return t.size()[-1] ** -1 # -0.5

def clamp(t):
  bound = get_clamp_bound(t)
  return torch.clamp(t, -bound, bound)

def regularize(t): # apply softplus
  return t
  #return torch.log(1 + torch.exp(t)) * ((t.size()[-1] / 2) ** -0.5)

def parse_clean(fun_str, remove_parens=True):
  # fun_str\n -> fun_str
  if fun_str[-1] == "\n": fun_str = fun_str[:-1]

  fun_str = fun_str.strip()

  # (fun_str) -> fun_str
  if fun_str[0] == "(" and fun_str[-1] == ")" and remove_parens:
    fun_str = fun_str[1:-1].strip()

  return fun_str

#def normalize_columns(t):
#  "Normalizes all the columns in t"
#  return t / ((t ** 2).sum(dim=0) ** 0.5)

#class Tree:
#
#  def __init__(self, l=None, r=None, v=None):
#    self.l = l
#    self.r = r
#    self.v = v
#
#  def is_leaf(self):
#    return self.l is None # N.B. self.l is None  =>  self.r is None
#
#  def __str__h(self, strs):
#    if self.is_leaf():
#      strs.append(str(self.v))
#    elif self.r.is_leaf():
#      self.l.__str__h(strs)
#      self.r.__str__h(strs)
#    else:
#      self.l.__str__h(strs)
#      strs.append("(")
#      self.r.__str__h(strs)
#      strs.append(")")
#
#  def __str__(self):
#    strs = []
#    self.__str__h(strs)
#    return " ".join(strs)
#    #if self.is_leaf(): return str(self.v)
#    #elif self.r.is_leaf(): return str(self.l) + " " + str(self.r)
#    #else: return str(self.l) + " " + "( " + str(self.r) + " )"
#
#  def __repr__(self):
#    if self.is_leaf(): return repr(self.v)
#    return "(" + repr(self.l) + ", " + repr(self.r) + ")"
#
#  def to_tuple(self):
#    if self.is_leaf(): return self.v
#    else: return self.l.to_tuple(), self.r.to_tuple()
#
#  def shape(self):
#    return self.map(lambda _: 1).to_tuple()
#
#  def check(self):
#    if self.l is None and self.r is None:
#      return True
#    elif self.l is not None and self.r is not None:
#      return self.l.check() and self.r.check()
#    else:
#      return False
#
#  def map_(self, f):
#    if self.v is not None:
#      self.v = f(self.v)
#    if not self.is_leaf():
#      self.l.map_(f)
#      self.r.map_(f)
#    return self
#
#  def map(self, f):
#    if self.is_leaf(): return Tree(v=f(self.v))
#    else: return Tree(self.l.map(f),
#                      self.r.map(f),
#                      None if self.v is None else f(self.v))
#
#  def zip(self, other, enforce_size=True):
#    if self.is_leaf() and other.is_leaf(): return Tree(v=(self.v, other.v))
#    assert not (enforce_size and (self.is_leaf() or other.is_leaf())), "Can't zip trees of different shape"
#    return Tree(self.l.zip(other.l), self.r.zip(other.r), (self.v, other.v))
#
#  def get_pos_embedding_h(self, mu_u, mu_d, lam):
#    if self.is_leaf():
#      return Tree(v=lam)
#    else:
#      l = self.l.get_pos_embedding_h(mu_u, mu_d, lam)
#      v = mu_u @ l.v
#      r = self.r.get_pos_embedding_h(mu_u, mu_d, mu_d @ v)
#      return Tree(l, r, v)
#
#  def get_pos_embedding2(self, mu_u, mu_d, lam, max_len):
#    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#    mu_u = clamp(mu_u.type(dtype))
#    mu_d = clamp(mu_d.type(dtype))
#    lam = clamp(lam.type(dtype))
#    pe = self.get_pos_embedding_h(mu_u, mu_d, lam).flatten()
#    pe += [torch.zeros(lam.size()[-1]).type(dtype) for _ in range(max_len - len(pe))]
#    return torch.stack(pe)
#
#  def pos_embedding_inside(self, mu_l, mu_r, lam):
#    if self.is_leaf():
#      return Tree(v=clamp(lam))
#    else:
#      l = self.l.pos_embedding_inside(mu_l, mu_r, lam)
#      r = self.r.pos_embedding_inside(mu_l, mu_r, lam)
#      v = (mu_l @ l.v) * (mu_r @ r.v) * (lam.size()[-1])
#      return Tree(l, r, clamp(v))
#
#  def pos_embedding_outside(self, inside, mu_l, mu_r, p):
#    l = None
#    r = None
#    if not self.is_leaf():
#      lp = torch.einsum("i,ij,i->j", p, mu_l, mu_r @ inside.r.v) * (p.size()[-1])
#      #  = torch.einsum("i,ij,ik,k->j", p, mu_l, mu_r, inside.r.v) * (p.size()[-1])
#      rp = torch.einsum("i,i,ij->j", p, mu_l @ inside.l.v, mu_r) * (p.size()[-1])
#      #  = torch.einsum("i,ij,ik,j->k", p, mu_l, mu_r, inside.l.v) * (p.size()[-1])
#      l = self.l.pos_embedding_outside(inside.l, mu_l, mu_r, lp)
#      r = self.r.pos_embedding_outside(inside.r, mu_l, mu_r, rp)
#    return Tree(l, r, clamp(p))
#
#  def get_pos_embedding(self, mu_l, mu_r, lam, max_len):
#    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#    mu_l = mu_l.type(dtype) #clamp(mu_l.type(dtype))
#    mu_r = mu_r.type(dtype) #clamp(mu_r.type(dtype))
#    lam  =  lam.type(dtype) #clamp( lam.type(dtype))
#    inside = self.pos_embedding_inside(mu_l, mu_r, lam)
#    outside = self.pos_embedding_outside(inside, mu_l, mu_r, lam)
#    #pe = inside.zip(outside).map(lambda io: torch.mul(io[0], io[1])).flatten()
#    pe = [p / p.norm() for p in outside.flatten()]
#    pe += [torch.zeros(lam.size()[-1]).type(dtype)] * (max_len - len(pe))
#    return torch.stack(pe).type(dtype)
#    
##  def flatten(self, depth=0, path=0):
##    if self.is_leaf(): return [(self.v, depth, path)]
##    else: return self.l.flatten(depth + 1, (path << 1)) + self.r.flatten(depth + 1, (path << 1) + 1)
#
#  def flatten(self, only_leaf_values=True):
#    if self.is_leaf(): return [self.v]
#    elif not only_leaf_values and self.v is not None:
#      return self.l.flatten(only_leaf_values) + [self.v] + self.r.flatten(only_leaf_values)
#    else: return self.l.flatten(only_leaf_values) + self.r.flatten(only_leaf_values)
#
#  def size(self):
#    if self.is_leaf(): return 1
#    else: return self.l.size() + self.r.size()
#
#  def push(self, new):
#    if self.is_leaf() and not isinstance(new, Tree) and self.v is None:
#      self.v = new
#      return self
#    elif self.is_leaf():
#      self.l = Tree(v=self.v)
#    elif self.r is not None:
#      self.l = Tree(self.l, self.r, self.v)
#    self.v = None
#    self.r = new if isinstance(new, Tree) else Tree(v=new)
#    return self
#
#  def pop(self):
#    if self.is_leaf(): return self #.v
#    elif self.r is None: return self.l
#    else: return self.r
#
#
#
#def parse(fun_str):
#  if len(fun_str) == 0: return None
#  fun = Tree()
#  funs = []
#  word_start = None
#
#  fun_str = parse_clean(fun_str)
#
#  for i in range(len(fun_str)):
#    c = fun_str[i]
#
#    if word_start is not None and (c == " " or c == ")" or c == "("):
#      word = fun_str[word_start:i]
#      word_start = None
#      fun.push(word)
#
#    if c == "(":
#      funs.append(fun)
#      fun = fun.push(Tree()).pop()
#
#    elif c == ")":
#      fun = funs.pop()
#
#    elif word_start is None and c != " ":
#      word_start = i
#
#  if word_start is not None:
#    word = fun_str[word_start:]
#    fun.push(word)
#
#  assert len(funs) == 0, \
#      "Missing {} right parentheses in function {}".format(len(funs), fun_str)
#
#  #assert fun.check(), "Could not parse \"{}\" as {}".format(fun_str, repr(fun))
#
#  return fun
#
#def parse_file(fp):
#  with open(fp, "r") as f:
#    return [parse(line) for line in f.readlines()]


class Tree:
  
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

  def size(self):
    return len(self.flatten())

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

  def get_pos_embedding2(self, mu_l, mu_r, lam):
    embed_dim = lam.size()[-1]
    step_scale = embed_dim ** 0.5

    f_in = lambda _, l, r: (mu_l @ l) * (mu_r @ r) * step_scale

    def f_out(in_vlr, p, is_left):
      in_v, in_l, in_r = in_vlr
      in_p, out_p = p
      if is_left:
        in_r = in_r if in_r is not None else lam
        return in_l, torch.einsum("i,ij,i->j", out_p, mu_l, mu_r @ in_r) * step_scale
      else:
        in_l = in_l if in_l is not None else lam
        return in_r, torch.einsum("i,i,ij->j", out_p, mu_l @ in_l, mu_r) * step_scale

    f_in_aux = lambda v, l, r: (v, l[0], r[0])
    f_mult = lambda io: io[0] * io[1] * step_scale

    pe = self
    pe = pe.fold_up_tree(f_in, lam)
    pe = pe.fold_up_tree(f_in_aux, (None, None))
    pe = pe.fold_down_tree(f_out, (pe.v[0], lam))
    pe = pe.map(f_mult)
    pe = pe.flatten()
    pe = pe + [torch.zeros(embed_dim).type(dtype)] * (pad_len - len(pe))
    return torch.stack(pe).type(dtype)

  def get_pos_embedding(self, mu_l, mu_r, lam, pad_len):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    lam  =  lam.type(dtype)
    mu_l = mu_l.type(dtype)
    mu_r = mu_r.type(dtype)
    embed_dim = lam.size()[-1]
    f = lambda _, p, is_left: regularize((mu_l if is_left else mu_r) @ p)
    pe = self.fold_down_tree(f, lam).flatten()
    pe += [torch.zeros(embed_dim).type(dtype)] * (pad_len - len(pe))
    return torch.stack(pe).type(dtype)

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




#import time
#fp = "data/fun2com/dev.fun"
#start_time = time.time()
#funs = parse_file(fp)
#end_time = time.time()
#print("Parsing {} ({} lines) took {} seconds".format(fp, len(funs), end_time - start_time))
#for fun in funs[:5]:
#  print(fun)
#  print(fun.flatten())
#  print("")
