import torch

#def renormalize(t):
#  return t / t.norm()

def clamp(t):
  bound = t.size()[-1] ** -0.25
  return torch.clamp(t, -bound, bound)

def normalize_columns(t):
  "Normalizes all the columns in t"
  return t / ((t ** 2).sum(dim=0) ** 0.5)

class Tree:

  def __init__(self, l=None, r=None, v=None):
    self.l = l
    self.r = r
    self.v = v

  def is_leaf(self):
    return self.l is None # N.B. self.l is None  =>  self.r is None

  def __str__h(self, strs):
    if self.is_leaf():
      strs.append(str(self.v))
    elif self.r.is_leaf():
      self.l.__str__h(strs)
      self.r.__str__h(strs)
    else:
      self.l.__str__h(strs)
      strs.append("(")
      self.r.__str__h(strs)
      strs.append(")")

  def __str__(self):
    strs = []
    self.__str__h(strs)
    return " ".join(strs)
    #if self.is_leaf(): return str(self.v)
    #elif self.r.is_leaf(): return str(self.l) + " " + str(self.r)
    #else: return str(self.l) + " " + "( " + str(self.r) + " )"

  def __repr__(self):
    if self.is_leaf(): return repr(self.v)
    return "(" + repr(self.l) + ", " + repr(self.r) + ")"

  def to_tuple(self):
    if self.is_leaf(): return self.v
    else: return self.l.to_tuple(), self.r.to_tuple()

  def shape(self):
    return self.map(lambda _: 1).to_tuple()

  def check(self):
    if self.l is None and self.r is None:
      return True
    elif self.l is not None and self.r is not None:
      return self.l.check() and self.r.check()
    else:
      return False

  def map_(self, f):
    if self.v is not None:
      self.v = f(self.v)
    if not self.is_leaf():
      self.l.map_(f)
      self.r.map_(f)
    return self

  def map(self, f):
    if self.is_leaf(): return Tree(v=f(self.v))
    else: return Tree(self.l.map(f),
                      self.r.map(f),
                      None if self.v is None else f(self.v))

  def zip(self, other, enforce_size=True):
    if self.is_leaf() and other.is_leaf(): return Tree(v=(self.v, other.v))
    assert not (enforce_size and (self.is_leaf() or other.is_leaf())), "Can't zip trees of different shape"
    return Tree(self.l.zip(other.l), self.r.zip(other.r), (self.v, other.v))

  def get_pos_embedding_h(self, mu_u, mu_d, lam):
    if self.is_leaf():
      return Tree(v=clamp(lam))
    else:
      l = self.l.get_pos_embedding_h(mu_u, mu_d, lam)
      v = mu_u @ l.v
      r = self.r.get_pos_embedding_h(mu_u, mu_d, mu_d @ v)
      return Tree(l, r, clamp(v))

  def get_pos_embedding(self, mu_u, mu_d, lam, max_len):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    pe = self.get_pos_embedding_h(mu_u.type(dtype), mu_d.type(dtype), lam.type(dtype)).flatten()
    pe += [torch.zeros(lam.size()[-1]).type(dtype) for _ in range(max_len - len(pe))]
    return torch.stack(pe)

#  def pos_embedding_inside(self, mu_l, mu_r, lam):
#    if self.is_leaf():
#      return Tree(v=renormalize(lam))
#    else:
#      l = self.l.pos_embedding_inside(mu_l, mu_r, lam)
#      r = self.r.pos_embedding_inside(mu_l, mu_r, lam)
#      v = (mu_l @ l.v) * (mu_r @ r.v) * (lam.size()[-1] ** 0.5)
#      return Tree(l, r, renormalize(v))
#
#  def pos_embedding_outside(self, inside, mu_l, mu_r, p):
#    l = None
#    r = None
#    if not self.is_leaf():
#      lp = torch.einsum("i,ij,i->j", p, mu_l, mu_r @ inside.r.v) * (lam.size()[-1] ** 0.5)
#      rp = torch.einsum("i,i,ij->j", p, mu_l @ inside.l.v, mu_r) * (lam.size()[-1] ** 0.5)
#      #lp = torch.einsum("i,ij,ik,k->j", p, mu_l, mu_r, inside.r.v) * (lam.size()[-1] ** 0.5)
#      #rp = torch.einsum("i,ij,ik,j->k", p, mu_l, mu_r, inside.l.v) * (lam.size()[-1] ** 0.5)
#      l = self.l.pos_embedding_outside(inside.l, mu_l, mu_r, lp)
#      r = self.r.pos_embedding_outside(inside.r, mu_l, mu_r, rp)
#    return Tree(l, r, renormalize(p))
#
#  def get_pos_embedding(self, mu_l, mu_r, lam, max_len):
#    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#    inside = self.pos_embedding_inside(mu_l, mu_r, lam)
#    outside = self.pos_embedding_outside(inside, mu_l, mu_r, lam)
#    #pe = inside.zip(outside).map(lambda io: torch.mul(io[0], io[1])).flatten()
#    pe = outside.flatten()
#    pe += [torch.zeros(lam.size()[-1]).type(dtype)] * (max_len - len(pe))
#    return torch.stack(pe).type(dtype)
    
#  def flatten(self, depth=0, path=0):
#    if self.is_leaf(): return [(self.v, depth, path)]
#    else: return self.l.flatten(depth + 1, (path << 1)) + self.r.flatten(depth + 1, (path << 1) + 1)

  def flatten(self, only_leaf_values=True):
    if self.is_leaf(): return [self.v]
    elif not only_leaf_values and self.v is not None:
      return self.l.flatten(only_leaf_values) + [self.v] + self.r.flatten(only_leaf_values)
    else: return self.l.flatten(only_leaf_values) + self.r.flatten(only_leaf_values)

  def weight(self):
    if self.is_leaf(): return 1
    else: return self.l.weight() + self.r.weight()

  def push(self, new):
    if self.is_leaf() and not isinstance(new, Tree) and self.v is None:
      self.v = new
      return self
    elif self.is_leaf():
      self.l = Tree(v=self.v)
    elif self.r is not None:
      self.l = Tree(self.l, self.r, self.v)
    self.v = None
    self.r = new if isinstance(new, Tree) else Tree(v=new)
    return self

  def pop(self):
    if self.is_leaf(): return self #.v
    elif self.r is None: return self.l
    else: return self.r

def parse(fun_str):
  if len(fun_str) == 0: return None
  fun = Tree()
  funs = []
  word_start = None

  # fun_str\n -> fun_str
  if fun_str[-1] == "\n": fun_str = fun_str[:-1]

  fun_str = fun_str.strip()

  # (fun_str) -> fun_str
  if fun_str[0] == "(" and fun_str[-1] == ")": fun_str = fun_str[1:-1]

  for i in range(len(fun_str)):
    c = fun_str[i]

    if word_start is not None and (c == " " or c == ")" or c == "("):
      word = fun_str[word_start:i]
      word_start = None
      fun.push(word)

    if c == "(":
      funs.append(fun)
      fun = fun.push(Tree()).pop()

    elif c == ")":
      fun = funs.pop()

    elif word_start is None and c != " ":
      word_start = i

  if word_start is not None:
    word = fun_str[word_start:]
    fun.push(word)

  assert len(funs) == 0, \
      "Missing {} right parentheses in function {}".format(len(funs), fun_str)

  #assert fun.check(), "Could not parse \"{}\" as {}".format(fun_str, repr(fun))

  return fun

def parse_file(fp):
  with open(fp, "r") as f:
    return [parse(line) for line in f.readlines()]

#(a b c d) -> 
#
#a: a1, d
#
#a1: b, c

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
