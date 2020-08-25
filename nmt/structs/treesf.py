import torch
from nmt.utils import get_position_encoding, get_float_type, get_device
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils

class SequenceTree(Struct):

  def __init__(self, data):
    self.data = data

  def __str__(self):
    return ' '.join([str(x) for x in self.data])

  def flatten(self):
    return self.data

  def map(self, f):
    return SequenceTree([f(x) for x in self.data])

  def set_clip_length(self, clip):
    self.data = self.data[:clip]

  def get_pos_embedding(self, embed_dim, mu, lam, rho, eps):
    size = self.size()
    device = get_device()

    insides = torch.empty(size, embed_dim, device=device)
    x = lam
    for i in range(size):
      insides[i, :] = x
      x = mu @ x

    #outsides = torch.empty(size, embed_dim, device=device)
    #x = rho
    #for i in range(size):
    #  outsides[i, :] = x
    #  r = insides[i + 1, :] if i + 1 < size else eps
    #  x = torch.einsum("i,ij,i->j", x, mu, r)
    
    return SequenceTree(insides)


def parse(s, clip=None):
  return SequenceTree(s.strip().split(maxsplit=(clip or -1))[slice(clip)])

def normalize(x):
  return (x - x.mean()) / x.std()

def get_params(config):
  embed_dim = config['embed_dim']
  device = get_device()
  return dict(
    mu  = tree_utils.init_tensor(embed_dim, embed_dim),
    lam = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)),
    rho = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)),
    eps = normalize(torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0., std=1.)),
  )
