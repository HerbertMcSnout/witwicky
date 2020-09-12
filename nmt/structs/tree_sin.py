import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

class Tree(tree_utils.Tree):

  def get_pe_h(self, mu_l, mu_r, lam_r, pe_sin, i):
    if self.l:
      l, i = self.l.get_pe_h(mu_l, mu_r, lam_r, pe_sin, i)
      r, i = self.r.get_pe_h(mu_l, mu_r, lam_r, pe_sin, i) if self.r else (None, i)
      rv = r.v if r else lam_r
      v = (mu_l @ l.v) * (mu_r @ rv)
      return self.new(v, l, r), i
    else:
      r, j = self.r.get_pe_h(mu_l, mu_r, lam_r, pe_sin, i + 1) if self.r else (None, i + 1)
      return self.new(pe_sin[i, :], l=None, r=r), j

  def get_pos_embedding(self, embed_dim, mu_l, mu_r, lam_r, c_l, c_r):
    pe_sin = ut.get_position_encoding(embed_dim, self.size()) # technically, self.size() is more than we actually need
    return self.get_pe_h(c_l * mu_l, c_r * mu_r, lam_r, pe_sin, 0)[0]
    

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)

def get_params(config):
  embed_dim = config['embed_dim']
  device = ut.get_device()
  return dict(
    mu_l = torch.nn.init.orthogonal_(torch.empty(embed_dim, embed_dim, device=device)),
    mu_r = torch.nn.init.orthogonal_(torch.empty(embed_dim, embed_dim, device=device)),
    lam_r = torch.nn.init.normal_(torch.empty(embed_dim, device=device), mean=0, std=2**-0.5),
    c_l = torch.tensor(2**0.25, device=device),
    c_r = torch.tensor(2**0.25, device=device),
  )

