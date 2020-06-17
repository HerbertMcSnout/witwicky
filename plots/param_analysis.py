import sys
sys.path.append('../')
import nmt.structs as structs
import nmt
import nmt.configurations as config
import plot_tree
import torch
import matplotlib.pyplot as plt
from os.path import exists
from os import listdir

dpi = 1000

sample = "(a (b (b3 (b4))) (c (d e f)) (g g1 g2) h (i (j k)))"

saved_models_dir = "../nmt/saved_models"

def cast_tree(t):
  return t.fold_up(plot_tree.Tree)

for model in listdir(saved_models_dir):
  fp = "{0}/{1}/{1}.pth".format(saved_models_dir, model)

  if exists(fp) and hasattr(config, model):
    cnfg = config.get_config(model, getattr(config, model))
    struct = cnfg["struct"]
    tree = struct.parse(sample)
    if isinstance(tree, structs.tree_utils.Tree):
      ks = struct.get_params(cnfg).keys()
      m = torch.load(fp, map_location=torch.device("cpu"))
      params = [m[k] for k in ks] # get param names
      
      pe = cast_tree(struct.parse(sample).get_pos_embedding(cnfg["embed_dim"], params))
      
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1, projection="polar")
      ax2 = fig.add_subplot(1,2,2, projection="polar")
      
      plot_tree.plot_tree(ax1, pe, cm="PiYG", median=0)
      plot_tree.plot_tree(ax2, pe.map(torch.norm), cm="cividis", median=1)
      
      ax1.set_title("Position Embedding")
      ax2.set_title("Frobenius Norm")
      
      plt.tight_layout()
      plt.savefig("pe-{}.png".format(model), dpi=dpi)
      plt.close("all")

  else:
    print("{} doesn't exist".format(model))

#def plot(title, *t, height=None):
#    if len(t) != 1: t = torch.stack(list(t))
#    elif len(t[0].size()) == 1: t = torch.stack(list(t)*(t[0].size()[0]//3))
#    else: t = t[0]
#    fig = plt.imshow(t, title=title)
#    if height is not None: fig.update_layout(height=height)
#    fig.show()

