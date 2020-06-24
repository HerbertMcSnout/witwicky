import sys
sys.path.append('../')
import nmt.structs as structs
import nmt
import nmt.configurations as config
import plot_tree
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from os.path import exists
from os import listdir

dpi = 400

sample2 = "(a (b (b3 (b4))) (c (d e f)) (g g1 g2) h (i (j k)))"
sample = "(unit (function (specifier public) (type (name expression)) (name get case expression) parameter_list (block (return return (expr (name case expression))))))"
sample3 = "(u (f (s p) (t (n i)) (n g d p h) p (b (d (d (t (n s)) (n p n) (i (e (c (n (n t) o (n g p p)) a) o (c (n g s) (a (a (e (n (n p c) o (n d s p n)))))))))) (r r (e (c (n (n r p) o (n g w)) a) o (c (n g r) a) o (c (n g p) (a (a (e (n p n))))))))))" # size: 82

saved_models_dir = "../nmt/saved_models"

cmap_pe = "PiYG"
cmap_norm = "cividis"
cmap_param = "PiYG" # "RdBu"

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
      params = [m[k] for k in ks]
      
      pe = cast_tree(struct.parse(sample).get_pos_embedding(cnfg["embed_dim"], params))
      
      w, h = matplotlib.figure.figaspect(0.5)
      fig = plt.figure(figsize=(w, h))
    
      ks_mtx = []
      height_ratios = []
      text_already = -1
      for k in ks:
        p = m[k]
        if p.dim() == 2:
          ks_mtx.append(p)
          height_ratios.append(4)
        elif (p.dim() == 1 and len(p) > 1):
          ks_mtx.append(p)
          height_ratios.append(1)
        elif text_already == -1:
          text_already = len(height_ratios)
          height_ratios.append(2)
        else:
          height_ratios[text_already] += 1

      height_ratios.append(sum(height_ratios)//3)
      
      gs = gridspec.GridSpec(nrows=len(height_ratios), ncols=3, figure=fig, height_ratios=height_ratios, width_ratios=[2,2,1])
      ax_pe = fig.add_subplot(gs[:, 0], projection="polar")
      ax_pe_norm = fig.add_subplot(gs[:, 1], projection="polar")
      
      plot_tree.plot_tree(ax_pe, pe, cm=cmap_pe, median=0)
      plot_tree.plot_tree(ax_pe_norm, pe.map(torch.norm), cm=cmap_norm, median=1)
      
      ax_pe.set_title("Position Embedding", pad=-60)
      ax_pe_norm.set_title("Frobenius Norm", pad=-60)

      cmin = min(p.min() for p in ks_mtx)
      cmax = max(p.max() for p in ks_mtx)
      cmin, cmax = plot_tree.get_value_range(0, min(cmin, -0.3), max(cmax, 0.3))
      norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
      ims = []
      gi = 0
      param_axs = []
      text_ax = None
      num_texts = 0
      for k in ks:
        p = m[k]
        if p.dim() == 0 or len(p) == 1:
          if text_ax is None:
            text_ax = fig.add_subplot(gs[gi, 2])
            text_ax.axis("off")
            gi += 1
          num_texts += 1
          text_ax.text(0.5, (num_texts + 1)/(height_ratios[text_already]),
                       "{} = {:.4f}".format(k, p.item()),
                       ha="center", va="center",
                       fontdict=fontdict)
        else:
          ax_params = fig.add_subplot(gs[gi, 2])
          gi += 1
          fontdict = dict(fontsize=6)
          param_axs.append(ax_params)
          ax_params.set_title(k, fontdict=fontdict)
          ax_params.axis("off")
          if p.dim() == 2:
            ims.append(ax_params.imshow(p.numpy(), norm=norm, cmap=cmap_param))
          elif p.dim() == 1:
            ims.append(ax_params.imshow(p.unsqueeze(0).expand(len(p)//4, len(p)).numpy(), norm=norm, cmap=cmap_param))
          else:
            print("Can't display parameters of >= 3 dims, in model {}".format(model))
          ax_params.set_xticks([])
          ax_params.set_yticks([])
      
      for im in ims: im.set_norm(norm)
      
      ax_cb = fig.add_subplot(gs[:, 2])
      ax_cb.axis("off")
      cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_param), ax=ax_cb, orientation="horizontal", aspect=12.5)
      cbar.ax.tick_params(labelsize="small")
      cbar.ax.ticklabel_format(style="sci", axis="x", scilimits=(-3,3))
      
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

