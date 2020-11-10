import sys
sys.path.append('../')
import nmt.structs as structs
import nmt
import nmt.configurations as config
import nmt.train # sets torch's random seed
import nmt.all_constants as ac
import plot_tree
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from os.path import exists
from os import listdir
import random

random.seed(ac.SEED)
dpi = 400

java_sample = '(function (type (specifier public) (name int)) (name size) parameter_list (block_content (return (call (name (name parameters) (operator .) (name size)) argument_list))))'
en2vi_sample = '(S (S (NP (PRP$ My ) (NN family ) ) (VP (VBD was ) (RB not ) (ADJP (JJ poor ) ) ) ) (, , ) (CC and ) (S (NP (PRP myself ) ) (, , ) (NP (PRP I ) ) (VP (VBD had ) (ADVP (RB never ) ) (VP (VBN experienced ) (NP (NN hu@@ nger ) ) ) ) ) (. . ) )'
py_sample = "(FUNCTIONDEF (NAME url slash cleaner) (ARGUMENTS (ARG (NAME url))) (BODY (RETURN (CALL (ATTRIBUTE (NAME re) (ATTR sub)) (ARGS STR STR (NAME url))))))"
tree2rel_sample = '(0 (0 0 (0 0)) (0 (0 0)) (0 0) (0 (0 (1 (0 (0 2)) (0 0)))))'

#sample, saved_models_dir = py_sample, '../nmt/py2doc_models'
sample, saved_models_dir = java_sample, '../nmt/java2doc_models'
#sample, saved_models_dir = en2vi_sample, '../nmt/saved_models'

models = listdir(saved_models_dir)
#models = ['py2doc17_small', 'py2doc17a_small', 'py2doc17a2_small', 'py2doc17a', 'py2doc17a2']
models = ['java2doc17_small', 'java2doc17a2', 'java2doc17a', 'java2doc17a2_small', 'java2doc17a_small']
#models = ['en2vi17_small', 'en2vi17a2', 'en2vi17a']

diverging_cmap = 'Spectral'
sequential_cmap = 'viridis'

def cast_tree(t):
  return t.fold_up(plot_tree.Tree)

for model in models:
  fp = '{0}/{1}/{1}.pth'.format(saved_models_dir, model)

  if hasattr(config, model):
    cnfg = config.get_config(model, getattr(config, model))
    params = None
    struct = cnfg['struct']
    tree = struct.parse(sample)
    if structs.tree_utils.Tree in tree.__class__.__bases__:
        if exists(fp):
          params = struct.get_params(cnfg) if hasattr(struct, "get_params") else {}
          m = torch.load(fp, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')['model']
          params = {k:m[k] for k in params}
        else:
          params = struct.get_params(cnfg)

        pe = tree.get_pos_embedding(cnfg['embed_dim'], **params)
        pe = cast_tree(pe)
      
        w, h = matplotlib.figure.figaspect(0.5)
        fig = plt.figure(figsize=(w, h))
        
        ks_mtx = []
        height_ratios = []
        text_already = -1
        for k, p in params.items():
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
  
        ncols = 3 #4
        width_ratios = [(2 if i != ncols - 1 else 1) for i in range(ncols)]
        
        gs = gridspec.GridSpec(nrows=len(height_ratios), ncols=ncols, figure=fig, height_ratios=height_ratios, width_ratios=width_ratios)
        ax_pe = fig.add_subplot(gs[:, 0], projection='polar')
        ax_pe_norm = fig.add_subplot(gs[:, 1], projection='polar')
        #ax_pe_sum = fig.add_subplot(gs[:, 2], projection='polar')
        
        plot_tree.plot_tree(ax_pe, pe.map(lambda x: x.detach().cpu()), cm=sequential_cmap, median=0, diverging_cm=diverging_cmap)
        plot_tree.plot_tree(ax_pe_norm, pe.map(lambda x: torch.norm(x).detach().cpu()), cm=sequential_cmap, median=1, diverging_cm=diverging_cmap)
        #plot_tree.plot_tree(ax_pe_sum, pe.map(lambda x: torch.sum(x).detach().cpu()), cm=cmap_sum, median=0)
        
        pad = 0-40 # -60
        ax_pe.set_title('Position Embedding', pad=pad)
        ax_pe_norm.set_title('Frobenius Norm', pad=pad)
        #ax_pe_sum.set_title('Sum', pad=pad)
    
        cmin = min(p.min() for p in ks_mtx)
        cmax = max(p.max() for p in ks_mtx)
        cmin, cmax = plot_tree.get_value_range(0, min(cmin, -0.3), max(cmax, 0.3))
        cmap_param = sequential_cmap if cmin + cmax else diverging_cmap
        norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
        ims = []
        gi = 0
        param_axs = []
        text_ax = None
        num_texts = 0
        for k, p in params.items():
          if p.dim() == 0 or len(p) == 1:
            if text_ax is None:
              text_ax = fig.add_subplot(gs[gi, -1])
              text_ax.axis('off')
              gi += 1
            num_texts += 1
            text_ax.text(0.5, (num_texts + 1)/(height_ratios[text_already]),
                         '{} = {:.4f}'.format(k, p.item()),
                         ha='center', va='center',
                         fontdict=fontdict)
          else:
            ax_params = fig.add_subplot(gs[gi, -1])
            gi += 1
            fontdict = dict(fontsize=6)
            param_axs.append(ax_params)
            ax_params.set_title(k, fontdict=fontdict)
            ax_params.axis('off')
            if p.dim() == 2:
              ims.append(ax_params.imshow(p.detach().cpu().numpy(), norm=norm, cmap=cmap_param))
            elif p.dim() == 1:
              ims.append(ax_params.imshow(p.unsqueeze(0).expand(len(p)//4, len(p)).detach().cpu().numpy(), norm=norm, cmap=cmap_param))
            else:
              print('Can\'t display parameters of >= 3 dims, in model {}'.format(model))
            ax_params.set_xticks([])
            ax_params.set_yticks([])
        
        for im in ims: im.set_norm(norm)
        
        ax_cb = fig.add_subplot(gs[:, -1])
        ax_cb.axis('off')
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_param), ax=ax_cb, orientation='horizontal', aspect=12.5)
        cbar.ax.tick_params(labelsize='small')
        cbar.ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
        
        plt.tight_layout()
        plt.savefig('png/pe-{}.png'.format(model), dpi=dpi)
        plt.close('all')
#      except:
#        print("Couldn't plot " + model)
  
  else:
    print('{} doesn\'t have a config'.format(model))

