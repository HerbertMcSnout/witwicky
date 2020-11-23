import sys
import os
os.chdir('../')
sys.path.append('./')
import nmt.structs as structs
import nmt.configurations as config
import nmt.train # sets torch's random seed
import nmt.all_constants as ac
import nmt.utils as ut
import nmt.structs.tree_utils as tu
import plot_tree
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import random
import torch.nn.functional as F

random.seed(ac.SEED)
dpi = 400

fontsize = 110
nrows = 2

device = ut.get_device()

just_strengths = True
plot_strengths = False
reverse_cmap = True

relationship_reordering = [0, 3, 4, 5, 6, 7, 8, 1, 2]

name_suffix = '_mask_strengths' if just_strengths else '_heads'

chosen_cmap = plt.get_cmap('Greens')
if reverse_cmap:
  chosen_cmap = chosen_cmap.reversed()

en_sample = '(S (NP (PRP He ) ) (VP (VBZ is ) (NP (PRP\$ my ) (NN father ) ) ) (. . ) )'
# German for 'He is my father.' -> 'Er ist mein Vater.' (according to Google Translate :P)
de_sample = '(VROOT (S (PPER-SB Er) (VAFIN-HD ist) (NP-PD (PPOSAT-NK mein) (NN-NK Vater))) ($. .))'

models = [
  (en_sample, 'en2vi_att_sin', 'en2vi_att_ast'),
  (en_sample, 'en2ur_att_sin', 'en2ur_att_ast'),
  (en_sample, 'en2tu_att_sin', 'en2tu_att_ast'),
  (en_sample, 'en2ha_att_sin', 'en2ha_att_ast'),
  (en_sample, 'en2de_att_sin_100k', 'en2de_att_ast'),
  (de_sample, 'de2en_att_sin_100k', 'de2en_att_ast'),

  (en_sample, 'en2vi17a2', 'en2vi_tree'),
  (en_sample, 'en2ur17a2', 'en2ur_tree'),
  (en_sample, 'en2tu17a2', 'en2tu_tree'),
  (en_sample, 'en2ha17a2', 'en2ha_tree'),
  (en_sample, 'en2de17a2_100k', 'en2de_tree'),
  (de_sample, 'de2en17a2_100k', 'de2en_tree'),
]

def plot_mask_strength(ax, params, norm):
  #ax.set_title('Mask Strength')

  im_mtx = params['self_attn_weights'][:-1][relationship_reordering].detach()
  ax_im = ax.imshow(im_mtx, cmap=chosen_cmap, norm=norm)
  xticklabels = [i for i in range(im_mtx.size()[1])]
  ax.set_xticks(xticklabels)
  ax.set_xticklabels([i + 1 for i in xticklabels])
  ax.set_xlabel("Attention Head")
  yticklabels = tu.HEAD_NAMES[1:-1]
  ax.set_yticks(range(len(yticklabels)))
  ax.set_yticklabels([yticklabels[relationship_reordering[i]] for i in range(len(yticklabels))])
  divider = make_axes_locatable(ax)

  cax = divider.append_axes("right", size="6.25%", pad=0.1)
  plt.colorbar(ax_im, cax=cax)



for sample, model_name, save_name in models:
  save_name += name_suffix
  print(save_name + '...', end='')
  cnfg = config.get_config(model_name, getattr(config, model_name))
  
  struct = cnfg['struct']
  
  model_fp = os.path.join(cnfg['save_to'], model_name + '.pth')
  
  if True:
    model = nmt.model.Model(cnfg, load_from=model_fp).to(device)
    tree_words = model.data_manager.parse_line(sample, True, to_ids=False)
    words = tree_words.flatten()
    tree = model.data_manager.parse_line(sample, True, to_ids=True)
    toks = torch.tensor(tree.flatten(), device=device).unsqueeze(0)
  
    self_att_layers = model.encoder.self_atts
    x = F.dropout(toks, p=model.encoder.dropout, training=False)
    params = model.struct_params
  else:
    if os.path.exists(model_fp):
      params = struct.get_params(cnfg) if hasattr(struct, 'get_params') else {}
      m = torch.load(model_fp, map_location=device)['model']
      params = {k:m[k] for k in params}
    else:
      params = struct.get_params(cnfg)
  
  
  num_heads = cnfg['num_enc_heads']
  att_mask, _ = struct.get_enc_mask(toks, [tree], num_heads, **params)
  att_mask = att_mask.squeeze(0).detach().cpu()
  
  att_min = att_mask.min().detach().cpu().item()
  att_max = att_mask.max().detach().cpu().item()
  orig_range = att_min, att_max
  #if att_max > 0 and att_min < 0:
  #  att_max, att_min = max(att_max, -att_min), min(-att_max, att_min)
  norm = matplotlib.colors.Normalize(vmin=att_min, vmax=att_max)

  if just_strengths:
    fig = plt.figure(figsize=(4, 3))
    plot_mask_strength(plt.gca(), params, norm)
  else:
    ncols = (num_heads + plot_strengths + nrows - 1)//nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols + 1, 3*nrows + 1))
  
    #fig.suptitle(title, size='xx-large', weight='bold', va='top')
    
    try: axs = sum([[x for x in row] for row in axs], [])
    except: pass

    if plot_strengths:
      ax = axs[0]
      plot_mask_strength(ax, params, norm)

    for i, ax in enumerate(axs[int(plot_strengths):]):
      ax.set_title(f'Attention Head {i+1}')
      ax_im = ax.imshow(att_mask[i], cmap=chosen_cmap, norm=norm)
      ax.set_xticks(range(len(words)))
      ax.set_xticklabels(words, fontsize=fontsize//len(words))
      ax.set_yticks(range(len(words)))
      ax.set_yticklabels(words, fontsize=fontsize//len(words))
      plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
  plt.tight_layout(pad=1.25, w_pad=0.6, h_pad=0.6)
  plt.savefig(f'plots/png/{save_name}.png', dpi=dpi, transparent=False, pad_inches=0.0,)
  plt.close('all')
  print(f' done ({orig_range[0]:0.4f}, {orig_range[1]:0.4f})')
  
