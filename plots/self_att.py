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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import os
import random
import torch.nn.functional as F

random.seed(ac.SEED)
dpi = 400

device = ut.get_device()

chosen_cmap = 'bwr'#'viridis'

py2doc_sample = '(FUNCTIONDEF (NAME url slash cleaner) (ARGUMENTS (ARG (NAME url))) (BODY (RETURN (CALL (ATTRIBUTE (NAME re) (ATTR sub)) (ARGS STR STR (NAME url))))))'
java2doc_sample = '(function (type (specifier public) (specifier static) (name void)) (name copy) (parameter_list (parameter (decl (type (name (name byte) index)) (name in))) (parameter (decl (type (name output stream)) (name out)))) (throws (name ioexception)) (block_content (call (name (name assert) (operator .) (name not null)) (argument_list (name in) (name string))) (call (name (name assert) (operator .) (name not null)) (argument_list (name out) (name string))) (call (name (name out) (operator .) (name write)) (argument_list (name in)))))' #'(function (type (specifier public) (name int)) (name size) parameter_list (block_content (return (call (name (name parameters) (operator .) (name size)) argument_list))))'
en2vi_sample = '(S (S (NP (PRP$ My ) (NN family ) ) (VP (VBD was ) (RB not ) (ADJP (JJ poor ) ) ) ) (, , ) (CC and ) (S (NP (PRP myself ) ) (, , ) (NP (PRP I ) ) (VP (VBD had ) (ADVP (RB never ) ) (VP (VBN experienced ) (NP (NN hu@@ nger ) ) ) ) ) (. . ) )'

suffix = '_att_sin'
py2doc_model = 'py2doc' + suffix
java2doc_model = 'java2doc' + suffix
en2vi_model = 'en2vi' + suffix
en2tu_model = 'en2tu' + suffix
en2ha_model = 'en2ha' + suffix

nrows = 3

models = [
  (en2vi_sample, en2vi_model),
  (en2vi_sample, en2tu_model),
  (en2vi_sample, en2ha_model),
  (java2doc_sample, java2doc_model),
  (py2doc_sample, py2doc_model)
]

for sample, model_name in models:
  print(model_name + '...', end='')
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
  if att_max > 0 and att_min < 0:
    att_max, att_min = max(att_max, -att_min), min(-att_max, att_min)
  norm = matplotlib.colors.Normalize(vmin=att_min, vmax=att_max)
  
  ncols = num_heads//nrows + 1
  fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols + 1, 3*nrows + 1))
  
  ax = axs[0][0]
  im_mtx = params['self_attn_weights'][:-1].detach()
  ax_im = ax.imshow(im_mtx, cmap=chosen_cmap, norm=norm)
  xticklabels = [i for i in range(im_mtx.size()[1])]
  ax.set_xticks(xticklabels)
  ax.set_xticklabels([i + 1 for i in xticklabels])
  ax.set_xlabel("Attention Head")
  #ax.set_ylabel("Node Relationship")
  yticklabels = tu.HEAD_NAMES[1:-1]
  ax.set_yticks(range(len(yticklabels)))
  ax.set_yticklabels(yticklabels)
  plt.colorbar(ax_im, ax=ax)
  axs = sum([[x for x in row] for row in axs], [])
  
  for i, ax in enumerate(axs[1:]):
    ax_im = ax.imshow(att_mask[i], cmap=chosen_cmap, norm=norm)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=140//len(words))
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=140//len(words))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
  
  plt.tight_layout()
  plt.savefig(f'plots/png/self-att-{model_name}.png', dpi=dpi)
  plt.close('all')
  print(' done')
  
