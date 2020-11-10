import torch
import numpy
import matplotlib.pyplot as plt
import os
import colorsys
import sys
sys.path.append('../')

import nmt.structs as structs
import nmt.configurations as config


dpi = 500
lw = 0.8
adjust_right = 0.7 # make room for legend


lines = dict() # e.g. dict(en2vi='--')
  
colors = dict() # e.g. dict(en2vi='k')

def plot_dev_scores(scores):
  max_epoch = max(max(len(scores[model]['bleu']), len(scores[model]['perp'])) for model in scores)
  
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  
  for model, score in scores.items():
    perps = score['perp']
    epochs = list(range(1, len(perps) + 1))
    marker = '' if len(perps) > 1 else '.'
    ls = lines[model] if model in lines else None
    color = colors[model] if model in colors else None
    ax1.plot(epochs, perps, label=model, lw=lw, marker=marker, linestyle=ls, color=color)
  ax1.set(ylabel='Perplexity')
  plt.xlim(left=0, right=max_epoch)
  plt.ylim(top=250, bottom=0)
  
  ax2 = fig.add_subplot(212)
  for model, score in scores.items():
    bleus = score['bleu']
    epochs = list(range(1, len(bleus) + 1))
    marker = '' if len(bleus) > 0 else '.'
    ls = lines[model] if model in lines else None
    color = colors[model] if model in colors else None
    ax2.plot([0] + epochs, [0] + bleus, lw=lw, marker=marker, linestyle=ls, color=color)
  ax2.set(xlabel='Epoch', ylabel='BLEU')
  plt.xlim(left=0, right=max_epoch)
  
  fig.legend(loc=7)
  fig.tight_layout()
  fig.subplots_adjust(right=adjust_right)
  
  plt.savefig(save_to, bbox_inches='tight', dpi=dpi)
  plt.close('all')

def new_name(name, not_in=None):
  not_in = not_in or []
  i = len(name)
  while name[i-1:].isdigit(): i -= 1
  num = int(name[i:] or 2)
  while name[:i] + str(num) in not_in: num += 1
  return name[:i] + str(num)

  
if __name__ == '__main__':
  saved_models_dirs = ['../nmt/java2doc_models', '../nmt/py2doc_models', '../nmt/saved_models']
  save_tos = ['java2doc_dev_scores.png', 'py2doc_dev_scores.png', 'en2vi_dev_scores.png']
  
  for saved_models_dir, save_to in zip(saved_models_dirs, save_tos):
    scores = {}
    if isinstance(saved_models_dir, str):
      saved_models_dir = [os.path.join(saved_models_dir, x) for x in os.listdir(saved_models_dir)]
    for model_dir in saved_models_dir:
      bleu_fp = os.path.join(model_dir, 'bleu_scores.npy')
      perp_fp = os.path.join(model_dir, 'dev_perps.npy')
      model_name = os.path.basename(model_dir)
      if model_name in scores:
        model_name = new_name(model_name, not_in=scores)
      scores[model_name] = {'bleu':[], 'perp':[]}
      if os.path.exists(bleu_fp): scores[model_name]['bleu'] = list(numpy.load(bleu_fp))
      if os.path.exists(perp_fp): scores[model_name]['perp'] = list(numpy.load(perp_fp))
    if scores:
      plot_dev_scores(scores)
  
