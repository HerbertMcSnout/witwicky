import sys
sys.path.append('../')

import nmt.structs as structs
import nmt.configurations as config
import torch
import matplotlib.pyplot as plt
from os.path import exists
from os import listdir
import numpy as np

dpi = 500
lw = 0.8
adjust_right = 0.63 # make room for legend

saved_models_dir = "../nmt/saved_models"
save_to = "dev_scores.png"

scores = {}

for model in listdir(saved_models_dir):
  bleu_fp = "{}/{}/bleu_scores.npy".format(saved_models_dir, model)
  perp_fp = "{}/{}/dev_perps.npy".format(saved_models_dir, model)
  scores[model] = {"bleu":[], "perp":[]}
  if exists(bleu_fp): scores[model]["bleu"] = list(np.load(bleu_fp))
  if exists(perp_fp): scores[model]["perp"] = list(np.load(perp_fp))

#scores["fun2com11"]["perp"][1] = 400

max_epoch = max(max(len(scores[model]["bleu"]), len(scores[model]["perp"])) for model in scores)

fig = plt.figure()
ax1 = fig.add_subplot(211)

for model, score in scores.items():
  perps = score["perp"]
  epochs = list(range(1, len(perps) + 1))
  marker = "" if len(perps) > 1 else "."
  ax1.plot(epochs, perps, label=model, lw=lw, marker=marker)
ax1.set(ylabel="Perplexity")
plt.xlim(left=0, right=max_epoch)
plt.ylim(top=250, bottom=0)

ax2 = fig.add_subplot(212)
for model, score in scores.items():
  bleus = score["bleu"]
  epochs = list(range(1, len(bleus) + 1))
  marker = "" if len(bleus) > 0 else "."
  ax2.plot([0] + epochs, [0] + bleus, lw=lw, marker=marker)
ax2.set(xlabel="Epoch", ylabel="BLEU")
plt.xlim(left=0, right=max_epoch)

fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=adjust_right)

plt.savefig(save_to, bbox_inches="tight", dpi=dpi)
plt.close("all")
