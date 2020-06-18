import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import numpy as np
import random

### Global Constants ###

line_width = 0.15

range_theta = (math.pi, 2*math.pi)

# 0 => all layers have equal thickness
# 1 => each layer is 1/2 the thickness of the last
# float in range (0., 1.) => weighted average of methods (0) and (1)
radius_exp_weight = 0.75

# See in_ballpark(x, xm, xM)
ballpark = 2


# Ensure range_theta[0] and [1] are in range [0, 2pi]
range_theta = (range_theta[0] + (2*math.pi if range_theta[0] < 0 else 0),
               range_theta[1] + (2*math.pi if range_theta[1] < 0 else 0))

# Calculate plot dimensions (deprecated)
#plot_x1 = math.cos(range_theta[0])
#plot_x2 = math.cos(range_theta[1])
#plot_y1 = math.sin(range_theta[0])
#plot_y2 = math.sin(range_theta[1])
#plot_y_min = -1 if (range_theta[0] <= 3*math.pi/2 <= range_theta[1]) else min(plot_y1, plot_y2)
#plot_y_max =  1 if (range_theta[0] <= 1*math.pi/2 <= range_theta[1]) else max(plot_y1, plot_y2)
#plot_x_min = -1 if (range_theta[0] <= 2*math.pi/2 <= range_theta[1]) else min(plot_x1, plot_x2)
#plot_x_max =  1 if (range_theta[0] <= 4*math.pi/2 <= range_theta[1]) else max(plot_x1, plot_x2)  

def in_ballpark(x, xm, xM):
  "Is x in the ballpark of xm and xM?"
  return xM - (xM - xm)*ballpark <= x <= xm + (xM - xm)*ballpark

def set_globals(**kwargs):
  "Sets global params. Defaults: line_width=0.15, range_theta=(math.pi, 2*math.pi), radius_exp_weight=0.75, ballpark=2"
  global line_width, range_theta, radius_exp_weight, ballpark
  if "line_width" in kwargs: line_width = kwargs["line_width"]
  if "range_theta" in kwargs: range_theta = kwargs["range_theta"]
  if "radius_exp_weight" in kwargs: radius_exp_weight = kwargs["radius_exp_weight"]
  if "ballpark" in kwargs: ballpark = kwargs["ballpark"]
  return {"line_width":line_width,
          "range_theta":range_theta,
          "radius_exp_weight":radius_exp_weight,
          "ballpark":ballpark,
  }

class Tree:

  def __init__(self, v, l=None, r=None):
    self.v = v
    self.l = l
    self.r = r

  def map(self, f):
    l = self.l.map(f) if self.l else None
    r = self.r.map(f) if self.r else None
    return Tree(f(self.v), l=l, r=r)

  def min(self):
    v = np.min(self.v)
    l = self.l.min() if self.l else v
    r = self.r.min() if self.r else v
    return min(v, l, r)

  def max(self):
    v = np.max(self.v)
    l = self.l.max() if self.l else v
    r = self.r.max() if self.r else v
    return max(v, l, r)

  def depth(self):
    l = self.l.depth() if self.l else 0
    r = self.r.depth() if self.r else 0
    return 1 + max(l, r)


def mktree_uniform(depth):
    if depth == 1: return Tree(1/2)
    l = mktree_uniform(depth - 1).map(lambda x: x / 2)
    r = mktree_uniform(depth - 1).map(lambda x: 1 - (1 - x) / 2)
    return Tree(1/2, l=l, r=r)

def mktree_randomized(child_prob, decay=1.0, max_depth=15):
    if max_depth == 0: return Tree(random.random())
    l = mktree_randomized(child_prob*decay, decay, max_depth - 1).map(lambda x: x / 2) if random.random() < child_prob else None
    r = mktree_randomized(child_prob*decay, decay, max_depth - 1).map(lambda x: 1 - (1 - x) / 2) if random.random() < child_prob else None
    return Tree(l, r, random.random())


def get_radius(depth, max_depth):
    return (1 - 1/2 ** depth) * radius_exp_weight + depth / max_depth * (1 - radius_exp_weight)    

def draw_slice(depth, theta1, theta2, v, max_depth, has_left, has_right):
    acc = []
    w = len(v)
    r = get_radius(depth, max_depth)
    r0 = get_radius(depth - 1, max_depth)
    lw = line_width * (r - r0) # we want this to decrease, but 2^(d-1) was too much
    lw_r = lw / (2 * math.pi) # lw is relative to [0, 2pi], but r is [0, 1], so scale lw to [0, 1]
    lw_r *= 3 # lw_r looks too small, so we'll scale it by a constant factor
    if depth != max_depth: r -= lw_r / 2
    if depth != 1: r0 += lw_r / 2
    if has_left: theta1 += lw/2
    if has_right: theta2 -= lw/2
    for x in range(w):
        vx = float(v[x].item())
        acc.append([vx, # value
                    r - r0, # outer r
                    r0, # inner r
                    min(theta1, theta2) + abs(theta2 - theta1) * (x + 0.0) / w, # right(?) edge
                    abs(theta1 - theta2) / w, # width
                    depth,
        ])
    return acc

def draw_tree(acc, tree, depth, theta1, theta2, max_depth, has_left, has_right):
    if tree is not None:
        acc.extend(draw_slice(depth, theta1, theta2, tree.v, max_depth, has_left, has_right))
        theta3 = (theta1 + theta2)/2
        draw_tree(acc, tree.l, depth + 1, theta1, theta3, max_depth, has_left, tree.r)
        draw_tree(acc, tree.r, depth + 1, theta3, theta2, max_depth, tree.l, has_right)


def plot_tree(ax, tree, cm="cividis", median=None):

    def maybe_expand(x):
        a = np.asanyarray(x)
        return a if len(a.shape) else a[np.newaxis]

    tree = tree.map(maybe_expand)

    min_v, max_v, depth = tree.min(), tree.max(), tree.depth()
    extra_ticks = [min_v, max_v]
    if median is not None and in_ballpark(median, min_v, max_v):
      if min_v > median:
        min_v = median
      elif max_v < median:
        max_v = median
      else:
        mag = max(abs(min_v - median), abs(max_v - median))
        min_v, max_v = median - mag, median + mag

    # Normalize tree values to [0, 1]
    tree = tree.map(lambda xs: [(x - min_v) / (max_v - min_v) for x in xs])
  
    acc = []
    acc.append([0.5, 0, 1, 0, 2*math.pi, 0]) # makes sure 2pi = one revolution
    draw_tree(acc, tree, 1, range_theta[0], range_theta[1], depth, False, False)
    acc = sorted(acc, key=lambda x: x[-1]) # sort by depth

    df = pd.DataFrame(acc, columns=["v", "r", "r0", "theta", "dtheta", "depth"])

    cm = plt.cm.get_cmap(cm)
    plot = ax.bar(df["theta"], df["r"], width=df["dtheta"], bottom=df["r0"], color=cm(df["v"]), align="edge")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(min_v, max_v))
    sm.set_array(df["v"])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, orientation="horizontal", pad=0.0)
    cbar.ax.tick_params(labelsize="small")
    cbar.ax.ticklabel_format(style="sci", axis="x", scilimits=(-3,3))
    ax.set_thetamin(range_theta[0]/2/math.pi*360)
    ax.set_thetamax(range_theta[1]/2/math.pi*360)
    ax.set_thetagrids([])
    ax.set_rgrids([])
    ax.grid(False)
    ax.set_axis_off()

    return plot, cbar
