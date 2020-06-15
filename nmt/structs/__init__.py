from .struct import Struct
from .sequence import SequenceStruct
from .tree import Tree
from .tree2 import Tree as Tree2
from .tree3 import Tree as Tree3
from .tree4 import Tree as Tree4
from .tree5 import Tree as Tree5
from .tree6 import Tree as Tree6
from .tree7 import Tree as Tree7
from .tree8 import Tree as Tree8

"""
Each struct is a module with
- a subclass of Struct
- a 'parse' function (str -> Struct)
- a 'get_params' function (config -> [torch.Tensor])
"""
