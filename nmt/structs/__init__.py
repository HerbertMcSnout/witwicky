from .struct import Struct
from .sequence import SequenceStruct
from .tree import Tree
from .tree2 import Tree as Tree2

"""
Each struct is a module with
- a subclass of Struct
- a 'parse' function (str -> Struct)
- a 'get_params' function (config -> [torch.Tensor])
"""
