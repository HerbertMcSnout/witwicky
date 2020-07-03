from .struct import Struct
from .tree_utils import parse
from .sequence import SequenceStruct
from .tree import Tree
from .tree2 import Tree as Tree2
from .tree14 import Tree as Tree14
from .tree15 import Tree as Tree15
from .tree18 import Tree as Tree18
from .tree19 import Tree as Tree19
from .tree20 import Tree as Tree20
from .tree21 import Tree as Tree21
from .tree22 import Tree as Tree22
from .tree142 import Tree as Tree142
from .tree143 import Tree as Tree143
from .tree144 import Tree as Tree144
from .tree14_log import Tree as TreeLog
from .tree14_sum import Tree as TreeSum
from .tree14_3d import Tree as Tree3d
# There's probably some easier way to include relative files in a module...

"""
Each struct is a module with
- a subclass of Struct
- a 'parse' function (str -> Struct)
- a 'get_params' function (config -> [torch.Tensor(embed_dim)])
- (optional) a get_reg_penalty function (torch.Tensor(batch_size, max_len) -> torch.Tensor(batch_size, max_len))
    This should map each node's position embedding Frobenius norm to a regularization penalty number.
"""
