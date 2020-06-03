class Struct:
  """
  Interface for the structure of a source language.
  Each subclass must implement
    flatten(self) - return a list of words contained in this Struct
    get_pos_embedding(self, embed_dim, params) - return a Struct with values that are torch.Tensors (vectors of length embed_dim)
    map(self, f) - return a Struct, after applying f to each value
  Above, params is the list of torch.Tensors given in nmt.configurations under the field 'struct_params'.
  Each Struct subclass must also have some function that parses a string and returns the subclass,
  to be referenced in nmt.configurations under the field 'struct_parser'.
  """

  def flatten(self):
    "Returns a list of words contained in this Struct"
    assert False, "Subclasses of Struct must implement flatten"

  def get_pos_embedding(self, embed_dim, params):
    "Returns a Struct with values that are torch.Tensors (vectors of length embed_dim)"
    assert False, "Subclasses of Struct must implement get_pos_embedding"

  def map(self, f):
    "Returns a Struct, after applying f to each value"
    assert False, "Subclasses of Struct must implement map"

  def size(self):
    "Returns the number of words this Struct flattens to"
    return len(self.flatten())
