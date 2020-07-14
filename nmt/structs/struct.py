class Struct:
  '''
  Interface for the structure of a source language.
  Each subclass must implement
    _flatten(self) - return a list of words contained in this Struct
    get_pos_embedding(self, embed_dim, params) - return a Struct with values that are torch.Tensors (vectors of length embed_dim)
    map(self, f) - return a Struct, after applying f to each value
    __str__ - must be isomorphic with parser
  Above, params is the list of torch.Tensors given in nmt.configurations under the field 'struct_params'.
  Each Struct subclass must also have some function that parses a string and returns the subclass,
  to be referenced in nmt.configurations under the field 'struct_parser'.
  '''

  def __str__(self):
    'Format as string. Must be isomorphic with the parser.'
    assert False, 'Subclasses of Struct must implement __str__'

  def _flatten(self):
    'Returns a list of words contained in this Struct'
    assert False, 'Subclasses of Struct must implement flatten'
    
  def flatten(self):
    xs = self._flatten()
    max_len = self.get_clip_length()
    return xs[:max_len] if max_len and len(xs) > max_len else xs

  def get_pos_embedding(self, embed_dim, params):
    'Returns a Struct with values that are torch.Tensors (vectors of length embed_dim)'
    assert False, 'Subclasses of Struct must implement get_pos_embedding'

  def map(self, f):
    'Returns a Struct, after applying f to each value'
    assert False, 'Subclasses of Struct must implement map'

  def size(self):
    'Returns the number of words this Struct flattens to'
    return len(self.flatten())

  def set_clip_length(self, max_len):
    'Sets the length to clip this struct to, in calls to self.flatten()'
    self.clip_length = max_len

  def get_clip_length(self):
    'Returns this structs\'s clip length'
    return self.clip_length if hasattr(self, 'clip_length') else None

  def forget(self):
    'Sets all non-null node values to None'
    return self.map(lambda x: None)
  
  def maybe_add_eos(self, EOS_ID):
    '(Optional) Override if this struct needs to get and EOS token'
    pass
