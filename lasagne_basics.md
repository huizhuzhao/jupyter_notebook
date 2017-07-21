

```python
from lasagne.layers import InputLayer, DenseLayer, MergeLayer
```

## DenseLayer
layer.input_shape, layer.output_shape


```python
l_in = InputLayer(shape=(100, 20, 20, 3))
l_hidden = DenseLayer(l_in, num_units=50, num_leading_axes=2)
print l_in.output_shape
print l_hidden.output_shape, l_hidden.input_shape
```

    (100, 20, 20, 3)
    (100, 20, 50) (100, 20, 20, 3)


## MergeLayer
**layer.input_shapes**, layer.output_shape


```python
max_atom_len = 100
max_bond_len = 120
atom_feat_dim = 50
bond_feat_dim = 60
l_in_atom = InputLayer(shape=(max_atom_len, atom_feat_dim))
l_in_bond = InputLayer(shape=(max_bond_len, bond_feat_dim))
print l_in_atom.output_shape, l_in_bond.output_shape
```

    (100, 50) (120, 60)



```python
merge_layer = MergeLayer([l_in_atom, l_in_bond])
print merge_layer.input_shapes
print merge_layer.input_layers
```

    [(100, 50), (120, 60)]
    [<lasagne.layers.input.InputLayer object at 0x7f3ccc989690>, <lasagne.layers.input.InputLayer object at 0x7f3ccc9896d0>]



```python
from lasagne.layers import MergeLayer
class HiddenLayer(MergeLayer):
    def __init__(incoming):
        pass
```


```python

```
