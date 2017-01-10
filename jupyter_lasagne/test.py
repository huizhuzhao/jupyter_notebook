#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年01月10日 星期二 13时57分10秒
# last modified: 

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import Layer
from lasagne.layers import InputLayer
from lasagne import init
from lasagne.nonlinearities import rectify
from lasagne.layers.merge import autocrop, autocrop_array_shapes

class DenseLayer(Layer):
    def __init__(self, incoming, num_units, W=init.GlorotUniform(), b=init.Constant(0.),
            nonlinearity=rectify, num_leading_axes=1, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units

        self.num_leading_axes = num_leading_axes
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.b = self.add_param(b, (num_units, ), name='b', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units, )

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            input = input.flatten(num_leading_axes + 1)

class MergeLayer(Layer):
    def __init__(self, incomings, name=None):
        self.input_shapes = [incoming.output_shape for incoming in incomings]
        self.input_layers = [incoming for incoming in incomings]
        self.name = name
        self.get_output_kwargs = []


class ConcatLayer(MergeLayer):
    def __init__(self, incomings, axis=1, cropping=None, **kwargs):
        super(ConcatLayer, self).__init__(incomings, **kwargs)
        self.axis = axis
        if cropping is not None:
            cropping = list(cropping)
            cropping[axis] = None
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        return T.concatenate(inputs, axis=self.axis)


class ElemwiseMergeLayer(MergeLayer):
    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


def main():
    l_input = InputLayer((1, 2, 3, 4))
    l_hidden = DenseLayer(l_input, 100)
    #print l_hidden.get_output_shape_for((1, 2, 3))
    #print l_hidden.output_shape
    a = (None, 2, 3, 4)
    b = (5, 4, 3, 4)
    c = (7, 1, 8, 9)
    l_concat = ConcatLayer([l_input, l_hidden], cropping=[None]*4)
    l_merge = ElemwiseMergeLayer([l_input, l_hidden], 'a')
    print l_merge.get_output_shape_for([a, a])


def foo():

    x = T.ftensor3('x')
    l_hidden.get_output_for(x)

    a = np.random.random((1, 2, 3, 4))
    b = np.random.random((5, 4, 4, 2))
    c = np.random.random((7, 1, 8, 9))
    cropping = [None, 'lower', 'center', 'upper']

    xa, xb, xc = autocrop([theano.shared(a), theano.shared(b), theano.shared(c)], cropping)
    xa, xb, xc = xa.eval(), xb.eval(), xc.eval()
    print np.all(xa == a[:, :1, :3, -2:])
    print np.all(xb == b[:, :1, :3, -2:])
    print np.all(xc == c[:, :1, 2:5, -2:])

    inputs = [theano.shared(a), theano.shared(b), theano.shared(c)]
    shapes = [input.shape for input in inputs]
    shapes_tensor = T.as_tensor_variable(shapes)
    min_shape = T.min(shapes_tensor, axis=0)
    print shapes
    print shapes_tensor

    fn = theano.function([], shapes)
    print fn()
    fn = theano.function([], shapes_tensor)
    print fn()
    fn = theano.function([], min_shape)
    print fn()

    a = (1, 2, 3, 4)
    b = (5, 4, 4, 2)
    c = (7, 1, 8, 9)
    cropped_shapes = autocrop_array_shapes([a, b, c], cropping)
    print cropped_shapes





if __name__ == '__main__':
    main()
