#!/usr/bin/env python3
"""
Run TF Lite model using numpy rather than the TensorFlow TF Lite implementation

The goal: code a version so I know for fact that I can load and parse flatbuffer
and then use the weights, biases, etc. to generate the desired bounding boxes
and class information on input images.

Next step: implement this on the GPU.

How to use this "reference implementation" (a.k.a. a hacky script):
 - Running this will output tflite_manual.npy
 - Then run tflite_numpy_visualize.py to check the results on the last image in
   test_images (index == -1 at the moment)
"""
import os
import sys
import copy
import flatbuffers
import numpy as np
from PIL import Image
from enum import Enum

import tflite
import tflite.TensorType
import tflite.BuiltinOperator
import tflite.BuiltinOptions
import tflite.Padding
import tflite.ActivationFunctionType
import tflite.Conv2DOptions
import tflite.DepthwiseConv2DOptions
import tflite.ConcatenationOptions
import tflite.ReshapeOptions
import tflite.SubGraph
import tflite.Model
import tflite.QuantizationParameters
from image import find_files, load_image_into_numpy_array

import tensorflow as tf
tf.enable_eager_execution()

Padding = Enum("Padding", "VALID SAME")

class Concat:
    def __repr__(self):
        return "Concat"

    def __call__(self, input_tensors, input_buffers, options):
        return np.concatenate(input_buffers, options["axis"]).astype(options["out_type"])

class Reshape:
    def __repr__(self):
        return "Reshape"

    def __call__(self, input_tensors, input_buffers, options):
        assert len(input_buffers) == 2, \
            "Reshape takes tensor and shape as input"
        assert all(input_buffers[1] == options["shape"]), \
            "input_buffers[1] != options[\"shape\"]"
        return np.reshape(input_buffers[0], options["shape"]).astype(options["out_type"])

class Logistic:
    def __repr__(self):
        return "Logistic"

    def __call__(self, input_tensors, input_buffers, options):
        assert len(input_buffers) == 1, "Logistic assumes single input"
        return (1 / (1 + np.exp(-input_buffers[0]))).astype(options["out_type"])

class TFLite_Detection_PostProcess:
    """ Refer to:
    https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/detection_postprocess.cc
    """
    def __repr__(self):
        return "TFLite_Detection_PostProcess"

    def __call__(self, input_tensors, input_buffers, options):
        raise NotImplementedError("TFLite_Detection_PostProcess")

def zero_pad(x, pad_before_h, pad_after_h, pad_before_w, pad_after_w):
    """
    Input: x (batch_size, n_H, n_W, n_C), padding amount before/after
    """
    # Not an integer, must have different padding on either side
    #if int(pad) != pad:
    #    pad1 = int(np.floor(pad))
    #    pad2 = int(np.ceil(pad))
    #
    #    #return np.pad(x, ((0,0), (pad1,pad2), (pad1,pad2), (0,0)), 'constant')
    #    return np.pad(x, ((0,0), (0,pad1+pad2), (0,pad1+pad2), (0,0)), 'constant')
    # Integer, meaning we can have same padding on both sides
    #else:
    #    pad = int(pad)
    #    #return np.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant')
    #    return np.pad(x, ((0,0), (0,pad*2), (0,pad*2), (0,0)), 'constant')
    return np.pad(x, ((0,0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0,0)), 'constant')

def conv(x, w, b):
    """ x (f, f, n_C), W (f, f, n_C_prev), b (scalar) """
    return np.sum(np.multiply(x, w)) + float(b)

def conv2d_tf(x, W, b, stride, pad, out_type):
    """ Check that it's my implementation of this that's the problem """
    pad_name = pad.name # Get "SAME" or "VALID"
    return np.array(
        tf.nn.conv2d(x, W, [1, stride, stride, 1], pad_name) + b,
    dtype=out_type)

def calc_padding(input_size, filter_size, stride, pad_type):
    """
    See:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L20

    Official equations given on:
    https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
    https://www.tensorflow.org/api_guides/python/nn#Convolution
    """
    if pad_type == Padding.VALID:
        output_size = int((input_size - filter_size + stride) / stride)
        print(input_size, filter_size, stride)
        pad_before = 0
        pad_after = 0
    elif pad_type == Padding.SAME:
        output_size = int((input_size + stride - 1) / stride)
        pad_needed = max(0, (output_size - 1)*stride + filter_size - input_size)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
    else:
        raise NotImplementedError("Only SAME and VALID padding types implemented")

    print("Output size:", output_size, "Pad before:", pad_before, "Pad after:", pad_after)
    old_calc = int((input_size-filter_size+pad_before+pad_after)/stride)+1
    if output_size != old_calc:
        print("Old calc:", old_calc, "New calc:", output_size)

    assert output_size >= 0, "output_size must be non-negative after padding"
    return output_size, pad_before, pad_after

def conv2d_mine(x, W, b, stride, pad, out_type):
    """
    Input: x (m, n_H_prev, n_W_prev, n_C_prev), W (f, f, n_C_prev, n_C), b (1, 1, 1, n_C)
    Output: (m, n_H, n_W, n_C)
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_ops.cc#L416
    https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/conv.cc#L262

    For faster implementation, maybe see:
    https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Dimensions of output volume
    #n_H = int((n_H_prev-f+2*pad)/stride)+1
    #n_W = int((n_W_prev-f+2*pad)/stride)+1
    #n_H = int(np.ceil((n_H_prev - f + 1) / stride))
    #n_W = int(np.ceil((n_W_prev - f + 1) / stride))

    # Pad input
    n_H, pad_before_h, pad_after_h = calc_padding(n_H_prev, f, stride, pad)
    n_W, pad_before_w, pad_after_w = calc_padding(n_W_prev, f, stride, pad)
    A_prev_pad = zero_pad(x, pad_before_h, pad_after_h, pad_before_w, pad_after_w)

    # Init output with zeros
    output = np.zeros((m,n_H,n_W,n_C), dtype=out_type)

    for i in range(m): # over batches (probably only 1)
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):         # vertical axis
            for w in range(n_W):     # horiz axis
                for c in range(n_C): # for each output filter
                    # Portion of image for input to convolution
                    assert h*stride+f <= a_prev_pad.shape[0], "out of bounds"
                    assert w*stride+f <= a_prev_pad.shape[1], "out of bounds"
                    a_slice_prev = a_prev_pad[h*stride:h*stride+f, w*stride:w*stride+f, :]
                    # Convolve
                    assert len(b.shape) == 1, "Assuming single bias per channel/filter"
                    output[i, h, w, c] = conv(a_slice_prev, W[:,:,:,c], b[c])

    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h#L193
    """
    for batch in range(m):
        for out_y in range(n_H):
            for out_x in range(n_W):
                for out_channel in range(n_C):
                    in_x_origin = out_x * stride - int(pad)
                    in_y_origin = out_y * stride - int(pad)
                    total = 0.0

                    for filter_y in range(f):
                        for filter_x in range(f):
                            in_x = in_x_origin + filter_x
                            in_y = in_y_origin + filter_y

                            if in_x >= 0 and in_y >= 0 and in_x < n_W_prev and in_y < n_H_prev:
                                for in_channel in range(n_C_prev):
                                        input_value = x[batch, in_y, in_x, in_channel]
                                        filter_value = W[filter_y, filter_x, in_channel, out_channel]

                                        total += input_value * filter_value

                    output[batch, out_y, out_x, out_channel] = total + b[out_channel]
    """

    return output

def depthwise_conv2d_tf(x, W, b, stride, pad, out_type):
    """ Check that it's my implementation of this that's the problem """
    pad_name = pad.name # Get "SAME" or "VALID"
    return np.array(
        tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], pad_name) + b,
    dtype=out_type)

def depthwise_conv2d_mine(x, W, b, stride, pad, out_type):
    """
    See "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
        https://arxiv.org/pdf/1704.04861.pdf

    Note: this does not do the 1x1 step afterward. The graph appears to have a
    separate conv2d that does the 1x1's.

    Input: x (m, n_H_prev, n_W_prev, n_C_prev), W (f, f, n_C_prev, n_C), b (1, 1, 1, n_C)
    Output: (m, n_H, n_W, n_C_prev)
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
    (f, f, n_C_prev, n_C) = W.shape
    assert n_C == 1, "first dimension == 1 for depthwise conv2d weights"

    # Dimensions of output volume
    #n_H = int((n_H_prev-f+2*pad)/stride)+1
    #n_W = int((n_W_prev-f+2*pad)/stride)+1

    # Pad input
    n_H, pad_before_h, pad_after_h = calc_padding(n_H_prev, f, stride, pad)
    n_W, pad_before_w, pad_after_w = calc_padding(n_W_prev, f, stride, pad)
    A_prev_pad = zero_pad(x, pad_before_h, pad_after_h, pad_before_w, pad_after_w)

    # Init output with zeros
    output = np.zeros((m,n_H,n_W,n_C_prev), dtype=out_type)

    for i in range(m): # over batches (probably only 1)
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):         # vertical axis
            for w in range(n_W):     # horiz axis
                for c in range(n_C_prev): # for each output filter (same as # of input filters)
                    # Portion of image for input to convolution
                    a_slice_prev = a_prev_pad[h*stride:h*stride+f, w*stride:w*stride+f, c]
                    # Convolve
                    assert W.shape[3] == 1, "Assuming W.shape[3] == 1"
                    assert len(b.shape) == 1, "Assuming single bias per channel/filter"
                    output[i, h, w, c] = conv(a_slice_prev, W[:,:,c,0], b[c])

    return output

class Conv:
    """ Base class for Conv2D and DepthwiseConv2D, which are almost the same
    but call slightly different functions """
    def __repr__(self):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, input_tensors, input_buffers, options):
        assert len(input_buffers) == 3, str(self)+" assumes three inputs"

        input_data = input_buffers[0]
        weights = input_buffers[1]
        weights = np.transpose(weights, (1,2,3,0)) # TODO transpose once?
        biases = input_buffers[2]
        activation = options["activation"]
        stride = options["stride"]

        print("Input shape:", input_data.shape)
        print("Weights shape:", weights.shape)
        print("Output shape:", options["out_shape"])

        n = input_data.shape[1] # width (or height, since same)
        f = weights.shape[1] # fxf filter
        #padding = options["padding"](n, f, stride)
        padding = options["padding"]
        self.options = options

        result = self.eval(input_data, weights, biases, stride, padding,
                options["out_type"])
        
        return activation(result)

class Conv2D(Conv):
    def __repr__(self):
        return "Conv2D"

    def eval(self, *args, **kwargs):
        return conv2d_mine(*args, **kwargs)
        #return conv2d_tf(*args, **kwargs)

class DepthwiseConv2D(Conv):
    def __repr__(self):
        return "DepthwiseConv2D"

    def eval(self, *args, **kwargs):
        return depthwise_conv2d_mine(*args, **kwargs)
        #return depthwise_conv2d_tf(*args, **kwargs)

class ActivationNone:
    def __repr__(self):
        return "ActivationNone"

    def __call__(self, inputs):
        return inputs

class ActivationRELU6:
    def __repr__(self):
        return "ActivationRELU6"

    def __call__(self, inputs):
        return np.minimum(np.maximum(inputs, 0), 6)

def get_model(filename):
    """ Get .tflite model from the FlatBuffer file """
    with open(filename, "rb") as f:
        buf = bytearray(f.read())

    model = tflite.Model.Model.GetRootAsModel(buf, 0)

    assert model.Version() == 3, \
        "Only support schema version 3 at the moment"
    assert model.MetadataBufferLength() == 0, \
        "Do not support metadata_buffer at the moment"

    return model

def get_op(op):
    """ Right now return a string for the operator, later return a function
    that'll actually execute the operator """
    operator = None
    custom = op.CustomCode()
    builtin = op.BuiltinCode()

    if builtin == tflite.BuiltinOperator.BuiltinOperator.CONCATENATION:
        operator = Concat()
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.CONV_2D:
        operator = Conv2D()
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D:
        operator = DepthwiseConv2D()
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.LOGISTIC:
        operator = Logistic()
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.RESHAPE:
        operator = Reshape()
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.CUSTOM:
        if custom.decode() == "TFLite_Detection_PostProcess":
            operator = TFLite_Detection_PostProcess()
        else:
            raise NotImplementedError("custom op "+custom.decode()+" not implemented")
    else:
        raise NotImplementedError("builtin op "+str(builtin)+" not implemented")

    return operator

def get_activation(act):
    """ Right now return a string for the activation function, later return a
    function that'll actually execute the activation function """
    activation = None

    if act == tflite.ActivationFunctionType.ActivationFunctionType.NONE:
        activation = ActivationNone()
    elif act == tflite.ActivationFunctionType.ActivationFunctionType.RELU6:
        activation = ActivationRELU6()
    else:
        raise NotImplementedError("activation "+str(act)+" not implemented")

    return activation

def get_padding(pad):
    """ Right now return a string for the padding name """
    padding = None

    if pad == tflite.Padding.Padding.SAME:
        padding = Padding.SAME
    elif pad == tflite.Padding.Padding.VALID:
        padding = Padding.VALID
    else:
        raise NotImplementedError("padding "+str(pad)+" not implemented")

    return padding

def get_ops(model):
    """ Get all operators used in a model """
    ops = []
    op_codes_len = model.OperatorCodesLength()

    for i in range(op_codes_len):
        op = model.OperatorCodes(i)
        ops.append(get_op(op))

    return ops

def conv2d_options(op):
    """ Get Conv2D options from BuiltinOptions union """
    conv2d_options = tflite.Conv2DOptions.Conv2DOptions()
    conv2d_options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    padding = get_padding(conv2d_options.Padding())
    stride = conv2d_options.StrideW()
    stride_h = conv2d_options.StrideH()
    activation = get_activation(conv2d_options.FusedActivationFunction())
    dilation_w_factor = conv2d_options.DilationWFactor()
    dilation_h_factor = conv2d_options.DilationHFactor()

    assert stride == stride_h, \
        "Only support stride_w == stride_h at the moment"
    assert dilation_w_factor == 1, \
        "Only support dilation_w_factor == 1 at the moment"
    assert dilation_h_factor == 1, \
        "Only support dilation_h_factor == 1 at the moment"

    return {"activation": activation, "padding": padding, "stride": stride}

def depthwise_options(op):
    """ Get DepthwiseConv2D options from BuiltinOptions union """
    options = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    padding = get_padding(options.Padding())
    stride = options.StrideW()
    stride_h = options.StrideH()
    depth_multiplier = options.DepthMultiplier()
    activation = get_activation(options.FusedActivationFunction())
    dilation_w_factor = options.DilationWFactor()
    dilation_h_factor = options.DilationHFactor()

    assert stride == stride_h, \
        "Only support stride_w == stride_h at the moment"
    assert dilation_w_factor == 1, \
        "Only support dilation_w_factor == 1 at the moment"
    assert dilation_h_factor == 1, \
        "Only support dilation_h_factor == 1 at the moment"
    assert depth_multiplier == 1, \
        "Only support depth_multiplier == 1 at the moment"

    return {"activation": activation, "padding": padding, "stride": stride}

def concat_options(op):
    """ Get Concatenation options from BuiltinOptions union """
    options = tflite.ConcatenationOptions.ConcatenationOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    axis = options.Axis()
    activation = options.FusedActivationFunction()

    assert activation == tflite.ActivationFunctionType.ActivationFunctionType.NONE, \
        "Only support activation == None at the moment for concat"

    return {"axis": axis}

def reshape_options(op):
    """ Get Reshape options from BuiltinOptions union """
    options = tflite.ReshapeOptions.ReshapeOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    shape = options.NewShapeAsNumpy()

    return {"shape": shape}

def get_options(op):
    """ Get options for specified op """
    options = {}
    options_type = op.BuiltinOptionsType()

    if options_type == 0:
        pass
    elif options_type == tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions:
        options = conv2d_options(op)
    elif options_type == tflite.BuiltinOptions.BuiltinOptions().DepthwiseConv2DOptions:
        options = depthwise_options(op)
    elif options_type == tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions:
        options = concat_options(op)
    elif options_type == tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions:
        options = reshape_options(op)
    else:
        raise NotImplementedError("options "+str(options_type)+" not implemented")

    # TODO also handle custom_options probably

    assert op.MutatingVariableInputsLength() == 0, \
        "Do not support mutating_variable_inputs at the moment"

    return options

def get_type(t):
    """ Get type of tensor """
    tensor_type = None

    if t == tflite.TensorType.TensorType.FLOAT32:
        tensor_type = np.float32
    elif t == tflite.TensorType.TensorType.INT32:
        tensor_type = np.int32
    elif t == tflite.TensorType.TensorType.UINT8:
        tensor_type = np.uint8
    else:
        raise NotImplementedError("tensor type "+str(t)+" not implemented")

    return tensor_type

def get_graph(model):
    """ Get the graph from the model """
    subgraph_len = model.SubgraphsLength()

    assert subgraph_len == 1, \
        "Only support subgraph_len == 1 at the moment"

    return model.Subgraphs(0)

def get_bufs(model):
    """ Get all the buffers from the model """
    bufs = []
    bufs_len = model.BuffersLength()

    for i in range(bufs_len):
        buf = model.Buffers(i)
        bufs.append(buf.DataAsNumpy())

    return bufs

def get_tensors(subgraph):
    """ Get all tensors in the subgraph """
    tensors = []
    tensors_len = subgraph.TensorsLength()

    for j in range(tensors_len):
        tensor = subgraph.Tensors(j)
        name = tensor.Name().decode()
        shape = tensor.ShapeAsNumpy()
        tensor_type = get_type(tensor.Type())
        tensor_buf_index = tensor.Buffer()
        quant = tensor.Quantization()
        quant_scale = quant.ScaleAsNumpy()
        quant_zero_point = quant.ZeroPointAsNumpy()
        is_variable = tensor.IsVariable()

        assert is_variable == False, \
            "Only support is_variable == False at the moment"
        assert quant_scale == 0 and quant_zero_point == 0, \
            "Do not support quantization at the moment "+ \
            "(float probably faster on GPU anyway)"

        tensors.append({
            "name": name,
            "shape": shape,
            "type": tensor_type,
            "buffer": tensor_buf_index,
            #"quant_scale": quant_scale,
            #"quant_zero": quant_zero_point,
        })

    return tensors

def get_operators(graph):
    """ Get operators from graph """
    operators = []
    operators_len = graph.OperatorsLength()

    for i in range(operators_len):
        op = graph.Operators(i)
        op_index = op.OpcodeIndex()

        inputs = op.InputsAsNumpy()
        outputs = op.OutputsAsNumpy()
        options = get_options(op)

        operators.append({
            "op": op_index,
            "inputs": inputs,
            "outputs": outputs,
            "options": options
        })

    return operators

def display_model(model):
    """ For debugging, output the model """
    ops = get_ops(model)
    bufs = get_bufs(model)
    graph = get_graph(model)
    tensors = get_tensors(graph)
    operators = get_operators(graph)

    inputs = graph.InputsAsNumpy()
    outputs = graph.OutputsAsNumpy()

    print("Inputs")
    for i in inputs:
        t = tensors[i]
        print(i, t)

    print("Operators")
    for o in operators:
        # Make op readable
        pretty_o = copy.deepcopy(o)
        pretty_o["op"] = str(ops[pretty_o["op"]])
        print(pretty_o)

        for t in o["inputs"]:
            print(" in: ", t, tensors[t])
        for t in o["outputs"]:
            print(" out: ", t, tensors[t])

    print("Outputs")
    for o in outputs:
        t = tensors[o]
        print(o, t)

def get_tensors_by_index(tensors, indices):
    """ Return a list of the desired tensors """
    return [tensors[t] for t in indices]

def get_tensor_buffers(bufs, tensors):
    """ Return a list of buffers of specified by the given tensors """
    buffers = []

    for t in tensors:
        # Reinterpret bytes as correct type and reshape
        buf = bufs[t["buffer"]]
        buf = np.frombuffer(buf, dtype=t["type"]).reshape(t["shape"])
        buffers.append(buf)

    return buffers

def run_model(model, input_data):
    """ Run model on given input data """
    ops = get_ops(model)
    bufs = get_bufs(model)
    graph = get_graph(model)
    tensors = get_tensors(graph)
    operators = get_operators(graph)

    inputs = graph.InputsAsNumpy()
    outputs = graph.OutputsAsNumpy()

    assert len(inputs) == 1, \
        "Only supports models with a single input at the moment"

    # Set input data
    input_tensor = tensors[inputs[0]]

    assert all(input_data.shape == input_tensor["shape"]), \
        "Input data must be of shape "+str(input_tensor["shape"])+\
        " but is of shape "+str(input_data.shape)

    bufs[input_tensor["buffer"]] = input_data

    # Execute operations
    for operator in operators:
        # What operation to perform
        op = ops[operator["op"]]
        options = operator["options"]

        input_name = tensors[operator["inputs"][0]]["name"]
        print("Input", input_name, "running op", op)

        # TODO Skipping the custom op for now
        if isinstance(op, TFLite_Detection_PostProcess):
            continue

        # We need to know what format to create the result in
        output_tensor = tensors[operator["outputs"][0]]
        options["out_type"] = output_tensor["type"]
        options["out_shape"] = output_tensor["shape"]
        options["input_name"] = input_name

        # Get input tensors
        input_tensors = get_tensors_by_index(tensors, operator["inputs"])

        for t in input_tensors:
            # Some are by default just a 0, so make sure it's not when we use it
            assert not isinstance(bufs[t["buffer"]], int), \
                "Input buffer "+str(t["buffer"])+" must be defined by time it's used: "+ \
                str(bufs[t["buffer"]])

        input_buffers = get_tensor_buffers(bufs, input_tensors)

        # Run operation
        output_data = op(input_tensors, input_buffers, options)

        assert len(operator["outputs"]) == 1, \
            "Only support single output at the moment"

        # Save result to output tensor
        assert all(output_data.shape == output_tensor["shape"]), \
            "Output data must be of shape "+str(output_tensor["shape"])+\
            " but is of shape "+str(output_data.shape)

        bufs[output_tensor["buffer"]] = output_data

    # Get output
    #results = []

    #for o in outputs:
    #    t = tensors[o]
    #    buf = t["buffer"]
    #    results.append(buf)

    # Get different output not requiring the custom op
    prediction_boxes = None
    prediction_classes = None

    for t in tensors:
        if t["name"] == "Squeeze":
            prediction_boxes = bufs[t["buffer"]]
        elif t["name"] == "convert_scores":
            prediction_classes = bufs[t["buffer"]]

    np.save("tflite_manual.npy", {
        t["name"]: bufs[t["buffer"]] for t in tensors
    })
    print("Total number of tensors:", len(tensors))

    return prediction_boxes, prediction_classes

def load_test_image(test_image_dir, width=300, height=300,
        input_mean=127.5, input_std=127.5, index=-1):
    """ Load one test image """
    test_images = [os.path.join(d, f) for d, f in find_files(test_image_dir)]
    img = Image.open(test_images[index])
    img = img.resize((width, height))
    img = load_image_into_numpy_array(img)
    img = (np.float32(img) - input_mean) / input_std
    img = np.expand_dims(img, axis=0)
    return img

def tests():
    """ See if my implementation is consistent with the TensorFlow Lite ones """
    # Conv2D
    #
    # See:
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/conv_test.cc
    data = np.array([
        1, 1, 1, 1, 2, 2, 2, 2,
        1, 2, 3, 4, 1, 2, 3, 4,
    ]).reshape((2, 2, 4, 1)).astype(np.float32)
    weights = np.array([
        1, 2, 3, 4,
        -1, 1, -1, 1,
        -1, -1, 1, 1
    ]).reshape((3, 2, 2, 1)).transpose((1,2,3,0)).astype(np.float32)
    bias = np.array([1, 2, 3]).astype(np.float32)
    stride = 1
    result_tf = conv2d_tf(data, weights, bias, stride, Padding.VALID, np.float32)
    result = conv2d_mine(data, weights, bias, stride, Padding.VALID, np.float32)
    assert (result == np.array([
        18, 2, 5, 18, 2, 5, 18, 2, 5,
        17, 4, 3, 27, 4, 3, 37, 4, 3]).reshape((2, 1, 3, 3))).all(), \
        "Test 1 gives "+str(result)

    # HandCalculatedWithBiasFloat32
    w = 4; h = 3; depth = 1; f = 3; filters = 1; stride = 1;
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((1, h, w, depth)).astype(np.float32)
    weights = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape((depth, f, f, filters)).transpose((1,2,3,0)).astype(np.float32)
    bias = np.array([10]).astype(np.float32)
    result_tf = conv2d_tf(data, weights, bias, stride, Padding.SAME, np.float32)
    result = conv2d_mine(data, weights, bias, stride, Padding.SAME, np.float32)
    assert (result == np.array([
        115, 160, 193, 105, 245, 322,
        367, 188, 197, 244, 271, 131]).reshape(1, h, w, filters)).all(), \
        "Test 2 gives "+str(result)

    # HandCalculatedValidFloat32
    w = 4; h = 3; depth = 1; f = 3; filters = 1; stride = 1;
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((1, h, w, depth)).astype(np.float32)
    weights = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape((depth, f, f, filters)).transpose((1,2,3,0)).astype(np.float32)
    bias = np.array([0]).astype(np.float32)
    result = conv2d_mine(data, weights, bias, stride, Padding.VALID, np.float32)
    assert (result == np.array([312, 357]).reshape(1, 1, 2, 1)).all(), \
        "Test 3 gives "+str(result)

    # HandCalculatedWithBiasFloat32 but with stride of 2
    w = 4; h = 3; depth = 1; f = 3; filters = 1; stride = 2;
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((1, h, w, depth)).astype(np.float32)
    weights = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape((depth, f, f, filters)).transpose((1,2,3,0)).astype(np.float32)
    bias = np.array([10]).astype(np.float32)
    result_tf = conv2d_tf(data, weights, bias, stride, Padding.SAME, np.float32)
    result = conv2d_mine(data, weights, bias, stride, Padding.SAME, np.float32)
    assert (result == result_tf).all(), "Test 4 gives "+str(result)
    
    # Depthwise Conv2D
    #
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/depthwise_conv_test.cc
    # SimpleTest
    data = np.array([
        1, 2, 7, 8,
        3, 4, 9, 10,
        5, 6, 11, 12]).reshape((1, 3, 2, 2)).astype(np.float32)
    weights = np.array([
        1, 2, 3, 4,
        -9, 10, -11, 12,
        5, 6, 7, 8,
        13, -14, 15, -16]).reshape((1, 2, 2, 4)).astype(np.float32)
    bias = np.array([1, 2, 3, 4]).astype(np.float32)
    stride = 1
    # TODO
    #pad = 0
    #print(data)
    #print(weights)
    #result = depthwise_conv2d(data, weights, bias, stride, Padding.VALID, np.float32)
    #print(result)
    #print(result.shape)
    #assert (result == np.array([
    #    115, 160, 193, 105, 245, 322,
    #    367, 188, 197, 244, 271, 131]).reshape(1, h, w, filters)).all(), "Test 5"

if __name__ == "__main__":
    tests()

    model = get_model("detect_float.tflite")
    display_model(model)

    img = load_test_image("test_images")
    run_model(model, img)
