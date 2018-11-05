#!/usr/bin/env python3
import flatbuffers
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

from ..image import find_files, load_image_into_numpy_array

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
        operator = "Concat"
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.CONV_2D:
        operator = "Conv2D"
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D:
        operator = "DepthwiseConv2D"
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.LOGISTIC:
        operator = "Logistic"
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.RESHAPE:
        operator = "Reshape"
    elif builtin == tflite.BuiltinOperator.BuiltinOperator.CUSTOM:
        operator = "Custom:" + custom.decode()
    else:
        raise NotImplementedError("builtin op "+str(builtin)+" not implemented")
    
    return operator

def get_activation(act):
    """ Right now return a string for the activation function, later return a
    function that'll actually execute the activation function """
    activation = None

    if act == tflite.ActivationFunctionType.ActivationFunctionType.NONE:
        activation = "None"
    elif act == tflite.ActivationFunctionType.ActivationFunctionType.RELU6:
        activation = "RELU6"
    else:
        raise NotImplementedError("activation "+str(act)+" not implemented")

    return activation

def get_padding(pad):
    """ Right now return a string for the padding name """
    padding = None

    if pad == tflite.Padding.Padding.SAME:
        padding = "Same"
    elif pad == tflite.Padding.Padding.VALID:
        padding = "Valid"
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
        tensor_type = "Float32"
    elif t == tflite.TensorType.TensorType.INT32:
        tensor_type = "Int32"
    elif t == tflite.TensorType.TensorType.UINT8:
        tensor_type = "Uint8"
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
        is_variable = tensor.IsVariable()

        assert is_variable == False, \
            "Only support is_variable == False at the moment"

        tensors.append({
            "name": name,
            "shape": shape,
            "type": tensor_type,
            "buffer": tensor_buf_index,
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
        print(o)
        
        for t in o["inputs"]:
            print(" in: ", t, tensors[t])
        for t in o["outputs"]:
            print(" out: ", t, tensors[t])

    print("Outputs")
    for o in outputs:
        t = tensors[o]
        print(o, t)

def run_model(model):
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
        print(o)
        
        for t in o["inputs"]:
            print(" in: ", t, tensors[t])
        for t in o["outputs"]:
            print(" out: ", t, tensors[t])

    print("Outputs")
    for o in outputs:
        t = tensors[o]
        print(o, t)

if __name__ == "__main__":
    model = get_model("../detect_quantized.tflite")
    display_model(model)

    test_image_dir = "../test_images"
    test_images = [os.path.join(d, f) for d, f in find_files(test_image_dir)]
    print(test_images)
