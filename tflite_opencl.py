#!/usr/bin/env python3
"""
Run TF Lite model using OpenCL rather than the TensorFlow TF Lite implementation

The goal: take tflite_numpy.py and replace the numpy calculations with OpenCL
calculations. This will not be optimized, but it will hopefully generate the
correct results.

Next step: optimize and test on the RPi Zero.

Usage:
    time ./tflite_opencl.py

How to use this "reference implementation" (a.k.a. a hacky script):
 - Running this will output tflite_opencl.npy
 - Then run tflite_visualize.py to check the results on the last image in
   test_images (index == -1 at the moment)
"""
import os
import sys
import time
import flatbuffers
import numpy as np
import pyopencl as cl
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

Padding = Enum("Padding", "VALID SAME")
Activation = Enum("Activation", "NONE RELU6")
Operation = Enum("Operation", "CONCAT RESHAPE LOGISTIC CONV2D DEPTHWISECONV2D POSTPROCESS IM2COL MATMUL")

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

class TFLiteOpenCL:
    def __init__(self, model=None, interactive=False):
        # Allow for interactive choice of which OpenCL device, otherwise pick
        # the first platform (on RPi probably only the one GPU unless pocl is
        # also installed)
        #
        # If interactive, you can specify with environment variable PYOPENCL_CTX=
        if interactive:
            self.ctx = cl.create_some_context()
        else:
            platforms = cl.get_platforms()
            self.ctx = cl.Context(
                dev_type=cl.device_type.ALL,
                properties=[(cl.context_properties.PLATFORM, platforms[0])])

        self.prg = cl.Program(self.ctx, """
        /*
         * Non-linear functions -- compute function element-wise (so interpet
         * as a 1D array)
         */
        __kernel void relu6(__global const float* input, __global float* output)
        {
            const int id = get_global_id(0);
            output[id] = fmin(fmax(input[id], 0), 6);
        }

        __kernel void logistic(__global const float* input, __global float* output)
        {
            const int id = get_global_id(0);
            output[id] = 1.0 / (1.0 + exp(-input[id]));
        }

        /*
         * Reshaping does nothing except we reinterpret the output, so just copy
         */
        __kernel void copy(__global const float* input, __global float* output)
        {
            const int id = get_global_id(0);
            output[id] = input[id];
        }

        /*
         * Copy data from 6 arrays into one
         */
        __kernel void concat612(
            const int sz1, const int sz2, const int sz3,
            const int sz4, const int sz5, const int sz6,
            __global const float* input1, __global const float* input2,
            __global const float* input3, __global const float* input4,
            __global const float* input5, __global const float* input6,
            __global float* output
        )
        {
            int index = 0;
            for (int i = 0; i < sz1; ++i, ++index)
                output[index] = input1[i];
            for (int i = 0; i < sz2; ++i, ++index)
                output[index] = input2[i];
            for (int i = 0; i < sz3; ++i, ++index)
                output[index] = input3[i];
            for (int i = 0; i < sz4; ++i, ++index)
                output[index] = input4[i];
            for (int i = 0; i < sz5; ++i, ++index)
                output[index] = input5[i];
            for (int i = 0; i < sz6; ++i, ++index)
                output[index] = input6[i];
        }

        /* Conv2D
         * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_ops.cc#L416
         * https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/kernels/conv.cc#L262
         *
         * For faster implementation, maybe see:
         * https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
         */
        __kernel void conv2d(
            const int m, const int n_H, const int n_W, const int n_C,
            const int stride, const int filter_dim,
            const int pad_before_w, const int pad_before_h,
            const int pad_after_w, const int pad_after_h,
            const int n_H_prev, const int n_W_prev, const int n_C_prev,
            __global const float* x, __global const float* w, __global const float* b,
            __global float* output)
        {
            const int out_y = get_global_id(0);
            const int out_x = get_global_id(1);
            const int out_channel = get_global_id(2);
            const int in_x_origin = out_x*stride - pad_before_w;
            const int in_y_origin = out_y*stride - pad_before_h;

            // x offsets
            const int xo1 = n_H_prev*n_W_prev*n_C_prev;
            const int xo2 = n_W_prev*n_C_prev;
            const int xo3 = n_C_prev;

            // w offsets
            const int wo1 = filter_dim*n_C_prev*n_C;
            const int wo2 = n_C_prev*n_C;
            const int wo3 = n_C;

            // output offsets
            const int oo1 = n_H*n_W*n_C;
            const int oo2 = n_W*n_C;
            const int oo3 = n_C;

            for (int batch = 0; batch < m; ++batch) {
                float total = 0;

                for (int filter_y = 0; filter_y < filter_dim; ++filter_y) {
                    for (int filter_x = 0; filter_x < filter_dim; ++filter_x) {
                        const int in_x = in_x_origin + filter_x;
                        const int in_y = in_y_origin + filter_y;

                        if (in_x >= 0 && in_y >= 0 && in_x < n_W_prev && in_y < n_H_prev) {
                            for (int in_channel = 0; in_channel < n_C_prev; ++in_channel) {
                                const float input_value = x[batch*xo1 + in_y*xo2 + in_x*xo3 + in_channel];
                                const float filter_value = w[filter_y*wo1 + filter_x*wo2 + in_channel*wo3 + out_channel];

                                total += input_value * filter_value;
                            }
                        }
                    }
                }

                output[batch*oo1 + out_y*oo2 + out_x*oo3 + out_channel] = total + b[out_channel];
            }
        }

        __kernel void conv2d_relu6(
            const int m, const int n_H, const int n_W, const int n_C,
            const int stride, const int filter_dim,
            const int pad_before_w, const int pad_before_h,
            const int pad_after_w, const int pad_after_h,
            const int n_H_prev, const int n_W_prev, const int n_C_prev,
            __global const float* x, __global const float* w, __global const float* b,
            __global float* output)
        {
            const int out_y = get_global_id(0);
            const int out_x = get_global_id(1);
            const int out_channel = get_global_id(2);
            const int in_x_origin = out_x*stride - pad_before_w;
            const int in_y_origin = out_y*stride - pad_before_h;

            // x offsets
            const int xo1 = n_H_prev*n_W_prev*n_C_prev;
            const int xo2 = n_W_prev*n_C_prev;
            const int xo3 = n_C_prev;

            // w offsets
            const int wo1 = filter_dim*n_C_prev*n_C;
            const int wo2 = n_C_prev*n_C;
            const int wo3 = n_C;

            // output offsets
            const int oo1 = n_H*n_W*n_C;
            const int oo2 = n_W*n_C;
            const int oo3 = n_C;

            for (int batch = 0; batch < m; ++batch) {
                float total = 0;

                for (int filter_y = 0; filter_y < filter_dim; ++filter_y) {
                    for (int filter_x = 0; filter_x < filter_dim; ++filter_x) {
                        const int in_x = in_x_origin + filter_x;
                        const int in_y = in_y_origin + filter_y;

                        if (in_x >= 0 && in_y >= 0 && in_x < n_W_prev && in_y < n_H_prev) {
                            for (int in_channel = 0; in_channel < n_C_prev; ++in_channel) {
                                const float input_value = x[batch*xo1 + in_y*xo2 + in_x*xo3 + in_channel];
                                const float filter_value = w[filter_y*wo1 + filter_x*wo2 + in_channel*wo3 + out_channel];

                                total += input_value * filter_value;
                            }
                        }
                    }
                }

                output[batch*oo1 + out_y*oo2 + out_x*oo3 + out_channel] = fmin(fmax(total + b[out_channel], 0), 6);
            }
        }

        /*
         * Depthwise Conv2d
         * See "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
         * https://arxiv.org/pdf/1704.04861.pdf
         *
         * Note: this does not do the 1x1 step afterward. The graph appears to have a
         * separate conv2d that does the 1x1's.
         */
        __kernel void depthwise_conv2d(
            const int m, const int n_H, const int n_W, const int n_C,
            const int stride, const int filter_dim,
            const int pad_before_w, const int pad_before_h,
            const int pad_after_w, const int pad_after_h,
            const int n_H_prev, const int n_W_prev, const int n_C_prev,
            __global const float* x, __global const float* w, __global const float* b,
            __global float* output)
        {
            const int out_y = get_global_id(0);
            const int out_x = get_global_id(1);
            const int out_channel = get_global_id(2);
            const int in_x_origin = out_x*stride - pad_before_w;
            const int in_y_origin = out_y*stride - pad_before_h;

            // x offsets
            const int xo1 = n_H_prev*n_W_prev*n_C_prev;
            const int xo2 = n_W_prev*n_C_prev;
            const int xo3 = n_C_prev;

            // w offsets
            const int wo1 = filter_dim*n_C_prev*n_C;
            const int wo2 = n_C_prev*n_C;
            const int wo3 = n_C;

            // output offsets
            const int oo1 = n_H*n_W*n_C_prev;
            const int oo2 = n_W*n_C_prev;
            const int oo3 = n_C_prev;

            for (int batch = 0; batch < m; ++batch) {
                float total = 0;

                for (int filter_y = 0; filter_y < filter_dim; ++filter_y) {
                    for (int filter_x = 0; filter_x < filter_dim; ++filter_x) {
                        const int in_x = in_x_origin + filter_x;
                        const int in_y = in_y_origin + filter_y;

                        if (in_x >= 0 && in_y >= 0 && in_x < n_W_prev && in_y < n_H_prev) {
                            const float input_value = x[batch*xo1 + in_y*xo2 + in_x*xo3 + out_channel];
                            const float filter_value = w[filter_y*wo1 + filter_x*wo2 + out_channel*wo3 + 0];

                            total += input_value * filter_value;
                        }
                    }
                }

                output[batch*oo1 + out_y*oo2 + out_x*oo3 + out_channel] = total + b[out_channel];
            }
        }

        __kernel void depthwise_conv2d_relu6(
            const int m, const int n_H, const int n_W, const int n_C,
            const int stride, const int filter_dim,
            const int pad_before_w, const int pad_before_h,
            const int pad_after_w, const int pad_after_h,
            const int n_H_prev, const int n_W_prev, const int n_C_prev,
            __global const float* x, __global const float* w, __global const float* b,
            __global float* output)
        {
            const int out_y = get_global_id(0);
            const int out_x = get_global_id(1);
            const int out_channel = get_global_id(2);
            const int in_x_origin = out_x*stride - pad_before_w;
            const int in_y_origin = out_y*stride - pad_before_h;

            // x offsets
            const int xo1 = n_H_prev*n_W_prev*n_C_prev;
            const int xo2 = n_W_prev*n_C_prev;
            const int xo3 = n_C_prev;

            // w offsets
            const int wo1 = filter_dim*n_C_prev*n_C;
            const int wo2 = n_C_prev*n_C;
            const int wo3 = n_C;

            // output offsets
            const int oo1 = n_H*n_W*n_C_prev;
            const int oo2 = n_W*n_C_prev;
            const int oo3 = n_C_prev;

            for (int batch = 0; batch < m; ++batch) {
                float total = 0;

                for (int filter_y = 0; filter_y < filter_dim; ++filter_y) {
                    for (int filter_x = 0; filter_x < filter_dim; ++filter_x) {
                        const int in_x = in_x_origin + filter_x;
                        const int in_y = in_y_origin + filter_y;

                        if (in_x >= 0 && in_y >= 0 && in_x < n_W_prev && in_y < n_H_prev) {
                            const float input_value = x[batch*xo1 + in_y*xo2 + in_x*xo3 + out_channel];
                            const float filter_value = w[filter_y*wo1 + filter_x*wo2 + out_channel*wo3 + 0];

                            total += input_value * filter_value;
                        }
                    }
                }

                output[batch*oo1 + out_y*oo2 + out_x*oo3 + out_channel] = fmin(fmax(total + b[out_channel], 0), 6);
            }
        }

        __kernel void im2col(
            const int m, const int n_H, const int n_W, const int n_C,
            const int stride, const int filter_dim,
            const int pad_before_w, const int pad_before_h,
            const int pad_after_w, const int pad_after_h,
            const int n_H_prev, const int n_W_prev, const int n_C_prev,
            __global const float* x,  __global float* output)
        {
            const int out_y = get_global_id(0);
            const int out_x = get_global_id(1);
            const int in_x_origin = out_x*stride - pad_before_w;
            const int in_y_origin = out_y*stride - pad_before_h;

            // x offsets
            const int xo2 = n_W_prev*n_C_prev;
            const int xo3 = n_C_prev;

            // output offsets
            const int oo1 = n_W*filter_dim*filter_dim*n_C_prev;
            const int oo2 = filter_dim*filter_dim*n_C_prev;
            const int oo3 = filter_dim*n_C_prev;
            const int oo4 = n_C_prev;

            // TODO move output to end, copy to local buf, then copy to output after
            // the for loop

            for (int filter_y = 0; filter_y < filter_dim; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_dim; ++filter_x) {
                    const int in_x = in_x_origin + filter_x;
                    const int in_y = in_y_origin + filter_y;

                    for (int c = 0; c < n_C_prev; ++c) {
                        // I get "Optimizer: Filling dynamically sized memory is not yet implemented"
                        // errors if including an if statement, so instead read the potentially-invalid
                        // memory address and multiply it by 0 or 1 (if false, it's 0 ==> result is 0
                        // as desired)
                        const bool in_bounds = in_x >= 0 && in_y >= 0 && in_x < n_W_prev && in_y < n_H_prev;
                        output[out_y*oo1 + out_x*oo2 + filter_y*oo3 + filter_x*oo4 + c] = in_bounds*x[in_y*xo2 + in_x*xo3 + c];
                    }
                }
            }
        }

        /*
         * Based on: https://cnugteren.github.io/tutorial/pages/page4.html
         */
        // First naive implementation
        __kernel void matmul(const int M, const int N, const int K,
                            const __global float* A,
                            const __global float* B,
                            const __global float* bias,
                            __global float* C) {

            // Thread identifiers
            const int globalRow = get_global_id(0); // Row ID of C (0..M)
            const int globalCol = get_global_id(1); // Col ID of C (0..N)

            // Compute a single element (loop over K)
            float acc = 0.0f;
            for (int k=0; k<K; ++k) {
                acc += A[globalRow*K + k] * B[k*N + globalCol];
            }

            // Store the result
            C[globalRow*N + globalCol] = acc + bias[globalCol];
        }

        __kernel void matmul_relu6(const int M, const int N, const int K,
                            const __global float* A,
                            const __global float* B,
                            const __global float* bias,
                            __global float* C) {

            // Thread identifiers
            const int globalRow = get_global_id(0); // Row ID of C (0..M)
            const int globalCol = get_global_id(1); // Col ID of C (0..N)

            // Compute a single element (loop over K)
            float acc = 0.0f;
            for (int k=0; k<K; ++k) {
                acc += A[globalRow*K + k] * B[k*N + globalCol];
            }

            // Store the result
            C[globalRow*N + globalCol] = fmin(fmax(acc + bias[globalCol], 0), 6);
        }
        """).build(["-cl-fast-relaxed-math"])

        # List of what to do
        self.operations = []
        # List of weight/bias (read-only) buffers
        self.weight_buffers = []
        # Set (no duplicates) of the IDs of the buffers we need for input/output
        self.need_buffer = set()
        # Reshapes don't do anything, so just make buffer list substitutions
        # This is a list of (a,b) tuples replacing a with b
        self.buffer_replacements = []

        # Load model now if provided, otherwise manually call load() later
        self.loaded = False
        if model is not None:
            self.load(model)

    def load(self, filename):
        """ Run model on given input data """
        print("Loading model")
        assert not self.loaded, "Cannot load multiple models"
        self.loaded = True

        model = self.get_model(filename)
        ops = self.get_ops(model)
        self.bufs = self.get_bufs(model)
        graph = self.get_graph(model)
        self.tensors = self.get_tensors(graph)
        operators = self.get_operators(graph)

        inputs = graph.InputsAsNumpy()
        outputs = graph.OutputsAsNumpy()

        assert len(inputs) == 1, \
            "Only supports models with a single input at the moment"

        # Save where we should set the input data
        input_tensor = self.tensors[inputs[0]]
        self.input_shape = input_tensor["shape"]
        self.input_buffer = input_tensor["buffer"]
        self.bufs[self.input_buffer] = np.empty(self.input_shape).astype(input_tensor["type"])

        # Create list of operations
        for operator in operators:
            # What operation to perform
            op = ops[operator["op"]]
            options = operator["options"]

            input_name = self.tensors[operator["inputs"][0]]["name"]
            print("Input", input_name, "op", op)

            # We need to know what format to create the result in
            output_tensor = self.tensors[operator["outputs"][0]]
            options["out_type"] = output_tensor["type"]
            options["out_shape"] = output_tensor["shape"]
            options["input_name"] = input_name

            # Get input tensors
            input_tensors = self.get_tensors_by_index(self.tensors, operator["inputs"])

            # Check we're only using exiting tensors
            for t in input_tensors:
                buf = self.replace_buffers(t["buffer"]) # buffer replacements
                # Some are by default just a 0, so make sure it's not when we use it
                assert not isinstance(self.bufs[buf], int), \
                    "Input buffer "+str(buf)+" must be defined by time it's used: "+ \
                    str(self.bufs[buf])

            input_buffers = self.get_tensor_buffers(self.bufs, input_tensors)

            # Where we'll write the output
            output_buffer = output_tensor["buffer"]

            if op == Operation.POSTPROCESS:
                continue # Skip post process for now
            elif op == Operation.CONV2D or op == Operation.DEPTHWISECONV2D:
                assert len(input_buffers) == 3, str(self)+" assumes three inputs"

                x = input_buffers[0]
                W = np.ascontiguousarray(np.transpose(input_buffers[1], (1,2,3,0)))
                b = np.ascontiguousarray(input_buffers[2])
                activation = options["activation"]
                stride = options["stride"]
                padding = options["padding"]

                print("Input shape:", x.shape)
                print("Weights shape:", W.shape)
                print("Output shape:", options["out_shape"])

                # Dimensions
                (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
                (f, f, n_C_prev, n_C) = W.shape

                # Calculate padding
                n_H, pad_before_h, pad_after_h = self.calc_padding(n_H_prev, f, stride, padding)
                n_W, pad_before_w, pad_after_w = self.calc_padding(n_W_prev, f, stride, padding)

                if op == Operation.CONV2D:
                    out_channels = n_C
                else:
                    assert n_C == 1, "first dimension == 1 for depthwise conv2d weights"
                    out_channels = n_C_prev

                # Init output of correct shape
                output_empty = np.empty((m,n_H,n_W,out_channels), dtype=options["out_type"])

                mf = cl.mem_flags
                w_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W)
                b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

                if op == Operation.CONV2D:
                    out_im2col = np.empty((m,n_H*n_W,f*f*n_C_prev), dtype=options["out_type"])
                    out_im2col_buf = cl.Buffer(self.ctx, mf.READ_WRITE, out_im2col.nbytes)

                    cl_ndrange = (n_H,n_W)
                    cl_args = (np.int32(m), np.int32(n_H), np.int32(n_W), np.int32(n_C),
                        np.int32(stride), np.int32(f),
                        np.int32(pad_before_w), np.int32(pad_before_h),
                        np.int32(pad_after_w), np.int32(pad_after_h),
                        np.int32(n_H_prev), np.int32(n_W_prev), np.int32(n_C_prev))
                    cl_inputs = self.replace_buffers([input_tensors[0]["buffer"]])
                    cl_weights = (out_im2col_buf,) # actually an output...
                    cl_output = None # output is in "weights" since init above

                    # Used multiple times, so allocate later
                    for i in cl_inputs:
                        self.need_buffer.add(i)
                    #self.need_buffer.add(cl_output)

                    self.operations.append((
                        Operation.IM2COL,
                        None,
                        cl_ndrange,
                        cl_args,
                        cl_weights,
                        cl_inputs,
                        cl_output
                    ))

                    M = n_H*n_W
                    K = f*f*n_C_prev
                    N = out_channels
                    cl_ndrange = (M,N)
                    cl_args = (np.int32(M), np.int32(N), np.int32(K))
                    cl_inputs = () # input is in "weights" since we init above
                    cl_weights = (out_im2col_buf, w_buf, b_buf) # only used once, so create buffer here
                    cl_output = output_buffer

                    # Used multiple times, so allocate later
                    #for i in cl_inputs:
                    #    self.need_buffer.add(i)
                    self.need_buffer.add(cl_output)

                    self.operations.append((
                        Operation.MATMUL,
                        activation,
                        cl_ndrange,
                        cl_args,
                        cl_weights,
                        cl_inputs,
                        cl_output
                    ))

                #
                # TODO also implement im2col for depthwise conv2d
                #
                else:
                    cl_ndrange = output_empty.shape[1:]
                    cl_args = (np.int32(m), np.int32(n_H), np.int32(n_W), np.int32(n_C),
                        np.int32(stride), np.int32(f),
                        np.int32(pad_before_w), np.int32(pad_before_h),
                        np.int32(pad_after_w), np.int32(pad_after_h),
                        np.int32(n_H_prev), np.int32(n_W_prev), np.int32(n_C_prev))
                    cl_weights = (w_buf, b_buf) # only used once, so create buffer here
                    cl_inputs = self.replace_buffers([input_tensors[0]["buffer"]])
                    cl_output = output_buffer

                    # Used multiple times, so allocate later
                    for i in cl_inputs:
                        self.need_buffer.add(i)
                    self.need_buffer.add(cl_output)

                    self.operations.append((
                        op,
                        activation,
                        cl_ndrange,
                        cl_args,
                        cl_weights,
                        cl_inputs,
                        cl_output
                    ))
            elif op == Operation.LOGISTIC:
                assert len(input_buffers) == 1, "Logistic assumes single input"
                x = input_buffers[0]
                cl_inputs = self.replace_buffers([input_tensors[0]["buffer"]])
                cl_output = output_buffer
                self.operations.append((
                    op,
                    None,
                    (np.prod(x.shape),), # treat as 1D array
                    (), (), # no custom args/weights
                    cl_inputs,
                    cl_output
                ))
                for i in cl_inputs:
                    self.need_buffer.add(i)
                self.need_buffer.add(cl_output)

                output_empty = np.empty(x.shape).astype(options["out_type"])
            elif op == Operation.RESHAPE:
                assert len(input_buffers) == 2, \
                    "Reshape takes tensor and shape as input"
                assert all(input_buffers[1] == options["shape"]), \
                    "input_buffers[1] != options[\"shape\"]"

                in_buf = input_tensors[0]["buffer"]
                out_buf = output_buffer
                # Replace all out_buf with in_buf
                self.buffer_replacements.append((out_buf, in_buf))
                output_empty = None

                # x = input_buffers[0]
                # cl_inputs = (input_tensors[0]["buffer"],)
                # cl_output = output_buffer
                # self.operations.append((
                #     op,
                #     None,
                #     (np.prod(x.shape),), # treat as 1D array
                #     (), (), # no custom args/weights
                #     cl_inputs,
                #     cl_output
                # ))
                # for i in cl_inputs:
                #     self.need_buffer.add(i)
                # self.need_buffer.add(cl_output)

                # # Calculate shape with numpy -- inefficient but we only do this
                # # at start, not when processing frames
                # new_shape = np.reshape(input_buffers[0], options["shape"]).astype(options["out_type"]).shape
                # output_empty = np.empty(new_shape).astype(options["out_type"])
            elif op == Operation.CONCAT:
                assert len(input_buffers) == 6, \
                    "Only support concat 6 at the moment"
                assert options["axis"] == 1, \
                    "Only support concat axis=1 at the moment"
                for b in input_buffers:
                    assert b.shape[0] == 1, \
                        "Only support concat with axis 0 == length 1 (i.e. batch size==1)"
                    assert len([i for i in b.shape if i is not 1]) == 2, \
                        "Only support concat with 2 non-1 axes at the moment"

                cl_inputs = self.replace_buffers([i["buffer"] for i in input_tensors])
                cl_output = output_buffer
                self.operations.append((
                    op,
                    None,
                    (1,), # TODO can we parallelize this?
                    # size of each interpreted as a 1D array passed in as args
                    tuple([np.int32(np.prod(i.shape)) for i in input_buffers]),
                    (), # no weights
                    cl_inputs,
                    cl_output
                ))
                for i in cl_inputs:
                    self.need_buffer.add(i)
                self.need_buffer.add(cl_output)

                # Calculate shape with numpy -- inefficient but we only do this
                # at start, not when processing frames
                new_shape = np.concatenate(input_buffers, options["axis"]).astype(options["out_type"]).shape
                output_empty = np.empty(new_shape).astype(options["out_type"])

            if output_empty is not None:
                # Find the output buffer for this operation
                assert len(operator["outputs"]) == 1, \
                    "Only support single output at the moment"

                # Save the newly-created output buffer to our list of buffers
                assert all(output_empty.shape == output_tensor["shape"]), \
                    "Output data must be of shape "+str(output_tensor["shape"])+\
                    " but is of shape "+str(output_empty.shape)
                self.bufs[output_buffer] = output_empty

        # Get output
        #results = []

        #for o in outputs:
        #    t = tensors[o]
        #    buf = t["buffer"]
        #    results.append(buf)

        print("Allocating buffers")
        # Allocate all the buffers that we determined we'll need to run
        self.allocate_buffers()

    def replace_buffers(self, bufs):
        """ Buffer replacements to get rid of the need of reshapes """
        new_bufs = []

        # Also allow passing in only a single buffer to make the replacements
        is_list = True

        if not isinstance(bufs, list):
            is_list = False
            bufs = [bufs]

        # Make replacements
        for buf in bufs:
            found = False

            # Make replacement if in the list
            for a,b in self.buffer_replacements:
                if buf == a:
                    found = True
                    new_bufs.append(b)

            # No replacement, just copy it
            if not found:
                new_bufs.append(buf)

        # If we didn't pass in a list, then just return the single item
        if not is_list:
            return new_bufs[0]
        else:
            return new_bufs

    def allocate_buffers(self):
        """ Create OpenCL buffers for the needed I/O buffers """
        mf = cl.mem_flags
        self.opencl_bufs = {} # Indexed by buffer ID

        for buf in self.need_buffer:
            cl_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.bufs[buf].nbytes)
            self.opencl_bufs[buf] = cl_buf

    def enqueue_op(self, queue, op):
        cl_op, cl_act, cl_ndrange, cl_args, cl_weights, cl_inputs, cl_output = op

        if cl_op == Operation.POSTPROCESS:
            return # Skip post process for now
        elif cl_op == Operation.IM2COL:
            f = self.prg.im2col
        elif cl_op == Operation.MATMUL:
            if cl_act == Activation.RELU6:
                f = self.prg.matmul_relu6
            else:
                f = self.prg.matmul
        elif cl_op == Operation.CONV2D:
            if cl_act == Activation.RELU6:
                f = self.prg.conv2d_relu6
            else:
                f = self.prg.conv2d
        elif cl_op == Operation.DEPTHWISECONV2D:
            if cl_act == Activation.RELU6:
                f = self.prg.depthwise_conv2d_relu6
            else:
                f = self.prg.depthwise_conv2d
        elif cl_op == Operation.LOGISTIC:
            f = self.prg.logistic
        elif cl_op == Operation.RESHAPE:
            return
        elif cl_op == Operation.CONCAT:
            f = self.prg.concat612

        inputs = tuple([self.opencl_bufs[i] for i in cl_inputs])

        if cl_output is None:
            f(queue, cl_ndrange, None,
                    *cl_args,
                    *inputs,
                    *cl_weights)
        else:
            f(queue, cl_ndrange, None,
                    *cl_args,
                    *inputs,
                    *cl_weights,
                    self.opencl_bufs[cl_output])

    def load_buf(self, queue, buf):
        """
        Load data from OpenCL back into the desired buffer
        Data will be in self.bufs[buf] after this
        """
        cl.enqueue_copy(queue, self.bufs[buf], self.opencl_bufs[buf])
        return self.bufs[buf]

    def run(self, input_data):
        print("Running model")
        assert self.loaded, "Must have a model loaded first"

        assert all(input_data.shape == self.input_shape), \
            "Input data must be of shape "+str(self.input_shape)+\
            " but is of shape "+str(input_data.shape)

        # Set input data
        mf = cl.mem_flags
        self.bufs[self.input_buffer] = np.ascontiguousarray(input_data)
        self.opencl_bufs[self.input_buffer] = cl.Buffer(self.ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bufs[self.input_buffer])

        with cl.CommandQueue(self.ctx) as queue:
            # Enqueue operations
            for i, op in enumerate(self.operations):
                print("Enqueing op", i)
                t = time.time()
                self.enqueue_op(queue, op)
                t = time.time() - t
                print("Took", t, "s")

                # TODO do I need a cl.enqueue_barrier(queue)?
                # or maybe cl.wait_for_events(event) and handle which outputs
                # are used for certain inputs?
                cl.enqueue_barrier(queue)

            # Get different output not requiring the custom op
            #
            # Note: only when we request the result does it actually run the
            # network, so this takes a long time
            prediction_boxes = None
            prediction_classes = None

            print("Fetching results")
            t = time.time()
            for tensor in self.tensors:
                buf = self.replace_buffers(tensor["buffer"])
                if tensor["name"] == "Squeeze":
                    prediction_boxes = self.load_buf(queue, buf)
                elif tensor["name"] == "convert_scores":
                    prediction_classes = self.load_buf(queue, buf)
            t = time.time() - t
            print("Took", t, "s")

        # Note: only the prediction boxes/classes buffers will have valid data
        # in them though unless we load *all* the buffers in the above for loop
        np.save("tflite_opencl.npy", {
            t["name"]: self.bufs[t["buffer"]] for t in self.tensors
        })
        print("Total number of tensors:", len(self.tensors))

        return prediction_boxes, prediction_classes

    def get_model(self, filename):
        """ Get .tflite model from the FlatBuffer file """
        with open(filename, "rb") as f:
            buf = bytearray(f.read())

        model = tflite.Model.Model.GetRootAsModel(buf, 0)

        assert model.Version() == 3, \
            "Only support schema version 3 at the moment"
        assert model.MetadataBufferLength() == 0, \
            "Do not support metadata_buffer at the moment"

        return model

    def get_op(self, op):
        """ Right now return a string for the operator, later return a function
        that'll actually execute the operator """
        operator = None
        custom = op.CustomCode()
        builtin = op.BuiltinCode()

        if builtin == tflite.BuiltinOperator.BuiltinOperator.CONCATENATION:
            operator = Operation.CONCAT
        elif builtin == tflite.BuiltinOperator.BuiltinOperator.CONV_2D:
            operator = Operation.CONV2D
        elif builtin == tflite.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D:
            operator = Operation.DEPTHWISECONV2D
        elif builtin == tflite.BuiltinOperator.BuiltinOperator.LOGISTIC:
            operator = Operation.LOGISTIC
        elif builtin == tflite.BuiltinOperator.BuiltinOperator.RESHAPE:
            operator = Operation.RESHAPE
        elif builtin == tflite.BuiltinOperator.BuiltinOperator.CUSTOM:
            if custom.decode() == "TFLite_Detection_PostProcess":
                operator = Operation.POSTPROCESS
            else:
                raise NotImplementedError("custom op "+custom.decode()+" not implemented")
        else:
            raise NotImplementedError("builtin op "+str(builtin)+" not implemented")

        return operator

    def get_activation(self, act):
        """ Right now return a string for the activation function, later return a
        function that'll actually execute the activation function """
        activation = None

        if act == tflite.ActivationFunctionType.ActivationFunctionType.NONE:
            activation = Activation.NONE
        elif act == tflite.ActivationFunctionType.ActivationFunctionType.RELU6:
            activation = Activation.RELU6
        else:
            raise NotImplementedError("activation "+str(act)+" not implemented")

        return activation

    def get_padding(self, pad):
        """ Right now return a string for the padding name """
        padding = None

        if pad == tflite.Padding.Padding.SAME:
            padding = Padding.SAME
        elif pad == tflite.Padding.Padding.VALID:
            padding = Padding.VALID
        else:
            raise NotImplementedError("padding "+str(pad)+" not implemented")

        return padding

    def get_ops(self, model):
        """ Get all operators used in a model """
        ops = []
        op_codes_len = model.OperatorCodesLength()

        for i in range(op_codes_len):
            op = model.OperatorCodes(i)
            ops.append(self.get_op(op))

        return ops

    def conv2d_options(self, op):
        """ Get Conv2D options from BuiltinOptions union """
        conv2d_options = tflite.Conv2DOptions.Conv2DOptions()
        conv2d_options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        padding = self.get_padding(conv2d_options.Padding())
        stride = conv2d_options.StrideW()
        stride_h = conv2d_options.StrideH()
        activation = self.get_activation(conv2d_options.FusedActivationFunction())
        dilation_w_factor = conv2d_options.DilationWFactor()
        dilation_h_factor = conv2d_options.DilationHFactor()

        assert stride == stride_h, \
            "Only support stride_w == stride_h at the moment"
        assert dilation_w_factor == 1, \
            "Only support dilation_w_factor == 1 at the moment"
        assert dilation_h_factor == 1, \
            "Only support dilation_h_factor == 1 at the moment"

        return {"activation": activation, "padding": padding, "stride": stride}

    def depthwise_options(self, op):
        """ Get DepthwiseConv2D options from BuiltinOptions union """
        options = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
        options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        padding = self.get_padding(options.Padding())
        stride = options.StrideW()
        stride_h = options.StrideH()
        depth_multiplier = options.DepthMultiplier()
        activation = self.get_activation(options.FusedActivationFunction())
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

    def concat_options(self, op):
        """ Get Concatenation options from BuiltinOptions union """
        options = tflite.ConcatenationOptions.ConcatenationOptions()
        options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        axis = options.Axis()
        activation = options.FusedActivationFunction()

        assert activation == tflite.ActivationFunctionType.ActivationFunctionType.NONE, \
            "Only support activation == None at the moment for concat"

        return {"axis": axis}

    def reshape_options(self, op):
        """ Get Reshape options from BuiltinOptions union """
        options = tflite.ReshapeOptions.ReshapeOptions()
        options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        shape = options.NewShapeAsNumpy()

        return {"shape": shape}

    def get_options(self, op):
        """ Get options for specified op """
        options = {}
        options_type = op.BuiltinOptionsType()

        if options_type == 0:
            pass
        elif options_type == tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions:
            options = self.conv2d_options(op)
        elif options_type == tflite.BuiltinOptions.BuiltinOptions().DepthwiseConv2DOptions:
            options = self.depthwise_options(op)
        elif options_type == tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions:
            options = self.concat_options(op)
        elif options_type == tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions:
            options = self.reshape_options(op)
        else:
            raise NotImplementedError("options "+str(options_type)+" not implemented")

        # TODO also handle custom_options probably

        assert op.MutatingVariableInputsLength() == 0, \
            "Do not support mutating_variable_inputs at the moment"

        return options

    def get_type(self, t):
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

    def get_graph(self, model):
        """ Get the graph from the model """
        subgraph_len = model.SubgraphsLength()

        assert subgraph_len == 1, \
            "Only support subgraph_len == 1 at the moment"

        return model.Subgraphs(0)

    def get_bufs(self, model):
        """ Get all the buffers from the model """
        bufs = []
        bufs_len = model.BuffersLength()

        for i in range(bufs_len):
            buf = model.Buffers(i)
            bufs.append(buf.DataAsNumpy())

        return bufs

    def get_tensors(self, subgraph):
        """ Get all tensors in the subgraph """
        tensors = []
        tensors_len = subgraph.TensorsLength()

        for j in range(tensors_len):
            tensor = subgraph.Tensors(j)
            name = tensor.Name().decode()
            shape = tensor.ShapeAsNumpy()
            tensor_type = self.get_type(tensor.Type())
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

    def get_operators(self, graph):
        """ Get operators from graph """
        operators = []
        operators_len = graph.OperatorsLength()

        for i in range(operators_len):
            op = graph.Operators(i)
            op_index = op.OpcodeIndex()

            inputs = op.InputsAsNumpy()
            outputs = op.OutputsAsNumpy()
            options = self.get_options(op)

            operators.append({
                "op": op_index,
                "inputs": inputs,
                "outputs": outputs,
                "options": options
            })

        return operators

    def get_tensors_by_index(self, tensors, indices):
        """ Return a list of the desired tensors """
        return [tensors[t] for t in indices]

    def get_tensor_buffers(self, bufs, tensors):
        """ Return a list of buffers of specified by the given tensors """
        buffers = []

        for t in tensors:
            # Reinterpret bytes as correct type and reshape
            buf = bufs[self.replace_buffers(t["buffer"])]
            buf = np.frombuffer(buf, dtype=t["type"]).reshape(t["shape"])
            buffers.append(buf)

            # Save back the correct interpretation -- needed since the output
            # is done in OpenCL and when we copy back the results, we want it
            # to be copied into the correct data type
            bufs[t["buffer"]] = buf

        return buffers

    def calc_padding(self, input_size, filter_size, stride, pad_type):
        """
        See:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L20

        Official equations given on:
        https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
        https://www.tensorflow.org/api_guides/python/nn#Convolution
        """
        if pad_type == Padding.VALID:
            output_size = int((input_size - filter_size + stride) / stride)
            pad_before = 0
            pad_after = 0
        elif pad_type == Padding.SAME:
            output_size = int((input_size + stride - 1) / stride)
            pad_needed = max(0, (output_size - 1)*stride + filter_size - input_size)
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
        else:
            raise NotImplementedError("Only SAME and VALID padding types implemented")

        assert output_size >= 0, "output_size must be non-negative after padding"
        return output_size, pad_before, pad_after

if __name__ == "__main__":
    img = load_test_image("test_images")
    model = TFLiteOpenCL("detect_float.tflite")

    t = time.time()
    model.run(img)
    t = time.time() - t
    print("FPS", 1/t)
