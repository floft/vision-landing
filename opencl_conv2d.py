#!/usr/bin/env python3
"""
Try a simple convolution
"""
import numpy as np
import pyopencl as cl
import tensorflow as tf
from enum import Enum
tf.enable_eager_execution()

Padding = Enum("Padding", "VALID SAME")

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

def conv2d_mine(x, W, b, stride, pad, out_type):
    (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Calculate padding
    n_H, pad_before_h, pad_after_h = calc_padding(n_H_prev, f, stride, pad)
    n_W, pad_before_w, pad_after_w = calc_padding(n_W_prev, f, stride, pad)

    # Init output of correct shape
    output = np.empty((m,n_H,n_W,n_C), dtype=out_type)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, """
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
    """).build()

    mf = cl.mem_flags
    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x))
    w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(W))
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(b))
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg.conv2d(queue, (n_H,n_W,n_C), None,
        np.int32(m), np.int32(n_H), np.int32(n_W), np.int32(n_C),
        np.int32(stride), np.int32(f),
        np.int32(pad_before_w), np.int32(pad_before_h),
        np.int32(pad_after_w), np.int32(pad_after_h),
        np.int32(n_H_prev), np.int32(n_W_prev), np.int32(n_C_prev),
        x_buf, w_buf, b_buf, out_buf)
    cl.enqueue_copy(queue, output, out_buf)

    return output

if __name__ == "__main__":
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
    print("TF")
    print(result_tf)
    print("Mine")
    print(result)
    assert (result == np.array([
        18, 2, 5, 18, 2, 5, 18, 2, 5,
        17, 4, 3, 27, 4, 3, 37, 4, 3]).reshape((2, 1, 3, 3))).all(), \
        "Test 1 gives "+str(result)
