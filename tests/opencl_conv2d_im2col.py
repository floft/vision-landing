#!/usr/bin/env python3
"""
Try a convolution with im2col to see if it really improves speed like people
say it should

Good explanations:
https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
"""
import time
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

def im2col(x,hh,ww,stride):
    """
    From: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
    Args:
        x: image matrix to be translated into columns, (C,H,W)
        hh: filter height
        ww: filter width
        stride: stride
    Returns:
        col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
    From: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
    Args:
        mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
        h_prime: reshaped filter height
        w_prime: reshaped filter width
        C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
        if C == 0: (F,h_prime,w_prime) matrix
        Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

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

    platforms = cl.get_platforms()
    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])

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

    t = time.time()
    prg.conv2d(queue, (n_H,n_W,n_C), None,
        np.int32(m), np.int32(n_H), np.int32(n_W), np.int32(n_C),
        np.int32(stride), np.int32(f),
        np.int32(pad_before_w), np.int32(pad_before_h),
        np.int32(pad_after_w), np.int32(pad_after_h),
        np.int32(n_H_prev), np.int32(n_W_prev), np.int32(n_C_prev),
        x_buf, w_buf, b_buf, out_buf)
    cl.enqueue_copy(queue, output, out_buf)
    t = time.time() - t
    print("compute time", t)

    return output

def conv2d_im2col_cpu(x, W, b, stride, pad, out_type):
    """
    From: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Calculate padding
    n_H, pad_before_h, pad_after_h = calc_padding(n_H_prev, f, stride, pad)
    n_W, pad_before_w, pad_after_w = calc_padding(n_W_prev, f, stride, pad)

    # before: f,f,n_C_prev,n_C = w.shape
    # after: F,C,HH,WW = w.shape
    #        n_C,n_C_prev,f,f = w.shape
    W = W.transpose(3,2,0,1)
    # before: N,H,W,C = x.shape
    # after:  N,C,H,W = x.shape
    x = x.transpose(0,3,1,2)

    # Init output of correct shape
    H_prime = (n_H_prev+pad_before_h+pad_after_h-f) // stride + 1
    W_prime = (n_W_prev+pad_before_w+pad_after_w-f) // stride + 1
    out = np.zeros([m,n_C,H_prime,W_prime], dtype=out_type)

    for im_num in range(m):
        im = x[im_num,:,:,:]
        im_pad = np.pad(im, ((0,0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w)),'constant')
        im_col = im2col(im_pad,f,f,stride)
        filter_col = np.reshape(W,(n_C,-1))
        mul = im_col.dot(filter_col.T) + b
        out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,1)

    # before: N,C,H,W = out.shape
    # after:  N,H,W,C = out.shape
    out = out.transpose(0,2,3,1)

    return out

def cl_im2col(x, W, b, stride, pad, out_type):
    (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
    (f, f, n_C_prev, n_C) = W.shape
    assert m == 1, "only implemented im2col for batch of 1 image"

    # Calculate padding
    n_H, pad_before_h, pad_after_h = calc_padding(n_H_prev, f, stride, pad)
    n_W, pad_before_w, pad_after_w = calc_padding(n_W_prev, f, stride, pad)

    # Init output of correct shape
    out_im2col = np.empty((m,n_H*n_W,f*f*n_C_prev), dtype=out_type)
    out_conv2d = np.empty((m,n_H,n_W,n_C), dtype=out_type)

    platforms = cl.get_platforms()
    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, """
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
                    float value = 0;

                    if (in_x >= 0 && in_y >= 0 && in_x < n_W_prev && in_y < n_H_prev) {
                        value = x[in_y*xo2 + in_x*xo3 + c];
                    }

                    output[out_y*oo1 + out_x*oo2 + filter_y*oo3 + filter_x*oo4 + c] = value;
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
    """).build()

    mf = cl.mem_flags
    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x))
    w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(W))
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(b))
    out_im2col_buf = cl.Buffer(ctx, mf.READ_WRITE, out_im2col.nbytes)
    out_conv2d_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_conv2d.nbytes)

    t = time.time()
    prg.im2col(queue, (n_H,n_W), None,
        np.int32(m), np.int32(n_H), np.int32(n_W), np.int32(n_C),
        np.int32(stride), np.int32(f),
        np.int32(pad_before_w), np.int32(pad_before_h),
        np.int32(pad_after_w), np.int32(pad_after_h),
        np.int32(n_H_prev), np.int32(n_W_prev), np.int32(n_C_prev),
        x_buf, out_im2col_buf)
    #cl.enqueue_barrier(queue)
    cl.enqueue_copy(queue, out_im2col, out_im2col_buf) # For testing
    M = n_H*n_W
    K = f*f*n_C_prev
    N = n_C
    prg.matmul(queue, (M, N), None,
        np.int32(M), np.int32(N), np.int32(K),
        out_im2col_buf, w_buf, b_buf, out_conv2d_buf)
    cl.enqueue_copy(queue, out_conv2d, out_conv2d_buf)
    t = time.time() - t
    print("compute time", t)

    return out_im2col, out_conv2d

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

    t_tf = time.time()
    result_tf = conv2d_tf(data, weights, bias, stride, Padding.VALID, np.float32)
    t_tf = time.time() - t_tf
    print("TF", t_tf)
    print(result_tf)

    t_cl = time.time()
    result_cl = conv2d_mine(data, weights, bias, stride, Padding.VALID, np.float32)
    t_cl = time.time() - t_cl
    print("Mine", t_cl)
    print(result_cl)

    assert (result_cl == np.array([
        18, 2, 5, 18, 2, 5, 18, 2, 5,
        17, 4, 3, 27, 4, 3, 37, 4, 3]).reshape((2, 1, 3, 3))).all(), \
        "Test 1 gives "+str(result_cl)

    t_im2col_cpu = time.time()
    result_im2colcpu = conv2d_im2col_cpu(data, weights, bias, stride, Padding.VALID, np.float32)
    t_im2col_cpu = time.time() - t_im2col_cpu
    print("im2col cpu", t_im2col_cpu)
    print(result_im2colcpu)

    assert (result_im2colcpu == np.array([
        18, 2, 5, 18, 2, 5, 18, 2, 5,
        17, 4, 3, 27, 4, 3, 37, 4, 3]).reshape((2, 1, 3, 3))).all(), \
        "Test 1 gives "+str(result_im2colcpu)

    # t_im2col_gpu = time.time()
    # result_im2colgpu = cl_im2col(data, weights, bias, stride, Padding.VALID, np.float32)
    # t_im2col_gpu = time.time() - t_im2col_gpu
    # print("im2col gpu", t_im2col_gpu)
    # print(result_im2colgpu)

    # assert (result_im2colgpu == np.array([
    #     18, 2, 5, 18, 2, 5, 18, 2, 5,
    #     17, 4, 3, 27, 4, 3, 37, 4, 3]).reshape((2, 1, 3, 3))).all(), \
    #     "Test 1 gives "+str(result_im2colgpu)

    # Single output channel
    data = np.array([
        1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8
    ]).reshape((1,2,4,2)).astype(np.float32)
    weights = np.array([
        1, -1, 2, -2, 3, -3, 4, -4
    ]).reshape((2,2,2,1)).astype(np.float32)
    bias = np.array([0]).astype(np.float32)
    stride = 1

    t_im2col_gpu = time.time()
    result_im2colgpu = cl_im2col(data, weights, bias, stride, Padding.SAME, np.float32)
    t_im2col_gpu = time.time() - t_im2col_gpu
    print("im2col gpu test 1", t_im2col_gpu)
    assert (result_im2colgpu[1] == np.array([
         88, 108, 128, 56, 34, 40, 46, 16]).reshape((1, 2, 4, 1))).all(), \
         "im2col gpu test 1 gives "+str(result_im2colgpu)

    # Two output channels
    data = np.array([
        1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8
    ]).reshape((1,2,4,2)).astype(np.float32)
    weights = np.array([
        1, -1, -1, 1, 2, -2, -2, 2, 3, -3, -3, 3, 4, -4, -4, 4
    ]).reshape((2,2,2,2)).astype(np.float32)
    bias = np.array([0,1]).astype(np.float32)
    stride = 1

    t_im2col_gpu = time.time()
    result_im2colgpu = cl_im2col(data, weights, bias, stride, Padding.SAME, np.float32)
    t_im2col_gpu = time.time() - t_im2col_gpu
    print("im2col gpu test 2", t_im2col_gpu)
    assert (result_im2colgpu[1] == np.array([
         88, -88+1, 108, -108+1, 128, -128+1, 56, -56+1,
         34, -34+1, 40, -40+1, 46, -46+1, 16, -16+1]).reshape((1, 2, 4, 2))).all(), \
         "im2col gpu test 2 gives "+str(result_im2colgpu)
