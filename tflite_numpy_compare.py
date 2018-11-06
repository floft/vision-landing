#!/usr/bin/env python3
"""
Try to determine if I implemented it right (probably not)
"""
import numpy as np
np.set_printoptions(threshold=np.nan)

def load(filename):
    return np.load(filename).item()

def manual():
    return load("tflite_manual.npy")

def official():
    return load("tflite_official.npy")

if __name__ == "__main__":
    o = official()
    m = manual()
    same = lambda t: (m[t] == o[t]).all()

    for t in m.keys():
        print(t, np.mean(np.abs(m[t] - o[t])))
