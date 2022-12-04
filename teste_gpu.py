# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:34:04 2022

@author: jrmfi
"""

import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)


print(cuda.gpus)
# cuda.detect()

# Example 1.1: Add scalars
@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b

dev_c = cuda.device_array((1,), np.float32)

add_scalars[1, 1](2.0, 7.0, dev_c)

c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")



















# Brute force solution
# import datetime
# start_time = datetime.datetime.now()
# [(a, b) for a in (1, 3, 5) for b in (2, 4, 6)] # example snippet
# end_time = datetime.datetime.now()
# print(end_time - start_time)
# # timeit solution
# import timeit
# min(timeit.repeat("[(a, b) for a in (1, 3, 5) for b in (2, 4, 6)]"))
# # cProfile solution
# import cProfile
# cProfile.run("[(a, b) for a in (1, 3, 5) for b in (2, 4, 6)]")

# from numba import cuda

# obj = cuda.cudadrv.devices._DeviceList

# print(cuda.cudadrv.devices.gpus)


# for s in dir(obj):
#     print(s)
#     print(getattr(obj, s))
    
# import cupy
# from numba import cuda

# @cuda.jit
# def add(x, y, out):
#     start = cuda.grid(1)
#     stride = cuda.gridsize(1)
#     for i in range(start, x.shape[0], stride):
#         out[i] = x[i] + y[i]

# a = cupy.arange(10)
# b = a * 2
# out = cupy.zeros_like(a)

# add[1, 32](a, b, out)