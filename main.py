# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 08:33:40 2022

@author: jrmfi
"""

''' Resultado dos testes '''

Legion5i = {
            'configuracoes':{
                                'placa_de_rede': 'Intel(R) Wi-Fi 6 AX201 160MHz',
                                'processador': 'Intel(R) Core(TM) i7-10750H CPU @2.60GHz 2.59 GHz',
                                'nucleos':6,
                                'memoria_ram': '16,0GB DDR4 até 2933 MHz',
                                'memoria': '512GB SSD M.2 PCIe NVMe',
                                'gpu_0': 'Intel(R) UHD Graphics',
                                'gpu_1': 'Placa dedicada NVIDIA® GeForce® RTX 2060 6GB GDDR6',
                            },
            'teste_placa_de_rede': {
                                    'descricao': 'teste velocidade internet no navegador',
                                    'resultado':{'download': '74.4 Mbs' ,'upload':'31.2 Mbs'},
                }
    }


DellG15 = {
            'configuracoes':{
                                'placa_de_rede': 'Intel(R)  Wi-Fi 6 AX1650 160MHz',
                                'processador': '12ª geração Intel® Core™ i7-12700H 4.7GHz',
                                'nucleos':14,
                                'memoria_ram': '16,0GB DDR5 4800 MHz',
                                'memoria': 'SSD de 512GB PCIe NVMe M.2',
                                'gpu_0': 'Gráficos Intel® Iris® Xe elegíveis',
                                'gpu_1': 'NVIDIA® GeForce® RTX™ 3060, 6GB GDDR6',
                            },
            'teste_placa_de_rede': {
                                    'descricao': 'teste velocidade internet no navegador',
                                    'resultado':{'download': '47.9 Mbs' ,'upload':'32.5 Mbs'},
                }
    }

import math
import threading
from timeit import repeat

import numpy as np
from numba import jit

nthreads = 4
size = 10**6

def func_np(a, b):
    """
    Control function using Numpy.
    """
    return np.exp(2.1 * a + 3.2 * b)

@jit('void(double[:], double[:], double[:])', nopython=True,
     nogil=True)
def inner_func_nb(result, a, b):
    """
    Function under test.
    """
    for i in range(len(result)):
        result[i] = math.exp(2.1 * a[i] + 3.2 * b[i])

def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before the benchmark is
    # started
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(
        lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000))
    return res

def make_singlethread(inner_func):
    """
    Run the given function inside a single thread.
    """
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func

def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in
                   args] for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt

func_nb = make_singlethread(inner_func_nb)
func_nb_mt = make_multithread(inner_func_nb, nthreads)

a = np.random.rand(size)
b = np.random.rand(size)

correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
timefunc(correct, "numba (1 thread)", func_nb, a, b)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, a, b)