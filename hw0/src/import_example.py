import numpy as np
from time import perf_counter

from cython_example import slow_dot_prod

X,Y = np.random.randn(10 ** 6), np.random.randn(10 ** 6)
# timeit.timeit("slow_dot_prod(X,Y)", number= 10)
# timeit.timeit("X @ Y", number= 10)
start = perf_counter()
print("Our code:")
print(f"Computed {slow_dot_prod(X,Y)} in {perf_counter() - start} seconds")

start = perf_counter()
print("Dumb python:")
print(f"Computed {sum([x * y for x,y in zip(X,Y)])} in {perf_counter() - start} seconds")

start = perf_counter()
print("Numpy:")
print(f"Computed {X @ Y} in {perf_counter() - start} seconds")

"""
You need to do cythonize -i [insert name].pyx 
this will create a file with .so which will help
python know to talk to the c file cython_example.c
If you want an html file rather than a c file, then instead
do cythonize -a -i cython_example.pyx"""