import numpy as np
import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd


def pentagonal_num(n):
    return n*(3*n - 1)/2

def triangular_num(n):
    return n*(n+1)/2