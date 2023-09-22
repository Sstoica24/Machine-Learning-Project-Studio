import numpy as np
import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import needle as ndl

# x = np.array([[[1.95]], [[2.7]], [[3.75]]])
# print(x.transpose(2,0,1))
# #print(x.transpose((1,2)))
# print(x)
x =  (0,)
if 0 in x:
    print("0 is in the tuple")
else:
    print("0 is not in the tuple")

