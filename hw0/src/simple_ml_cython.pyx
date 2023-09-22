#cimport is the cython way to import. 
#allows you to import cython module 
#not regular python module

#NO NUMPY ALLOWED 
from libc.math cimport exp, log 
cimport numpy as np 

""" Notes on Cython:
    1) initialize all variables with cdef. 
    
    2) Initialize ALL arrays using a memoryview which
       acts like a buffer, and results in no conversion
       between Python and C types to happen, resuling in 
       much faster code
        ex) cdef int[:,:] arr = np.array([[1,2,3,4],[2,3,4,5]]) 
            (Note that the number of ":" is equal to the dimension
            of the array. In this case, arr is 2D ==> we have 2
            ":".
    
    3) use Py_ssize_t for indexing!

    4)

"""

def softmax_regression_epoch_cpp(double[:, :] X, int[:] y, double[:, :] theta, double lr=0.1, int batch=100):
    """Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).

        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)

        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)

        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    cdef Py_ssize_t num_examples, input_dim = X.shape
    cdef Py_ssize_t num_classes = theta.shape[1]
    
    #initialize logits and grads along with their buffers
    cdef Py_ssize_t i, j, k, f, r, c
    cdef double[:] softmax_probs
    cdef double[:,:] subtraction
    cdef int[:] one_hot_encoding 
    cdef int batch_size
    cdef double[:, :] X_batch
    cdef int[:] y_batch
    cdef double[:] x_batch_row_view
    cdef int y_batch_class
    cdef double[:] row_logits_view
    cdef int[:] one_hot_encoding_view
    cdef double sum_prods
    cdef double[:] exp_row_logits_view
    cdef double exp_sum
    cdef double[:] x_batch_row_view_T_view
    cdef Py_ssize_t dim_r
    cdef Py_ssize_t dim_c
    cdef double[:,:] dot_prod_view 
    cdef double[:, :] grads
    
    for i in range(0,num_examples, batch):
        #consider the batches of X and y
        #we need to make sure that we do 
        #not over index like we usually do in python by doing
        #X[i:i+batch] and y[i:i+batch] because it will slow down 
        #our cython implementation.
        batch_size = min(batch, num_examples - i)
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        #compute the logits, one_hot_encoding and softmax_probs for a single row
        for j in range(len(X_batch)):
            x_batch_row_view = X_batch[j] 
            y_batch_class = y_batch[j]

            row_logits_view = np.zeros((1,num_classes))[0] #[0] to make 1D matrix not 2D
            one_hot_encoding_view = np.zeros((1,num_classes))[0] #[0] to make 1D matrix not 2D 
            
            for k in range(num_classes):
                sum_prods = 0.0
                for f in range(input_dim):
                    sum_probs += x_batch_row_view[f] * theta[f,k]
                row_logits_view[k] = sum_prods

                #calculate the one_hot_encoding
                if k == y_batch_class:
                    one_hot_encoding_view [k] = 1
            
            #calculate the softmax
            exp_row_logits_view = [exp(logit) for logit in row_logits_view]
            exp_sum = sum(exp_row_logits_view)
            softmax_probs = np.array([exp_logit/exp_sum for exp_logit in exp_row_logits_view])

            #calculate the difference and the transpose of x_batch_row
            subtraction = (softmax_probs - one_hot_encoding).reshape(1,-1)
            x_batch_row_view_T_view = x_batch_row_view.T.reshape(-1,1)

            #calculate the gradient
            dim_r = x_batch_row_view_T_view.shape[0]
            dim_c =  subtraction.shape[1]
            dot_prod_view = np.zeros((dim_r, dim_c))
            for r in range(dim_r):
                for c in range(dim_c):
                    dot_prod_view[r,c] = x_batch_row_view_T_view[r,0] * subtraction[0,c]
            
            # grads = 1/batch * dot_prod_view
            # theta = theta - lr*grads
