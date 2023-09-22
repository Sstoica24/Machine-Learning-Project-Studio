def slow_dot_prod(double[:] X, double[:] Y):
    cdef double dot_prod
    cdef Py_ssize_t i
    for i in range(len(X)):
        dot_prod += X[i] * Y[i]
    return dot_prod


# def slow_matrix_mult(double[:,:] X, double[:, :] Y):
