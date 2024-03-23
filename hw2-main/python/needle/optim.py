"""Optimization module"""
import needle as ndl
import numpy as np
import pdb


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            #p.grad = None
            if hasattr(p, 'grad'):
                p.grad = p.grad * 0.
            else:
                p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        # populate self.u
        for w in self.params:
            self.u[w] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * (w.grad.data + self.weight_decay * w.data)
            # expected output is of type float32 and self.u[w].data is of type float64
            w.data = w.data - self.lr * self.u[w].data.numpy().astype(np.float32)

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.params = params

        self.m = {}
        self.v = {}

        # populate the dictionaries
        for w in self.params:
            self.m[w] = 0
            self.v[w] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            grad = w.grad.data + self.weight_decay * w.data
            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)
            unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * (unbiased_m.numpy().astype(np.float32) / (unbiased_v.numpy().astype(np.float32)**0.5 + self.eps))