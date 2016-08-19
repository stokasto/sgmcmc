import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .utils import sharedX

class SGLDSampler(object):

    def __init__(self, rng=None, precondition=False):
        if rng:
            self._srng = rng
        else:
            self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.precondition = precondition
        self.prepared = False

    def prepare_updates(self, cost, params, epsilon, A=1., inputs=[], scale_grad=1., **kwargs):
        self.updates = []
        grads = T.grad(cost, params)
        self.params = params
        self.cost = cost
        self.count = sharedX(0)
        self.epsilon = sharedX(np.float32(epsilon))
        self.A = T.cast(A, theano.config.floatX)
        self.inputs = inputs
        for theta,grad in zip(params, grads):
            xi = sharedX(theta.get_value() * 0. + 1)
            g = sharedX(theta.get_value() * 0. + 1)
            g2 = sharedX(theta.get_value() * 0. + 1)
            r_t = 1. / (xi + 1.)
            if self.precondition:
                g_t = (1. - r_t) * g + r_t * grad
                g2_t = (1. - r_t) * g2 + r_t * grad**2
                xi_t = 1 + xi * (1 - g * g / (g2 + 1e-16))
                Minv = 1. / (T.sqrt(g2 + 1e-16) + 1e-16)
                self.updates.append((g, g_t))
                self.updates.append((g2, g2_t))
                self.updates.append((xi, xi_t))
                noise = 0.
            else:
                Minv = 1.
                noise = 0.
            sigma = T.sqrt(2. * self.epsilon * (Minv * (self.A - noise))) / T.cast(scale_grad, dtype=theano.config.floatX)
            sample_t = self._srng.normal(size=theta.shape) * sigma
            theta_t = theta - self.epsilon * Minv * self.A * grad + sample_t 
            self.updates.append((theta, theta_t))
        self.prepared = True
        return self.updates

    def step(self, *inp):
        if not self.prepared:
            raise RuntimeError("You called step() without a prior call to prepare_updates()")
        if not hasattr(self, "step_fun"):
            #Tinput = [T.matrix() for i in range(len(inp))]
            print("... compiling theano function")
            self.step_fun = theano.function(self.inputs, self.cost, updates=self.updates)
        nll = self.step_fun(*inp)
        return self.params, nll
