import autograd
import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.util import flatten_func
from autograd import elementwise_grad

class SGMCMCSampler(object):

    def prepare_pdates(self, cost, params, epsilon, **kwargs):
        raise NotImplementedError("Not implemented in base class")

    def step(self, *inputs):
        raise NotImplementedError("Not implemented in base class")


class SGNHTSampler(SGMCMCSampler):

    def __init__(self, rng=None, resample_momentum=0):
        if rng:
            self._srng = rng
        else:
            self._srng = npr
        self.resample_momentum = resample_momentum

    def prepare_updates(self, cost, params, epsilon,
                        grad=None, A=1, callbacks=[],
                        callback_every=1000, **kwargs):
        self.theta = params
        if grad is not None:            
            self.flattened_grad = grad            
            self.unflatten = lambda x: x
        else:
            gradient = autograd.grad(cost)
            self.flattened_grad, self.unflatten, self.theta = flatten_func(gradient, params)
            self.hess = autograd.grad(self.flattened_grad)
            self.flattened_hess = lambda x, *inputs: np.diag(self.hess(x, *inputs)).reshape((-1,))
            
        self.epsilon = epsilon
        self.A = A

        self.p   = self._srng.normal(size=params.shape)
        self.xi  = np.ones_like(params) * self.A
        self.xi_acc  = np.ones_like(params) * self.A
        self.updates = np.zeros_like(params)
        self.count = 1
        self.callback_every = callback_every
        self.callbacks = callbacks
        return self.updates
        
    def step(self, *inputs):
        grad = self.flattened_grad(self.theta, *inputs)

        # optionally resample momentum 
        if self.resample_momentum > 0 and self.count % self.resample_momentum == 0:
            np.copyto(self.p, self._srng.normal(size=self.theta.shape))

        # Constant mass just defined here so that we can easily change it should we want to
        Minv  = 1.
        Minvh = 1.
        # pre-generate a sample
        sample = self._srng.normal(size=self.theta.shape) * np.sqrt(self.epsilon * 2 * self.A)
        # the SG-HMC update equations
        # update p
        self.p += - self.epsilon * Minvh * grad \
                  - self.epsilon * (self.xi - self.A) * self.p \
                  - self.epsilon * Minv * self.A * self.p  \
                  + Minvh * sample
        # in-place multiplication with epsilon to make sure
        # we have the values available in updates
        np.multiply(Minvh * self.p, self.epsilon, self.updates)
        # update theta
        self.theta += self.updates
        # update xi
        self.xi += self.epsilon * (self.p**2  - 1)
        self.xi_acc += self.xi

        # callbacks
        self.count += 1
        if self.count % self.callback_every == 0:
            #print(self.theta, (self.epsilon * r_t - self.epsilon * r_t**2))
            for callback in self.callbacks:
                callback(self.count, self)
        return self.unflatten(self.theta)


def G(theta, g, g2, r_t, flattened_grad, *inputs):
    grad = flattened_grad(theta, *inputs)
    # add additional noise for debugging
    #grad += npr.normal(size=grad.shape) * 4
    g = (1 - r_t) * g + r_t * grad
    g2 = (1 - r_t) * g2 + r_t * grad**2
    Minv = 1. / (np.sqrt(g2 + 1e-16) + 1e-16)
    noise =  g*g / (g2 + 1e-16)
    return Minv, (grad, Minv, noise)
    
class SGLDSampler(SGMCMCSampler):

    def __init__(self, rng=None, precondition=False, noise_correction=False, resample_momentum=0):
        if rng:
            self._srng = rng
        else:
            self._srng = npr
        self.precondition = precondition
        self.resample_momentum = resample_momentum
        self.noise_correction = noise_correction
        
        
    def prepare_updates(self, cost, params, epsilon,
                        grad=None, diag_hess=None, fd_hess=False, A=1,
                        callbacks=[], callback_every=1000, **kwargs):
        self.theta = params
        if grad is not None:
            if diag_hess is None:
                if self.precondition and not fd_hess:
                    raise ValueError("If precondition=True you must also prepare a function for" \
                                     " computing the diagonal of the Hessian! Alternatively specify fd_hess=True" \
                                     " in which case a noisy finite difference approximation will be used" \
                                     " note that this can bias the MCMC sampler!")
                else:
                    self.flattened_hess = None
            else:
                self.flattened_hess = diag_hess
            self.flattened_grad = grad            
            self.unflatten = lambda x: x
        else:
            gradient = autograd.grad(cost)
            self.flattened_grad, self.unflatten, self.theta = flatten_func(gradient, params)
            self.hess = autograd.grad(self.flattened_grad)
            self.flattened_hess = lambda x, *inputs: np.diag(self.hess(x, *inputs)).reshape((-1,))

        self.epsilon = epsilon
        self.A = A

        self.g  = np.ones_like(params)
        self.g2  = np.ones_like(params)

        # note that xi here is not the same as in the thermostat!
        self.xi  = np.ones_like(params) * self.A
        self.xi_acc  = np.ones_like(params) * self.A

        self.updates = np.zeros_like(params)
        self.count = 1
        self.callback_every = callback_every
        self.callbacks = callbacks
            
        def Ggrad(*args, **kwargs):
            saved = lambda: None
            def return_val_save_aux(*args, **kwargs):
                val, saved.aux = G(*args, **kwargs)
                return val
            gradval = elementwise_grad(return_val_save_aux, 0)(*args, **kwargs)
            return gradval, saved.aux
        self.Ggrad = Ggrad

        return self.updates
    
    def step(self, *inputs):
        r_t = 1. / (self.xi + 1.)
        if self.precondition and self.noise_correction:
            # this part adds a correction term based on our estimate for the preconditioning matrix
            # this is required theoretically, but in practice xi goes to 0 fairly fast and the effect of the correction term is marginal
            gMinv, (grad, Minv, noise) = self.Ggrad(self.theta, self.g, self.g2, r_t, self.flattened_grad, *inputs)
            self.g +=  - r_t * self.g + r_t * grad
            self.g2 += - r_t * self.g2 + r_t * grad**2
            self.xi = 1 + self.xi * (1 - self.g * self.g / (self.g2 + 1e-16))
            self.xi_acc += self.xi
        else:
            grad  = self.flattened_grad(self.theta, *inputs)
            #print(grad)
            Minv  = 1.
            gMinv = 0.
            noise = 0
            # add additional noise for debugging
            #grad += self._srng.normal(size=self.theta.shape) * 0.
            self.g +=  - r_t * self.g + r_t * grad
            self.g2 += - r_t * self.g2 + r_t * grad**2
            self.xi = 1 + self.xi * (1 - self.g * self.g / (self.g2 + 1e-16))
            self.xi_acc += self.xi
            if self.precondition:
                Minv = 1. / (np.sqrt(self.g2 + 1e-16) + 1e-16)
                noise = Minv**2 * 0.5 * self.epsilon
        # pre-generate a sample
        sample = self._srng.normal(size=self.theta.shape) * np.sqrt(self.epsilon * 2 * (self.A - noise) * Minv)
        if self.resample_momentum > 0 and self.count % self.resample_momentum == 0:
            # optionally resample momentum 
            np.copyto(self.updates, self._srng.normal(size=self.theta.shape))
        else:
            # compute updates
            self.updates = - self.epsilon * self.A * Minv * grad + sample + self.epsilon * gMinv
        # and apply 
        self.theta += self.updates

        # callbacks
        self.count += 1
        if self.count % self.callback_every == 0:
            #print(self.theta, r_t)
            for callback in self.callbacks:
                callback(self.count, self)
        return self.unflatten(self.theta)
    
