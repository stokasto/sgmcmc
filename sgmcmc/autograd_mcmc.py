import autograd
import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.util import flatten_func

def hess_nll(theta, i):
    h = (theta+1)*(theta-1)/14 + (theta+1)*(theta-3)/14 + (theta-1)*(theta-3)/14 \
        + (theta+4)*(theta-1)/14 + (theta+4)*(theta-3)/14 + (theta-1)*(theta-3)/14 \
        + (theta+4)*(theta+1)/14 + (theta+4)*(theta-3)/14 + (theta+1)*(theta-3)/14 \
        + (theta+4)*(theta+1)/14 + (theta+4)*(theta-1)/14 + (theta+1)*(theta-1)/14
    return h


class SGMCMCSampler(object):

    def __init__(self, rng=None, exp_weight=1., precondition=False, resample_momentum=0):
        if rng:
            self._srng = rng
        else:
            self._srng = npr
        self.exp_weight = exp_weight
        self.precondition = precondition
        self.resample_momentum = resample_momentum

    def prepare_updates(self, cost, params, epsilon,
                        grad=None, diag_hess=None, fd_hess=False, A=1,
                        callbacks=[], callback_every=1000):
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

        self.g  = np.zeros_like(params)
        self.g2  = np.zeros_like(params)
        self.last_g2  = np.ones_like(params)
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
        # accumulate squared gradient
        r_t = 1. / (abs(self.xi_acc / self.count) + self.exp_weight)
        #r_t = 1. / (self.xi**2 + self.exp_weight)
        self.g2 = 1e-5 + grad**2
        #self.g2 += - self.epsilon * r_t * self.g2 + self.epsilon * r_t * grad**2 #+ self.epsilon * grad**2 * r_t / (self.xi)**2
        #self.g2 += - self.epsilon * r_t * self.g2 + ((self.epsilon - self.epsilon * r_t**2) * r_t) * grad**2
        #self.g2 += - r_t * self.g2 +  r_t * grad**2
        #print(self.g2)
        if self.precondition:
            Minvh = 1. / (np.sqrt(self.g2 + 1e-16) + 1e-16)
            Minv  = 1. / (self.g2 + 1e-16)
            #Minvh = Minv
            if self.flattened_hess is not None:
                Hess  = 1. / np.square(self.flattened_hess(self.theta, *inputs))
            else:
                Minvh_last = 1. / (np.sqrt(self.last_g2 + 1e-16) + 1e-16)
                Hess = (1. - Minvh_last / (1e-16 + Minvh)) / (1e-16 + self.theta)
        else:
            Minv  = 1.
            Minvh = 1.
            Hess  = 0.
        self.last_g2[:] = self.g2[:]
        # pre-generate a sample
        sample = self._srng.normal(size=self.theta.shape) * np.sqrt(self.epsilon * 2 * self.A)
        # the SG-HMC update equations
        # update p
        self.p += - self.epsilon * Minvh * grad \
                  - self.epsilon * (self.xi - self.A) * self.p \
                  - self.epsilon * Minv * self.A * self.p  \
                  + Minvh * sample \
                  + self.epsilon * r_t * self.A * Hess 
        #print(self.epsilon * Hess, self.p)
        # in-place multiplication with epsilon to make sure
        # we have the values available in updates
        np.multiply(Minvh * self.p, self.epsilon, self.updates)
        # update theta
        self.theta += self.updates
        # update xi
        self.xi += self.epsilon * (self.p**2  - 1) #+ self.epsilon * r_t * self.g2 * grad**2 * 1./(abs(self.xi) + 1e-4)
        self.xi_acc += self.xi

        # callbacks
        self.count += 1
        if self.count % self.callback_every == 0:
            print(self.theta, self.g2, (self.epsilon * r_t - self.epsilon * r_t**2))
            for callback in self.callbacks:
                callback(self.count, self)
        return self.unflatten(self.theta)
            
        
class SGRHMCSampler(SGMCMCSampler):

    def step(self, *inputs):
        grad = self.flattened_grad(self.theta, *inputs)

        # optionally resample momentum 
        if self.resample_momentum > 0 and self.count % self.resample_momentum == 0:
            np.copyto(self.p, self._srng.normal(size=self.theta.shape))
        # accumulate squared gradient
        #r_t = 1. / (abs(self.xi_acc / self.count) + self.exp_weight)
        #r_t = 1. / (self.xi**2 + self.exp_weight)
        self.g2 = (0.01 + grad**2 * 0.01 )
        #self.g2 += - self.epsilon * r_t * self.g2 + self.epsilon * r_t * grad**2 #+ self.epsilon * grad**2 * r_t / (self.xi)**2
        #self.g2 += - self.epsilon * r_t * self.g2 + ((self.epsilon - self.epsilon * r_t**2) * r_t) * grad**2
        #self.g2 += - r_t * self.g2 +  r_t * grad**2
        #print(self.g2)
        if self.precondition:
            Minvh = 1. / (np.sqrt(self.g2 + 1e-16) + 1e-16)
            Minv  = 1. / (self.g2 + 1e-16)
            #Minvh = Minv
            #1. / (x)**2 * - 1/( x**2) * x*2
            Hess =  -Minv * self.g2 * hess_nll(self.theta, 0)
            """
            if self.flattened_hess is not None:
                Hess  = 1. / np.square(self.flattened_hess(self.theta, *inputs))
            else:
                Minvh_last = 1. / (np.sqrt(self.last_g2 + 1e-16) + 1e-16)
                Hess = (1. - Minvh_last / (1e-2 + Minvh)) / (1e-2 + self.theta)
            """
        else:
            Minv  = 1.
            Minvh = 1.
            Hess  = 0.
        self.last_g2[:] = self.g2[:]
        # pre-generate a sample
        sample = self._srng.normal(size=self.theta.shape) * np.sqrt(self.epsilon * 2 * Minv )#- self.epsilon * self.g2)
        #print(self.epsilon * 2 * Minv - self.epsilon * self.g2)
        # the SG-HMC update equations
        # update p        
        self.p += - self.epsilon * Minvh * grad \
                  - self.epsilon * Minv * self.p  \
                  + sample  \
                  #+ self.epsilon * Hess 
        #print(self.epsilon * Hess, self.p)
        # in-place multiplication with epsilon to make sure
        # we have the values available in updates
        np.multiply(Minvh * self.p, self.epsilon, self.updates)
        # update theta
        self.theta += self.updates

        # callbacks
        self.count += 1
        if self.count % self.callback_every == 0:
            print(self.theta, self.g2)
            for callback in self.callbacks:
                callback(self.count, self)
        return self.unflatten(self.theta)
    
class SGLDSampler(SGMCMCSampler):    

    def step(self, *inputs):
        grad = self.flattened_grad(self.theta, *inputs)
        # add additional noise for debugging
        #grad += self._srng.normal(size=self.theta.shape) * 1

        r_t = 1. / (self.xi + 1.)
        #self.xi += self.epsilon * (grad**2) #+ self._srng.normal(size=self.theta.shape) * np.sqrt(2 * self.updates**2)
        self.g +=  - r_t * self.g + r_t * grad
        self.g2 += - r_t * self.g2 + r_t * grad**2
        self.xi = 1 + self.xi * (1 - self.g * self.g / (self.g2 + 1e-16))
        
        if self.precondition:
            #self.g2 = (0.5 + grad**2)
            #Minv = 1. / (self.xi_acc / self.count)#self.g2
            #Minv = 1. / (1e-6 + np.abs(self.xi))#self.g2
            #Minv = 1. / (1. + self.xi)  #self.g2
            #Minv = self.xi**2 / (1e-16 + self.g2)
            Minv = self.g*self.g / (self.g2 + 1e-16) / (np.sqrt(self.g2 + 1e-16) + 1e-16)
            #Minv = 1. / (np.sqrt(self.g2 + 1e-16) + 1e-16)
        else:
            Minv = 1.

        # pre-generate a sample
        sample = self._srng.normal(size=self.theta.shape) * np.sqrt(self.epsilon * 2 * self.A * Minv)
        # optionally resample momentum 
        if self.resample_momentum > 0 and self.count % self.resample_momentum == 0:
            np.copyto(self.updates, self._srng.normal(size=self.theta.shape))
        else:
            #self.updates += - self.epsilon * self.A * Minv * grad + sample - self.updates * self.epsilon * (self.xi - self.A)
            
            #np.copyto(self.updates,)
            self.updates = - self.epsilon * self.A * Minv * grad + sample
            #self.updates = - self.epsilon * self.A * Minv * grad + sample #- self.epsilon * grad/self.xi * self.xi


        #self.xi_acc = self.xi
        #self.xi_acc = self.xi * self.count
        self.theta += self.updates

        # callbacks
        self.count += 1
        if self.count % self.callback_every == 0:
            print(self.theta, self.g2)
            for callback in self.callbacks:
                callback(self.count, self)
        return self.unflatten(self.theta)
    
