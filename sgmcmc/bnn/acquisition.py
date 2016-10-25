import theano
import theano.tensor as T

import lasagne
from ..utils import *

PI = T.cast(np.pi, theano.config.floatX)
SQRT2 = T.cast(np.sqrt(2), theano.config.floatX)
epsilon = T.cast(1e-10, theano.config.floatX)

def pdf(sample, location=0, var=1):
    z = 0.5 * ((sample - location) ** 2) / var
    return T.exp(-z) / T.sqrt(2. * PI * var)

def cdf(sample, location=0, var=1):
    z = (sample - location) / T.sqrt(var)
    return .5 * (T.erfc(-z / SQRT2))


class Acquisition(object):

    def __init__(self, net):
        self.net = net
        self.prepared_for_optimization = False
        X = T.matrix()
        cost, acq, additional_args = self._prepare_for_computation(X)
        self.predict_fun = theano.function([X] + additional_args, acq)

    def acquisition_function(self, mean, std, f_min):
        raise NotImplementedError("Acquisition function not implemented in base class")

    def _prepare_for_computation(self, X):
        f_min = T.matrix()
        mean, var = self.net.approximate_mean_and_var(X)
        std = T.sqrt(var + epsilon)
        cost, acq, additional_args = self.acquisition_function(mean, std, f_min)
        return cost, acq, additional_args
        
    def _prepare_for_optimization(self, Xinit, incumbent, lrate, constrain_dim, constraint):
        if constrain_dim is not None and constrain_dim != -1:
            raise ValueError("hard constraints are currently only supported for dim -1!")
        if constrain_dim is None:
            self.Xinit_shape = Xinit.shape        
            self.Xbatch = sharedX(np.zeros_like(Xinit))
            cost, acq, additional_args = self._prepare_for_computation(self.Xbatch)
        else:
            self.Xinit_shape = (Xinit.shape[0], Xinit.shape[1]-1)
            self.Xbatch = sharedX(np.zeros(self.Xinit_shape))
            constraint_arr = np.zeros((Xinit.shape[0], 1))
            constraint_arr[:, 0] = constraint
            self.constraint = sharedX(constraint_arr)
            Xtmp = T.concatenate([self.Xbatch, self.constraint], axis=1)
            cost, acq, additional_args = self._prepare_for_computation(Xtmp)
        
        # setup optimization step function        
        updates, optim_params = smorms3(cost, [self.Xbatch], lrate=lrate, gather=True)
        self.optim_params = optim_params
        self.step_fun = theano.function(additional_args, (cost, acq),
                                        updates = updates)
        self.prepared_for_optimization = True

    def _update_for_optimization(self, Xinit, incumbent, lrate, constrain_dim, constraint):
        if Xinit.shape != self.Xinit_shape:
            print("Re-Compiling theano functions for acquisition")
            self._prepare_for_optimization(Xinit, incumbent, lrate, constrain_dim, constraint)
        else:
            smorms3_reset(self.optim_params)
        
    def optimize(self, Xinit, incumbent, bounds, steps, lrate=5e-4,
                 constrain_dim=None, constraint=None):
        if not self.prepared_for_optimization:
            print("Compiling theano functions for acquisition")
            self._prepare_for_optimization(Xinit, incumbent, lrate, constrain_dim, constraint)
        else:
            self._update_for_optimization(Xinit, incumbent, lrate, constrain_dim, constraint)
        if constrain_dim is not None:
            Xinit = Xinit[:, :-1]
        self.Xbatch.set_value(Xinit)
        for s in xrange(steps):
            cost,acq = self.step(incumbent)
            # TODO check if any of the self.Xbatch went out of bounds
            # if so -> project them back
            X = self.Xbatch.get_value()
            #for i,(lower,upper) in enumerate(bounds):
            for i in range(X.shape[1]):
                (lower,upper) = bounds[i]
                X[:, i] = np.clip(X[:, i], lower, upper)
            self.Xbatch.set_value(X)
            max_idx = np.argmax(acq)
            max_val = acq[max_idx]
            max_x   = X[max_idx]
            print("Cost : {} best Acquistion = {}, X = {}".format(cost, max_val, max_x))
        if constrain_dim is None:
            return X, acq, max_idx
        else:
            constraint_arr = np.zeros((Xinit.shape[0], 1))
            constraint_arr[:, 0] = constraint
            X = floatX(np.concatenate([X, constraint_arr], axis=1))
            return X, acq, max_idx

    def compute_acquisition(self, X, f_min):
        return self.predict_fun(X, f_min)

    def step(self, f_min):
        return self.step_fun(f_min)        
            

class EI(Acquisition):

    def __init__(self, net, par=0.01):
        self.par = par
        super(EI, self).__init__(net)

    def acquisition_function(self, mean, std, f_min):
        # prepare computation graph for EI
        f_min_rep = T.extra_ops.repeat(f_min.reshape((-1,1)), mean.shape[0], axis=0)
        diff = (f_min_rep - mean - self.par)
        std = std + 1e-3 #T.maximum(std + 1e-3, epsilon)
        diff_std = diff / std
        EI = diff * cdf(diff_std) + std * pdf(diff_std)
        cost = -T.sum(EI) 
        return cost, EI, [f_min]

        
class UCB(Acquisition):

    def __init__(self, net, kappa=2.):
        self.kappa = kappa
        super(UCB, self).__init__(net)

    def acquisition_function(self, mean, std, f_min):
        UCB = -mean + self.kappa * std
        cost = -T.sum(UCB)
        return cost, UCB, []

    def compute_acquisition(self, X, f_min):
        # no need for f_min
        return self.predict_fun(X)

    def step(self, f_min):
        # no need for f_min
        return self.step_fun()
