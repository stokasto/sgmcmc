import theano
import theano.tensor as T
import math
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..utils import *

#HALF_LOG2PI = T.constant((0.5 * np.log(2*np.pi)).astype(theano.config.floatX))
#SQUARE_2PI = T.constant(np.square(2.*np.pi))
c = - 0.5 * math.log(2*math.pi)

def log_normal2(x, mean, log_var, eps=1e-4):
    return c - log_var/2 - (x - mean)**2 / (2 * T.exp(log_var) + eps)

   
class WeightPrior(object):

    def __init__(self, rng=None, alpha=1, beta=1000.):
        if rng:
            self._srng = rng
        else:
            self._srng = RandomStreams(np.random.randint(1, 2147462579))

        self.alpha_prior = alpha
        self.beta_prior = beta
        self.wdecay = theano.shared(np.float32(1.))

    def get_decay(self):
        return self.wdecay

    def prepare_for_train(self, params, n_data):
        self.n_data = n_data
        return self.wdecay

    def update_for_train(self, n_data):
        self.n_data = n_data
        self.wdecay.set_value(1.)

    def log_like(self, params):
        ll = 0.
        # NOTE: we are dropping all constants here
        for p in params:
            ll += T.sum(-self.wdecay * 0.5 * T.square(p))
        return ll
    
    def update(self, params):
        W_sum = 0
        W_size = 0
        for p in params:
            W = p.get_value()
            W_sum += np.sum(np.square(W))
            W_size += np.prod(W.shape)
        alpha = self.alpha_prior + 0.5 * W_size
        beta  = self.beta_prior + 0.5 * W_sum
        p_wd = np.random.gamma(alpha, 1./(beta + 1e-4))
        # wd is the next weight decay
        wd = p_wd #/ self.n_data
        # the scaling with n_data above is now done in
        # the log likeliehood (where it should be!)
        self.wdecay.set_value(np.float32(wd))
    
class LogVariancePrior(object):

    def __init__(self, prior_out_std, prior_out_std_prec=0.1):
        self.prior_out_std = prior_out_std
        self.prior_out_std_prec = prior_out_std_prec

    def prepare_for_train(self, n_examples):
        self.n_examples = theano.shared(np.float32(n_examples))

    def update_for_train(self, n_examples):
        self.n_examples.set_value(np.float32(n_examples))
        
    def log_like(self, log_var, n_examples):
        #return T.sum(T.sum(-T.square(log_var - T.log(self.prior_out_std)) / (2*self.prior_out_std_prec) - 0.5*T.log(self.prior_out_std_prec) -HALF_LOG2PI, axis=1)) #/ self.n_examples
        return T.mean(T.sum(-T.square(log_var - T.log(self.prior_out_std)) / (2*self.prior_out_std_prec) - 0.5*T.log(self.prior_out_std_prec), axis=1)) #/ self.n_examples

class HorseShoePrior(object):

    def __init__(self, scale=0.1):
        self.scale = scale

    def prepare_for_train(self, n_examples):
        self.n_examples = theano.shared(np.float32(n_examples))

    def update_for_train(self, n_examples):
        self.n_examples.set_value(np.float32(n_examples))
        
    def log_like(self, log_var, n_examples):
        return T.mean(T.sum(T.log(T.log(1. + (self.scale / (T.exp(log_var))))), axis=1))

