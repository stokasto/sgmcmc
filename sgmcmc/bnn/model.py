import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne

from ..theano_mcmc import SGHMCSampler
from ..utils import sharedX, floatX, shuffle
from .priors import WeightPrior, LogVariancePrior

class HMCBNN(object):

    def __init__(self, f_net_fun,
                 burn_in=2000, capture_every=50, log_every=100,
                 update_prior_every=100, out_type='Gaussian',
                 updater=None, weight_prior=WeightPrior(),
                 variance_prior=LogVariancePrior(1e-4, 0.01),
                 n_target_nets=100, rng=None):
        if rng:
            self._srng = rng
        else:
            self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.updater = SGHMCSampler(precondition=True, rng=rng)
        self.f_net_fun = f_net_fun
        self.f_net = f_net_fun()
        if n_target_nets > 1:
            self.f_nets = [f_net_fun() for i in range(n_target_nets)]
        else:
            self.f_nets = []
        
        self.out_type = out_type
        self.weight_prior = weight_prior
        self.variance_prior = variance_prior
        self.burn_in = burn_in
        self.capture_every = capture_every
        self.update_prior_every = update_prior_every
        self.steps = 0
        self.bsize = 32
        self.log_every = log_every
        self.n_output_dim = len(self.f_net.output_shape)
        self.prepared = False
        if self.n_output_dim not in [2, 3]:
            raise ValueError('HMCBNN expects either 2 or 3 dimensional output from the net')
        Xbatch = T.matrix()
        self.out_fun = theano.function([Xbatch], lasagne.layers.get_output(self.f_net, Xbatch, deterministic=True))
        self.mcmc_samples = []
        if n_target_nets > 1:
            m_t, v_t = self.approximate_mean_and_var(Xbatch)
            self.predict_approximate_fun = theano.function([Xbatch], [m_t, v_t])

    def _log_like(self, X, Y, n_examples):
        f_out = lasagne.layers.get_output(self.f_net, X)
        f_mean = f_out[:, 0].reshape((-1, 1))
        f_log_var = f_out[:, 1].reshape((-1, 1))
        f_var_inv = 1. / (T.exp(f_log_var) + 1e-8)
        MSE = T.square(Y - f_mean)
        if self.out_type == 'Gaussian':
            log_like = T.sum(T.sum(-MSE * (0.5*f_var_inv) - 0.5*f_log_var, axis=1))
        else:
            raise RuntimeError('{} not implemented'.format(self.out_type))
        # scale by batch size to make this work nicely with the updaters above
        log_like /= T.cast(X.shape[0], theano.config.floatX)
        #priors, scale these by dataset size for the same reason
        # prior for the variance
        self.tn_examples = sharedX(np.float32(n_examples))
        log_like += self.variance_prior.log_like(f_log_var, n_examples) / self.tn_examples
        # prior for the weights
        log_like += self.weight_prior.log_like(lasagne.layers.get_all_params(self.f_net, regularizable=True)) / self.tn_examples
        return log_like, T.sum(MSE)


    def prepare_for_train(self, shape, bsize, epsilon, **kwargs):
        n_examples = shape[0]
        self.n_examples = n_examples
        self.steps = 0
        self.mcmc_samples = []
        self.params = lasagne.layers.get_all_params(self.f_net, trainable=True)
        self.variance_prior.prepare_for_train(n_examples)
        wdecay = self.weight_prior.prepare_for_train(self.params, n_examples)
        # setup variables for training
        Xbatch = T.matrix()
        Ybatch = T.matrix()
        print("... preparing costs")
        log_like, mse = self._log_like(Xbatch, Ybatch, n_examples)
        self.costs = -log_like
        print("... preparing updates")
        updates, burn_in_updates = self.updater.prepare_updates(self.costs, self.params, epsilon,
                                                                scale_grad=n_examples, **kwargs)
        # handle batch normalization (which we however don't use anyway)
        bn_updates = [u for l in lasagne.layers.get_all_layers(self.f_net) for u in getattr(l,'bn_updates',[])]
        updates += bn_updates

        # we have two functions, one for the burn in phase, including the burn_in_updates and one during sampling
        self.compute_cost_burn_in = theano.function([Xbatch, Ybatch], (self.costs, mse), updates=updates + burn_in_updates)
        self.compute_cost = theano.function([Xbatch, Ybatch], (self.costs, mse), updates=updates)

        # Data dependent initialization if required
        init_updates = [u for l in lasagne.layers.get_all_layers(self.f_net) for u in getattr(l,'init_updates',[])]
        
        self.data_based_init = theano.function([Xbatch], lasagne.layers.get_output(self.f_net, Xbatch, init=True), updates=init_updates)
        self.prepared = True
        self.first_step = True


    def update_for_train(self, shape, bsize, epsilon, retrain=True, **kwargs):
        n_examples = shape[0]
        self.n_examples = n_examples
        self.tn_examples.set_value(np.float32(n_examples))
        # reset the network parameters without having to recompile the theano graph
        if retrain:
            new_net = self.f_net_fun()
            new_params = lasagne.layers.get_all_param_values(new_net)
            lasagne.layers.set_all_param_values(self.f_net, new_params)
        self.first_step = True
        self.steps = 0
        self.mcmc_samples = []
        self.weight_prior.update_for_train(n_examples)
        self.variance_prior.update_for_train(n_examples)
        self.updater.reset(n_examples, epsilon, reset_opt_params=retrain, **kwargs)
    
    def step(self, X, Y, capture=True):
        if self.steps <= self.burn_in:
            cost, mse = self.compute_cost_burn_in(X, Y)
        else:
            cost, mse = self.compute_cost(X, Y)
        if capture and (self.steps > self.burn_in) \
           and (self.capture_every > 0) and (self.steps % self.capture_every == 0):
            self.mcmc_samples.append(lasagne.layers.get_all_param_values(self.f_net))
            if len(self.f_nets) > 0:
                # replace one of the f_nets
                idx = (len(self.mcmc_samples) - 1) % len(self.f_nets)
                #idx = np.random.randint(len(self.f_nets))
                #idx_mcmc = np.random.randint(len(self.mcmc_samples))
                lasagne.layers.set_all_param_values(self.f_nets[idx], self.mcmc_samples[-1])
        if self.steps % self.log_every == 0:
            print("Step: {} stored_samples : {} WD : {},  NLL = {}, MSE = {}, Noise = {}".format(self.steps, len(self.mcmc_samples), self.weight_prior.get_decay().get_value(), cost, mse, float(np.exp(self.f_net.b.get_value()))))
        if self.steps > 1 and self.steps % self.update_prior_every == 0:
            self.weight_prior.update(lasagne.layers.get_all_params(self.f_net, regularizable=True))
        self.steps += 1
        return cost

    def approximate_mean_and_var(self, Xbatch):
        if len(self.f_nets) == 0:
            raise RuntimeError("You called approximate_mean_and_var but n_target_nets is <= 1")
        mean_y = None
        ys2var = None
        mean_pred = None
        # use law of total variance to compute the overall variance
        for net in self.f_nets:
            f_out = lasagne.layers.get_output(net, Xbatch, deterministic=True)
            y = f_out[:, 0:1]
            var = T.exp(f_out[:, 1:2]) + 1e-16
            if mean_y is None:
                mean_y = y
                ys2var = T.square(y) + var
            else:
                mean_y += y
                ys2var += T.square(y) + var
        n_nets = T.cast(len(self.f_nets), theano.config.floatX)
        mean_y /= n_nets
        ys2var /= n_nets
        total_var = ys2var - T.square(mean_y)
        return mean_y, total_var

    def predict_approximate(self, X):
        return self.predict_approximate_fun(floatX(X))
    
    def predict(self, X):
        ys, var = self.sample_predictions(floatX(X))
        # compute predictive mean
        mean_pred = np.mean(ys, axis=0)
        # use the law of total variance to compute the overall variance
        var_pred = np.mean(ys ** 2 + var, axis=0) - mean_pred ** 2
        return mean_pred, var_pred

    def sample_predictions(self, X):
        y = []
        var = []
        for sample in self.mcmc_samples:
            lasagne.layers.set_all_param_values(self.f_net, sample)
            f_out = self.out_fun(X)
            y.append(f_out[:, 0])
            var.append(np.exp(f_out[:, 1]) + 1e-16)
                
        return np.asarray(y), np.asarray(var)
        

    def predict_online(self, Xtest, n_samples, X, Y, capture_every=50):
        # this runs the markov chain forward until we have made enough predictions
        old_cap = self.capture_every
        self.capture_every = capture_every
        n_steps = int(n_samples * self.capture_every)
        indices = np.random.permutation(np.arange(len(X)))
        y = []
        var = []
        for s in range(n_steps):
            # sample a random batch
            start = int(np.random.randint(len(indices - self.bsize)))
            idx = indices[start:start+int(self.bsize)]
            xmb = X[idx]
            ymb = Y[idx]
            # and push the markov chain forward
            self.step(xmb, ymb, capture=False)
            # predict if we are in a capture step
            if s % self.capture_every == 0:
                f_out = self.out_fun(Xtest)
                y.append(f_out[:, 0])
                #var.append(np.log(1. + np.exp(f_out[:, 1])))
                var.append(np.exp(f_out[:, 1]))
        self.capture_every = old_cap
        return np.asarray(y), np.asarray(var)
    
    def train(self, X, Y, n_steps, retrain=True, bsize=32, epsilon=1e-2, **kwargs):
        self.X = X
        self.Y = Y        
        
        if n_steps < self.burn_in:
            raise ValueError("n_steps must be larger than burn_in")
        print('X shape : {}'.format(X.shape))
        print('Y shape : {} '.format(Y.shape))
        ndata = X.shape[0]
        self.bsize = bsize
        if X.shape[0] < 2*self.bsize:
            self.bsize = X.shape[0]
        n_batches = int(np.ceil(ndata / self.bsize))
        n_epochs = int(np.floor(n_steps / n_batches))
        #data_per_batch = ndata / self.bsize
        if not self.prepared:
            self.prepare_for_train(X.shape, self.bsize, epsilon, **kwargs)
        else:
            self.update_for_train(X.shape, self.bsize, epsilon, retrain=retrain, **kwargs)
        for e in range(n_epochs):
            X, Y = shuffle(X, Y)
            #print("Starting epoch: {}".format(e))
            batches = len(X) // self.bsize
            for b in range(batches):
                start = b*self.bsize
                xmb = X[start:start+self.bsize]
                ymb = Y[start:start+self.bsize]
                if self.first_step:
                    print("Performing data based initialization")
                    self.data_based_init(xmb)
                    self.first_step = False
                self.step(xmb, ymb)
