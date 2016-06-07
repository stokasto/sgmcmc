import autograd
import autograd.numpy as np
import autograd.numpy.random as npr

from sgmcmc.autograd_mcmc import SGMCMCSampler, SGRHMCSampler, SGLDSampler
import matplotlib.pyplot as plt

def neg_log_like(theta, i):
    nll = (theta + 4) * (theta + 1) * (theta - 1) * (theta - 3) / 14 + 0.5
    return np.sum(nll)

def grad_nll(theta, i):
    g = (theta+1)*(theta-1)*(theta-3)/14 + (theta+4)*(theta-1)*(theta-3)/14 + (theta+4)*(theta+1)*(theta-3)/14 + (theta+4)*(theta+1)*(theta-1)/14
    return g

def hess_nll(theta, i):
    h = (theta+1)*(theta-1)/14 + (theta+1)*(theta-3)/14 + (theta-1)*(theta-3)/14 \
        + (theta+4)*(theta-1)/14 + (theta+4)*(theta-3)/14 + (theta-1)*(theta-3)/14 \
        + (theta+4)*(theta+1)/14 + (theta+4)*(theta-3)/14 + (theta+1)*(theta-3)/14 \
        + (theta+4)*(theta+1)/14 + (theta+4)*(theta-1)/14 + (theta+1)*(theta-1)/14
    return h
    

def print_xi_callback(i, s):
    Exi = s.xi_acc/s.count
    print("{} : xi = {} E[xi] = {} |xi| = {} 1/(E[xi]+{}) = {}".format(i, s.xi, Exi, abs(s.xi), s.exp_weight, 1./(abs(Exi) + s.exp_weight)))    

epsilon = 1
A = 1
theta = npr.normal(size=(1,))
#theta = np.array([-2.94])
#sampler = SGMCMCSampler(precondition=True, resample_momentum=0)
#sampler = SGLDSampler(precondition=False, resample_momentum=0)
sampler = SGLDSampler(precondition=True, resample_momentum=0)
#sampler = SGRHMCSampler(precondition=True, resample_momentum=50)
# using custom gradient function
updates = sampler.prepare_updates(neg_log_like, theta, epsilon, grad=grad_nll, A=A, callbacks=[print_xi_callback], fd_hess=True, callback_every=1000)
# using autograd
#updates = sampler.prepare_updates(neg_log_like, theta, epsilon, A=A, callbacks=[print_xi_callback], fd_hess=True)

n_samples = 12 * 10**4
#n_samples = 150
#sampler.p[:] = 0
samples = []
for i in range(n_samples):
    #print(sampler.p)
    sample = sampler.step(i)
    samples.append(np.copy(sample))
    #print(sample, sampler.p)

true_step = 0.001
xs = np.arange(-6, 6, true_step)
nlls = np.zeros_like(xs)
for i,x in enumerate(xs):
    nlls[i] = neg_log_like(x, i)

# compute likelihood
lls = np.exp(-nlls)
# approximately compute z via euler integration
z = np.sum(lls) * true_step

# approximate density from the samples
step_sample = 0.1
xgrid = np.arange(-6, 6, step_sample) 
ygrid = np.histogram(samples, xgrid, density=True)[0]
plt.plot(xs, lls/z)
plt.plot(xgrid[:len(ygrid)], ygrid)
plt.figure()
ygrid2 = np.histogram(samples, 1500)[0]
plt.scatter(np.arange(len(ygrid)), ygrid)
#plt.scatter(samples, samples)
plt.show()

