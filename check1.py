"""
This code is used to simulate data points along with bad data points (from the same function, however this is not necesary),
and then fit the data points using MCMC. The code also calculates the probability of each point being a bad point based on the fitted parameters.

This code is for the case where the function is double power law (equation 7 in Kulkarni et al. 2019).
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import sys
import emcee
import datetime

start_time = datetime.datetime.now()
current_time = start_time.strftime("%Y%m%d_%H%M%S")

flag = 0
sig = 0.2

def function(x, params):
    logphi = params[0]
    phi = 10**logphi
    M_star = params[1]
    alpha = params[2]
    beta = params[3]

    return np.log10(phi / (10**(0.4 * (x - M_star) * (1 + alpha)) + 10**(0.4 * (x - M_star) * (1 + beta))))


def create_data(num_good_points=100, num_bad_points=10):
    good_param = [-5.72, -21.3, -2.74, -1.07]
    bad_param = [-8.92, -25.3, -4.84, -1.57]

    xlim = (-31, -19)
    ylim = (-14, -2)

    x_good = np.sort(np.random.uniform(xlim[0], xlim[1], num_good_points))
    y_good = function(x_good, good_param) + np.random.normal(0, sig, num_good_points)

    x_bad = np.sort(np.random.uniform(xlim[0], xlim[1], num_bad_points))
    y_bad = function(x_bad, bad_param) + np.random.normal(0, sig, num_bad_points)

    plt.scatter(x_good, y_good, color='blue', label='Good Points')
    plt.scatter(x_bad, y_bad, color='red', label='Bad Points')
    x_arr = np.linspace(xlim[0], xlim[1], 1000)
    y_arr_good = function(x_arr, good_param)
    y_arr_bad = function(x_arr, bad_param)
    plt.plot(x_arr, y_arr_good, color='blue', label='Good Function')
    plt.plot(x_arr, y_arr_bad, color='red', label='Bad Function')
    plt.title('Good and Bad Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.xlim(xlim[1], xlim[0])
    # plt.ylim(ylim[0], ylim[1])
    plt.grid()
    plt.savefig(f"check1_real_data_plot_{current_time}.png") 
    plt.close()

    x_combined = np.concatenate((x_good, x_bad))
    y_combined = np.concatenate((y_good, y_bad))
    sorted_indices = np.argsort(x_combined)

    return np.sort(x_combined), y_combined[sorted_indices]

def logprior(params, NUM=2):
    logphi, M_star, alpha, beta = params[:NUM]
    Pb, Yb, Vb = params[NUM:]

    if Pb < 0 or Pb > 1:
        return -np.inf
    if Vb <= 0:
        return -np.inf
    
    if logphi < -10 or logphi > 0:
        return -np.inf
    
    if M_star < -50 or M_star > 0:
        return -np.inf
    
    if alpha < -7 or alpha > beta:
        return -np.inf
    
    if beta > 0:
        return -np.inf

    return - np.log(1 + Pb) - np.log(1 + Vb) 

def loglikelihood(params, x, y, NUM=2):
    func_params = params[:NUM]
    Pb, Yb, Vb = params[NUM:]


    epsilon = 1e-10  # Small value to prevent division by zero
    safe_sig2 = sig**2 + epsilon
    safe_Vb = Vb + epsilon

    logforeground_model = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip(((y - function(x, func_params)) / sig)**2, -1e10, 1e10))
    logbackground_model = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((y - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

    a = np.log(1 - Pb) + logforeground_model
    b = np.log(Pb) + logbackground_model

    log10L = np.sum(np.logaddexp(a, b)) / np.log(10)

    return log10L

def logposterior(params, x, y, NUM=2):
    lp = logprior(params, NUM)
    if not np.isfinite(lp):
        return -np.inf

    ll = loglikelihood(params, x, y, NUM)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def mcmc_2_main(data, nwalkers=50, nprod=1000, nburn=1000, NUM=2):
    x, y = data
    ndim = NUM + 3  # Number of parameters (2 for function params, 3 for Pb, Yb, Vb)
    nwalkers = nwalkers
    param_ranges = [(0, 5), (-20, 20), (-1, 1), (-1, 1), (0.1, 0.9), (-10, 10), (1, 5)]

    p0 = np.array([np.random.uniform(low, high, size=nwalkers) for low, high in param_ranges]).T

    assert p0.shape == (nwalkers, len(param_ranges)), "Shape of p0 is incorrect!"

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=[x, y, NUM])

    sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(None, nprod)

    samples = sampler.get_chain(flat=True)
    results = {}
    for i in range(samples.shape[1]):
        param_samples = samples[:, i]
        median = np.median(param_samples)
        lower_1sigma = np.percentile(param_samples, 16)
        upper_1sigma = np.percentile(param_samples, 84)

        results[f"param_{i}"] = {
            "median": median,
            "1sigma_lower": median - lower_1sigma,
            "1sigma_upper": upper_1sigma - median,
        }
    for param, stats in results.items():
        print(f"{param}:")
        print(f"  Median: {stats['median']}")
        print(f"  1-Sigma Lower: {stats['1sigma_lower']}")
        print(f"  1-Sigma Upper: {stats['1sigma_upper']}")

    fig = plt.figure(figsize=(12, 8), dpi=100)
    corner.corner(samples, labels=["logphi", "M_star", "alpha", "beta", "Pb", "Yb", "Vb"], fig=fig, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    plt.savefig(f"check1_corner_plot_3_{current_time}.png")  # Save the corner plot as a PNG file with the current time
    plt.close()


    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(sampler.get_chain()[:, :, i], alpha=0.5)
        ax.set_ylabel(f"param {i}")
        ax.yaxis.set_label_coords(-0.1, 0.5)

    plt.xlabel("step number")
    plt.savefig(f"check1_chain_plot_3_{current_time}.png")  # Save the chain plot as a PNG file with the current time
    plt.close()
    acceptance_rate = np.mean(sampler.acceptance_fraction)
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation time for each parameter:")
        for i, t in enumerate(tau):
            print(f"  Parameter {i}: {t:.2f}")
    except emcee.autocorr.AutocorrError:
        print("Warning: Autocorrelation time could not be reliably estimated.")

    return [stats["median"] for param, stats in results.items()]


def find_bad_data(data, param, NUM=2):
    x, y = data
    Pb, Yb, Vb = param[NUM:]

    model_vals = function(x, param[:NUM])

    Npoints = len(x)
    bad_prob_list = []
    true_cnt = 0

    for i in range(Npoints):
        p_fg = (1 / np.sqrt(2 * np.pi * sig**2)) * np.exp(-0.5 * ((y[i] - model_vals[i]) / sig)**2)
        p_bg = (1 / np.sqrt(2 * np.pi * (Vb + sig**2))) * np.exp(-0.5 * ((y[i] - Yb)**2 / (Vb + sig**2)))


        epsilon = 1e-20
        numerator = Pb * p_bg + epsilon
        denominator = (1 - Pb) * p_fg + Pb * p_bg + epsilon

        bad_prob = numerator / denominator

        if bad_prob > 0.8:
            true_cnt += 1

            ratio = p_fg / p_bg
            print("ratio = ", ratio)  # This should be ~0 if model fits well
        
        bad_prob_list.append(bad_prob)

    print("true_cnt = ", true_cnt)
    print("Implied bad points = ", true_cnt / Npoints * 100, "%")
    return np.array(bad_prob_list)

data = create_data(20, 5)
n_walkers = 50
n_prod = 100000
n_burn = 1000
n_thin = 50
NUM = 4


res_2 = mcmc_2_main(data, nwalkers=n_walkers, nprod=n_prod, nburn=n_burn, NUM=NUM)
x_arr = np.linspace(-31, -19, 1000)

print("res_2 = ", res_2)

prob_bad_points_2 = find_bad_data(data, res_2, NUM=NUM)

y_arr_2 = function(x_arr, res_2[:NUM])

print("prob_bad_points = ", prob_bad_points_2)
mask = prob_bad_points_2 > 0.5
print("mask = ", mask)


plt.errorbar(data[0], data[1], yerr=sig, fmt='o', ms=10, capsize=4, zorder=1)
plt.scatter(data[0][mask], data[1][mask], c='red', label='Bad Data Points', alpha=0.7, edgecolor='black', s=50, zorder=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting for x with Bad data in dataset')
plt.plot(x_arr, y_arr_2, color='blue', label='Good Function')
plt.xlim(-19, -31)
# plt.ylim(-14, -2)
plt.legend()
plt.grid()
plt.savefig(f"check1_plot_2_{current_time}.png")  # Save the plot as a PNG file with the current time


end_time = datetime.datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")


plt.show()
plt.close()
