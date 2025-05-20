"""
This code is used to simulate data points along with bad data points (from the same function, however this is not necesary),
and then fit the data points using MCMC. The code also calculates the probability of each point being a bad point based on the fitted parameters.

This code is for the case where the function tries to model figure 7 of Kulkarni et al. 2019.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import sys
import emcee
import datetime
from scipy.optimize import curve_fit

start_time = datetime.datetime.now()
current_time = start_time.strftime("%Y%m%d_%H%M%S")

flag = 0
sig = 0.2

zmin = 0
zmax = 15

method = 'double_power_law'  # Change this to 'poly' for polynomial fitting
if method == 'poly':
    NUM = 3
elif method == 'double_power_law':
    NUM = 4

def function(x, params):
    if method == 'poly':
        return np.polyval(params, x)
    elif method == 'double_power_law':
        log_rho_star = params[0]
        rho_star = 10**log_rho_star
        z_star = params[1]
        alpha = params[2]
        beta = params[3]
        return rho_star / (10**((x - z_star) * (1 + alpha)) + 10**((x - z_star) * (1 + beta)))
    else:
        raise ValueError("Invalid method. Use 'poly' or 'double_power_law'.")
    

def logprior(params, NUM=2):
    if method == 'poly':
        func_params = params[:NUM]
        Pb, Yb, Vb = params[NUM:]
        if Pb < 0 or Pb >= 1:
            return -np.inf
        if Vb <= 0:
            return -np.inf
        
        sum = - np.log(1 + Pb) - np.log(1 + Vb) 
        for i in range(NUM):
            # print("i = ", i)
            # print("func_params[i] = ", func_params[i])
            if i == 0 and not (-2.5 < func_params[i] < 0):
                print("Rejected func_params[0] = ", func_params[i])
                print("Returned -inf")
                return -np.inf
            if i == NUM - 1:
                if not (-100 < func_params[i] < 0):
                    print("Rejected func_params[NUM-1] = ", func_params[i])
                    print("Returned -inf")
                    return -np.inf
                sum -= np.log(1000 + func_params[i])
            else:
                if not (-40 < func_params[i] < 40):
                    print("Rejected func_params[i] = ", func_params[i])
                    print("i = ", i)
                    print("func_params = ", func_params)

                    print("Returned -inf")
                    return -np.inf
        return sum

    elif method == 'double_power_law':
        logrho, z_star, alpha, beta = params[:NUM]
        Pb, Yb, Vb = params[NUM:]

        if Pb < 0 or Pb > 1:
            return -np.inf
        if Vb <= 0:
            return -np.inf
        
        if logrho < -20 or logrho > 0:
            return -np.inf
        
        if z_star < -7 or z_star > 0:
            return -np.inf
        
        if alpha < -7 or alpha > beta:
            return -np.inf
        
        if beta > 0:
            return -np.inf

        return - np.log(1 + Pb) - np.log(1 + Vb) 
    else:
        raise ValueError("Invalid method. Use 'poly' or 'double_power_law'.")

def loglikelihood(params, x, y, sig, NUM=2):
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

def logposterior(params, x, y, sig, NUM=2):
    lp = logprior(params, NUM)
    if not np.isfinite(lp):
        return -np.inf

    ll = loglikelihood(params, x, y, sig, NUM)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def find_best_fit(data):
    if method == 'poly':
        return np.polyfit(data[0], data[1], NUM)
    elif method == 'double_power_law':
        x, y, sig = data
        def model_function(x, log_rho_star, alpha, beta):
            rho_star = 10**log_rho_star
            z_star = 2.5
            return rho_star / (10**((x - z_star) * (1 + alpha)) + 10**((x - z_star) * (1 + beta)))
        popt, pcov = curve_fit(model_function, x, y)
        return [popt[0], 3, popt[1], popt[2]]


def mcmc_2_main(data, nwalkers=50, nprod=1000, nburn=1000, NUM=2, plot_number=0):
    x, y, sig = data
    ndim = NUM + 3  # Number of parameters (NUM for function params, 3 for Pb, Yb, Vb)

    popt = find_best_fit(data)
    print("popt = ", popt)
    import sys
    sys.exit()

    if method == 'poly':
        param_ranges = [(-10, 10) for i in range(NUM)] + [(0.1, 0.9), (-10, 10), (1, 5)]
        labels = [f'a{i}' for i in range(NUM)] + ["Pb", "Yb", "Vb"]

    elif method == 'double_power_law':
        param_ranges = [(-10, 10), (-20, 20), (-1, 1), (-1, 1), (0.1, 0.9), (-10, 10), (1, 5)]
        labels = ["logrho", "z_star", "alpha", "beta", "Pb", "Yb", "Vb"]

    p0 = np.array([np.random.uniform(low, high, size=nwalkers) for low, high in param_ranges]).T

    assert p0.shape == (nwalkers, len(param_ranges)), "Shape of p0 is incorrect!"

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=[x, y, sig, NUM])

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
    corner.corner(samples, labels=labels, fig=fig, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    plt.savefig(f"check2_{plot_number}_corner_plot_3_{current_time}.png")  # Save the corner plot as a PNG file with the current time
    plt.close()

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(sampler.get_chain()[:, :, i], alpha=0.5)
        ax.set_ylabel(f"param {i}")
        ax.yaxis.set_label_coords(-0.1, 0.5)

    plt.xlabel("step number")
    plt.savefig(f"check2_{plot_number}_chain_plot_3_{current_time}.png")  # Save the chain plot as a PNG file with the current time
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


def find_bad_data(data, param, NUM=2, plot_number=0):
    x, y, sig = data
    Pb, Yb, Vb = param[NUM:]

    model_vals = function(x, param[:NUM])

    Npoints = len(x)
    bad_prob_list = []
    true_cnt = 0

    for i in range(Npoints):
        p_fg = (1 / np.sqrt(2 * np.pi * sig[i]**2)) * np.exp(-0.5 * ((y[i] - model_vals[i]) / sig[i])**2)
        p_bg = (1 / np.sqrt(2 * np.pi * (Vb + sig[i]**2))) * np.exp(-0.5 * ((y[i] - Yb)**2 / (Vb + sig[i]**2)))


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

def read_data(filename):
    data = np.loadtxt(filename, comments='#', delimiter=',', unpack=True)

    all_data = np.array([data[0], np.log10(data[3]), ((1 / (np.log(10) * data[3]) * data[4]) + (1 / (np.log(10) * data[3]) * data[5])) / 2])
    # Find breakpoints in the data
    breaks = np.where(np.diff(all_data[0]) <= 0)[0] + 1

    # Split the data into subarrays
    subarrays = np.split(all_data, breaks, axis=1)

    return subarrays
    


n_walkers = 50
n_prod = 100000
n_burn = 1000
n_thin = 50


subarrays = read_data("rhoqso_output_data.txt")
print("len(subarrays) = ", len(subarrays))


main_fig = plt.figure(figsize=(5, 10), dpi=300)

x_arr = np.linspace(zmin, zmax, 1000)

# Print the subarrays
for i, subarray in enumerate(subarrays):
    print(f"\nSubarray {i}:")
    z, rho, rho_sig = subarray
    print("z = ", z)
    print("rho = ", rho)
    print("rho_sig = ", rho_sig)

    result = mcmc_2_main(subarray, nwalkers=n_walkers, nprod=n_prod, nburn=n_burn, NUM=NUM, plot_number=i)

    prob_bad_points_2 = find_bad_data(subarray, result, NUM=NUM, plot_number=i)

    y_arr_2 = function(x_arr, result[:NUM])

    print("\n\n\t\tprob_bad_points = ", prob_bad_points_2)
    mask = prob_bad_points_2 > 0.5
    print("mask = ", mask)

    plt.figure(main_fig.number)
    plt.errorbar(z, rho, yerr=sig, fmt='o', ms=10, capsize=4, zorder=1)
    plt.scatter(z[mask], rho[mask], c='red', label='Bad Data Points', alpha=0.7, edgecolor='black', s=50, zorder=2)
    break






        

plt.figure(main_fig.number)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting for x with Bad data in dataset')
plt.plot(x_arr, y_arr_2, color='blue', label='Good Function')
plt.xlim(zmin, zmax)
plt.ylim(-11, -3)
# plt.ylim(-14, -2)
plt.legend()
plt.grid()
plt.savefig(f"check2_plot_2_{current_time}.png")  # Save the plot as a PNG file with the current time


end_time = datetime.datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")


plt.show()
plt.close()
