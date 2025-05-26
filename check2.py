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
import random

start_time = datetime.datetime.now()
current_time = start_time.strftime("%Y%m%d_%H%M%S")

flag = 0
sig = 0.2

zmin = 0
zmax = 15

k = 1 # Slope of the sigmoid function

method = 'smooth_piecewise_linear'  # Change this to 'poly' for polynomial fitting
if method == 'poly':
    NUM = 3
elif method == 'smooth_piecewise_linear':
    NUM = 4

def function(x, params):
    if method == 'poly':
        return np.polyval(params, x)
    elif method == 'smooth_piecewise_linear':
        m1 = params[0]
        c1 = params[1]
        m2 = params[2]
        c2 = params[3]
        
        a = - (c2 - c1) / (m2 - m1)
        w = 1 / (1 + 10**(-k * (x - a)))
        y1 = m1 * x + c1
        y2 = m2 * x + c2
        
        return w * y1 + (1 - w) * y2
    else:
        raise ValueError("Invalid method. Use 'poly' or 'smooth_piecewise_linear'.")
    

def logprior(params, NUM=2):
    if method == 'poly':
        func_params = params[:NUM]
        Pb, Yb, Vb = params[NUM:]
        if Pb < 0 or Pb >= 0.5:
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

    elif method == 'smooth_piecewise_linear':
        m1, c1, m2, c2 = params[:NUM]
        Pb, Yb, Vb = params[NUM:]

        a = - (c2 - c1) / (m2 - m1)

        if Pb < 0 or Pb > 1:
            # print("Rejected Pb = ", Pb)
            return -np.inf
        if Vb <= 0:
            # print("Rejected Vb = ", Vb)
            return -np.inf
        
        # if m1 > m2 or m1 < -10:
        #     return -np.inf
        # if m2 > 10:
        #     return -np.inf
        if m1 > 0 or m1 < -5:
            # print("Rejected m1 = ", m1)
            return -np.inf
        if m2 < 0 or m2 > 2:
            # print("Rejected m2 = ", m2)
            return -np.inf
        if c1 < -20 or c1 > 20:
            # print("Rejected c1 = ", c1)
            return -np.inf
        if c2 < -20 or c2 > 20:
            # print("Rejected c2 = ", c2)
            return -np.inf
        if not 0 < a < 5:
            # print("Rejected a = ", a)
            return -np.inf

        # # print("Accepted a = ", a)

        # ret = - np.log(10000 * Pb)
        # print("\n\nPb = ", Pb)
        # print("log(Pb) = ", np.log(Pb))
        # print("ret = ", ret)
        # return - np.log(Pb)**100 

        log_ret = - np.tan((Pb + 1 / 2) * np.pi)
        ret = np.exp(log_ret)
        ret_2 = np.log(log_ret)
        ret_3 = - np.tan((Pb + 1) * np.pi / 2)

        if random.random() < 1e-4:
            print("\n\nPb = ", Pb)
            print("log_ret = ", log_ret)
            print("ret = ", ret)
            print("ret_2 = ", ret_2)
            print("ret_3 = ", ret_3)
        return ret_3

    else:
        raise ValueError("Invalid method. Use 'poly' or 'smooth_piecewise_linear'.")

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
    elif method == 'smooth_piecewise_linear':
        x, y, sig = data
        mask = np.abs(x - 3) >= 0.75
        x = x[mask]
        y = y[mask]
        sig = sig[mask]
        def model_function(x, m1, c1, m2, c2):
            a = - (c2 - c1) / (m2 - m1)
            w = 1 / (1 + 10**(-k * (x - a)))
            y1 = m1 * x + c1
            y2 = m2 * x + c2
            return w * y1 + (1 - w) * y2
        p0 = [2, 1, 1, 4]
        popt, pcov = curve_fit(model_function, x, y, p0=p0)

        return popt


def mcmc_2_main(data, nwalkers=50, nprod=1000, nburn=1000, NUM=2, plot_number=0):
    x, y, sig = data
    ndim = NUM + 3  # Number of parameters (NUM for function params, 3 for Pb, Yb, Vb)

    popt = find_best_fit(data)
    print("popt = ", popt)
    # print('\n\n\n\n')
    # # Plot the function along with the data points before running MCMC
    # plt.figure(figsize=(8, 6))
    # plt.errorbar(x, y, yerr=sig, fmt='o', label='Data Points', capsize=4)
    # x_fit = np.linspace(np.min(x), np.max(x), 500)
    # y_fit = function(x_fit, popt[:NUM])
    # plt.plot(x_fit, y_fit, color='red', label='Initial Fit')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Data and Initial Fit')
    # plt.legend()
    # plt.grid()
    # # plt.show()
    # plt.close()
    print()
    # for i in range(len(x)):
    #     print(f'{x[i]:.3f} {y[i]:.3f}')
    # import sys
    # sys.exit()

    param_ranges = [(popt[i] - 0.01, popt[i] + 0.01) for i in range(NUM)] + [(0.001, 0.2), (-100, 100), (1, 50)]
    if method == 'poly':
        labels = [f'a{i}' for i in range(NUM)] + ["Pb", "Yb", "Vb"]

    elif method == 'smooth_piecewise_linear':
        labels = ["m1", "c1", "m2", "c2", "Pb", "Yb", "Vb"]

    print("\n\nparam_ranges =\n", np.array2string(np.array(param_ranges), precision=4))

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

        if i < NUM:
            results[f"param_{i}"] = {
                "median": median,
                "1sigma_lower": median - lower_1sigma,
                "1sigma_upper": upper_1sigma - median,
            }
        elif i == NUM:
            results["Pb"] = {
                "median": median,
                "1sigma_lower": median - lower_1sigma,
                "1sigma_upper": upper_1sigma - median,
            }
            
        elif i == NUM + 1:
            results["Yb"] = {
                "median": median,
                "1sigma_lower": median - lower_1sigma,
                "1sigma_upper": upper_1sigma - median,
            }
        elif i == NUM + 2:
            results["Vb"] = {
                "median": median,
                "1sigma_lower": median - lower_1sigma,
                "1sigma_upper": upper_1sigma - median,
            }

    for param, stats in results.items():
        print(f"{param}:")
        print(f"  Median: {stats['median']}")
        print(f"  1-Sigma Lower: {stats['1sigma_lower']}")
        print(f"  1-Sigma Upper: {stats['1sigma_upper']}")

    if results["Pb"]["median"] > 0.9:
            print("Warning: Pb is greater than 0.9. This may indicate a poor fit.")
            sys.exit('Pb > 0.9. Exiting.')

    fig = plt.figure(figsize=(5, 10), dpi=100)
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
    
    if acceptance_rate < 0.1:
        print("Warning: Low acceptance rate. Consider adjusting the number of walkers or the parameter ranges.")
        sys.exit('Low acceptance rate. Exiting.')
    if acceptance_rate > 0.5:
        print("Warning: High acceptance rate. Consider adjusting the number of walkers or the parameter ranges.")
        sys.exit('High acceptance rate. Exiting.')
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

    implied_bad_points = true_cnt / Npoints * 100
    print("true_cnt = ", true_cnt)
    print("Implied bad points = ", implied_bad_points, "%")

    if implied_bad_points > 90:
        print("Warning: High percentage of bad points detected. Consider adjusting the model or data.")
        sys.exit('High percentage of bad points. Exiting.')
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
n_prod = 10000
n_burn = 1000
n_thin = 50


subarrays = read_data("rhoqso_output_data.txt")
print("len(subarrays) = ", len(subarrays))


main_fig = plt.figure(figsize=(3, 5), dpi=300)

x_arr = np.linspace(zmin, zmax, 1000)

a = []

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
    plt.errorbar(z, rho, yerr=sig, fmt='o', ms=7, capsize=4, zorder=1)
    plt.scatter(z[mask], rho[mask], c='red', label='Bad Data Points', alpha=0.7, edgecolor='black', s=25, zorder=2)
    plt.plot(x_arr, y_arr_2, label='Good Function')

    A = - (result[3] - result[1]) / (result[2] - result[0])
    a.append(A)







        

plt.figure(main_fig.number)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.title('Fitting for x with Bad data in dataset', fontsize=9)
plt.legend(fontsize=3, loc='best')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('x', fontsize=7)
plt.ylabel('y', fontsize=7)
main_fig.set_size_inches(4, 2.5)

plt.xlim(zmin, zmax)
plt.ylim(-11, -3)
# plt.ylim(-14, -2)
plt.legend()
plt.grid()
plt.savefig(f"check2_plot_2_{current_time}.png")  # Save the plot as a PNG file with the current time

print("\na = ", a)
print()
end_time = datetime.datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")


plt.show()
plt.close()
