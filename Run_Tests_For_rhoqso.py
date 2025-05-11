import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import sys
from numpy.polynomial.chebyshev import chebfit
from numpy.polynomial import Chebyshev as T
import time
import random

from emcee.moves import StretchMove


zmin = 0
zmax = 15

steps_percentage = 0

####################################################################################
#### Even the function logrho = A * z ^ (- B - C * log(z)) - D works!!!!        ####
####                        It works better!!!                                  #### 
####            A, C and D strictly positive, B can be anything!                ####
#### Or Double Power law should also work!!!                                    ####
####################################################################################

def function(x, parameters):
    """
    Polynomial function in x.
    """
    return np.polyval(parameters, x)

# parameters = [1, 100] 
# x = np.linspace(0, 10, 100)
# y = function(x, parameters)

# plt.plot(x, y, label='Polynomial Fit')
# plt.plot(x, np.zeros_like(x))
# plt.show(block=False)

def Pb_penalise(Pb):
    """
    Penalise the Pb parameter.
    """
    if Pb < 0 or Pb > 1:
        # print("Rejected Pb = ", Pb)
        # print("Returned -inf")
        return -np.inf
    return - np.log(Pb + 1)

def Vb_penalise(Vb):
    """
    Penalise the Vb parameter.
    """
    if 0 < Vb < 3:
        return - Vb
    
    # print("Rejected Vb = ", Vb)
    # print("Returned -inf")
    return -np.inf

def Yb_penalise(Yb):
    """
    Penalise the Yb parameter.
    """
    if np.abs(Yb) > 50:
        # print("Rejected Yb = ", Yb)
        # print("Returned -inf")
        return -np.inf
    return 0

def function_penalise(params, NUM):
    """
    Penalise the function parameters.
    """
    func_params = params[:NUM]
    # print("func_params = ", func_params)
    if np.any(np.abs(func_params) > 50):
        # print("func_params = ", func_params)
        # print("Rejected func_params = ", func_params)
        # print("Returned -inf")
        return -np.inf

    sum = 0

    for i in range(NUM):
        # print("i = ", i)
        # print("func_params[i] = ", func_params[i])
        if i == 0 and not (-1.5 < func_params[i] < 0):
            # print("Rejected func_params[0] = ", func_params[i])
            # print("Returned -inf")
            return -np.inf
        if i == NUM - 1:
            if not (-50 < func_params[i] < 0):
                # print("Rejected func_params[NUM-1] = ", func_params[i])
                # print("Returned -inf")
                return -np.inf
            sum -= np.log(1000 + func_params[i])
        else:
            if not (-20 < func_params[i] < 20):
                # print("Rejected func_params[i] = ", func_params[i])
                # print("Returned -inf")
                return -np.inf

    return sum


# def plot_penalaty_function(NUM, ch):
#     """
#     Plot the penalty function for the parameters.
#     """
#     if ch == 1:
#         x = np.linspace(-2, 2, 10000)
#         # y1 = function_penalise([x] * NUM, NUM)

#         y2 = [Pb_penalise(x[i]) for i in range(len(x))]
#         y3 = [Vb_penalise(x[i]) for i in range(len(x))]
#         y4 = [Yb_penalise(x[i]) for i in range(len(x))]

#         print("x = ", x)
#         print("y2 = ", y2)
#         print("y3 = ", y3)
#         print("y4 = ", y4)

#         print()
#         print("len(x) = ", len(x))
#         print("len(y2) = ", len(y2))
#         print("len(y3) = ", len(y3))
#         print("len(y4) = ", len(y4))
#         # y2 = Pb_penalise(x)
#         # y3 = Vb_penalise(x)
#         # y4 = Yb_penalise(x)
#         # plt.plot(x, y1, label='Function Penalise')
#         plt.plot(x, y2, label='Pb Penalise', linewidth=5)
#         plt.plot(x, y3, label='Vb Penalise')
#         plt.plot(x, y4, label='Yb Penalise')
#         plt.axhline(0, color='black', linestyle='--')
#         plt.axvline(0, color='black', linestyle='--')
#         # plt.ylim(-10, 10)
#         plt.xlabel('x')
#         plt.ylabel('Penalty')
#         plt.title('Penalty Function')
#         plt.legend()
#         plt.grid()
#         plt.show()

#         plt.close('all')

#     if ch == 2:
#         x = np.linspace(-50, 5, 100)
#         y1 = [function_penalise([x[i], 0, 0], NUM) for i in range(len(x))]
#         y2 = [function_penalise([0, x[i], 0], NUM) for i in range(len(x))]
#         y3 = [function_penalise([0, 0, x[i]], NUM) for i in range(len(x))]

#         print("\nx = ", x)
#         print("\ny1 = ", y1)
#         print("\ny2 = ", y2)
#         print("\ny3 = ", y3)
#         print()
#         print("len(x) = ", len(x))
#         print("len(y1) = ", len(y1))
#         print("len(y2) = ", len(y2))
#         print("len(y3) = ", len(y3))

#         plt.plot(x, y1, label='a2')
#         plt.plot(x, y2, label='a1')
#         # plt.plot(x, y3, label='a0')
#         plt.axhline(0, color='black', linestyle='--')
#         plt.axvline(0, color='black', linestyle='--')
#         plt.ylim(-10, 10)
#         plt.xlabel('x')
#         plt.ylabel('Penalty')
#         plt.title('Penalty Function')
#         plt.legend()
#         plt.grid()
#         plt.show()



# plot_penalaty_function(3, ch=2)

# import sys
# sys.exit(0)

def logprior(params, NUM):
    """
    Log-prior function for the parameters.
    """
    func_params = params[:NUM]
    # print("func_params = ", func_params)
    Pb, Yb, Vb = params[NUM:]
    # print("Pb = ", Pb)
    # print("Yb = ", Yb)
    # print("Vb = ", Vb)

    # if 0 <= Pb <= 0.75 and 0 < Vb < 10:
    #     if np.all(np.abs(func_params) < 50) and 0 <= np.abs(Yb) <= 1000:
    #         return - 100 * Pb - np.log10(Vb) 
    # return -np.inf  # Reject everything else

    Pb_penalty = Pb_penalise(Pb)
    Vb_penalty = Vb_penalise(Vb)
    Yb_penalty = Yb_penalise(Yb)
    func_penalty = function_penalise(params, NUM)

    # print("\tPb_penalty = ", Pb_penalty)
    # print("Vb_penalty = ", Vb_penalty)
    # print("Yb_penalty = ", Yb_penalty)
    # print("func_penalty = ", func_penalty)
    if Pb_penalty == -np.inf or Vb_penalty == -np.inf or Yb_penalty == -np.inf or func_penalty == -np.inf:
        # print("Rejected params = ", params)
        # print("Returned -inf")
        return -np.inf

    return Pb_penalise(Pb) + Vb_penalise(Vb) + Yb_penalise(Yb) + function_penalise(params, NUM)

def scale(steps_percentage):
    """
    Scale the uncertainty based on the number of steps.
    """
    # print("steps_percentage = ", steps_percentage)
    if steps_percentage < 0.25:
        return 10
    elif steps_percentage < 0.55:
        return 5
    elif steps_percentage < 0.75:
        return 2
    elif steps_percentage < 0.85:
        return 1.5
    return 1

    

def loglikelihood(params, NUM, z, rho, rho_up, rho_low):
    """
    Log-likelihood function for the parameters.
    """
    # Assuming Gaussian likelihood
    # Assuming a simple linear model for demonstration
    # You can replace this with the actual model you want to fit
    func_params = params[:NUM]
    # print("func_params = ", func_params)
    Pb, Yb, Vb = params[NUM:]
    # print("Pb = ", Pb)
    # print("Yb = ", Yb)
    # print("Vb = ", Vb)
    
    rho_sigma = (rho_up + rho_low) / 2 * scale(steps_percentage)

    logforeground_model = np.log((1 / np.sqrt(2 * np.pi * rho_sigma**2))) + (-0.5 * ((rho - function(z, func_params)) / rho_sigma)**2)
    logbackground_model = np.log((1 / np.sqrt(2 * np.pi * (Vb + rho_sigma**2)))) + (-0.5 * ((rho - Yb)**2 / (Vb + rho_sigma**2)))
    # print("\nrho[0] = ", rho[0])
    # print("function(z[0], func_params) = ", function(z[0], func_params))
    # print("rho_sigma[0] = ", rho_sigma[0])
    # print("Yb = ", Yb)
    # print("Vb = ", Vb)
    # print("residual1 = ", rho[0] - function(z[0], func_params))
    # print("residual2 = ", rho[0] - Yb)
    # print("logforeground_model[0] = ", logforeground_model[0])
    if np.all(logforeground_model) == 0:
        print("All foreground_model values are zero.")
        return -np.inf
    
    if np.all(logbackground_model) == 0:
        print("All background_model values are zero.")
        return -np.inf
    if np.any(np.isnan(logforeground_model)) or np.any(np.isnan(logbackground_model)):
        print("NaN values in foreground_model or background_model.")
        return -np.inf

    rand_seed = random.random()
    # rand_seed = 0
    # print("rand_seed < 0.0001 = ", rand_seed < 0.00001)
    if rand_seed < 0.0001:
        print("\n\n\na2 = ", func_params[0])
        print("a1 = ", func_params[1])
        print("a0 = ", func_params[2])
        print("function(z, func_params) = ", function(z, func_params))
        print("rho = ", rho)
        print("\n\nPb = ", Pb)
        print("Yb = ", Yb)
        print("Vb = ", Vb)
        print("\nforeground_model = ", logforeground_model)
        print("rho = ", rho)
        print("function(z, func_params) = ", function(z, func_params))
        print("rho_sigma = ", rho_sigma)
        print("rho - function(z, func_params) = ", rho - function(z, func_params))
        print("rho - Yb = ", rho - Yb)
        print("(rho - function(z, func_params)) / rho_sigma = ", (rho - function(z, func_params)) / rho_sigma)
        print("(rho - Yb) / (Vb + rho_sigma) = ", (rho - Yb) / (Vb + rho_sigma))
        print("background_model = ", logbackground_model)
        ratio = np.abs(((rho - function(z, func_params)) / rho_sigma)/((rho - Yb) / (Vb + rho_sigma)))
        print("\n\nratio = ", ratio)
        ratio2 = logforeground_model / logbackground_model
        print("ratio2 = ", ratio2)

    # import sys
    # sys.exit(0)

    # np.sum(np.log10((1 - Pb) / np.sqrt(rho_sigma**2) * np.exp(-0.5 * ((rho - function(z, func_params)) / rho_sigma)**2) +
    #                  Pb / np.sqrt(Vb + rho_sigma**2) * np.exp(-0.5 * ((rho - Yb)**2 / (Vb + rho_sigma**2)))))

    a = np.log(1 - Pb) + logforeground_model
    b = np.log(Pb) + logbackground_model

    log10L = np.sum(np.logaddexp(a, b)) / np.log(10)
    # print("logL = ", logL)

    return log10L





def logposterior(params, NUM, z, rho, rho_up, rho_low):
    lp = logprior(params, NUM)
    if not np.isfinite(lp):
        # print("lp = ", lp)
        # print("params = ", params)
        # print("\n\n\n")
        # import sys
        # sys.exit(0)
        return -np.inf
    
    # import sys
    # sys.exit(2)
    
    ll = loglikelihood(params, NUM, z, rho, rho_up, rho_low)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_mcmc(NUM, z, rho, rho_up, rho_low, nwalkers=100, nsteps_burn=2000, nsteps_prod=1000, fittedCoeff=None):
    """
    Run MCMC to sample the posterior distribution.
    """
    # m, b, Pb, Yb, Vb
    ndim = 3 + NUM

    p0 = np.empty((nwalkers, ndim))

    if fittedCoeff is not None:
        epsilon = 0.01  # You can set this value as needed
        print("fittedCoeff = ", fittedCoeff)
        p0[:, :NUM] = np.random.uniform(fittedCoeff[:NUM] - epsilon, fittedCoeff[:NUM] + epsilon, size=(nwalkers, NUM))  # a2, a1, a0
        print("p0[:, :NUM] from fittedCoeff = ", p0[:, :NUM])

    else:
        p0[:, 0] = np.random.uniform(-0.5, 0, size=nwalkers)  # a2
        # print("p0[:, 0] from else = ", p0[:, 0])
        p0[:, 1:NUM] = np.random.uniform(-1, 0, size=(nwalkers, NUM-1))  # other function parameters
    p0[:, NUM] = np.random.uniform(0.2, 0.8, size=nwalkers)  # Pb
    p0[:, NUM + 1] = np.random.uniform(-10, 10, size=nwalkers)  # Yb
    p0[:, NUM + 2] = np.random.uniform(0.05, 2.5, size=nwalkers)  # Vb


    # Set up the sampler
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(NUM, z, rho, rho_up, rho_low))
    StretchMove(a=0.25)
    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, logposterior, args=(NUM, z, rho, rho_up, rho_low), moves=StretchMove()
)

    # # Run the MCMC
    # sampler.run_mcmc(p0, nsteps_burn, progress=True)
    # # Discard the burn-in samples
    # sampler.reset()

    # sampler.run_mcmc(None, nsteps_prod, progress=True)

    global steps_percentage

    for i in range(nsteps_burn):
        # print(f"Burn-in step {i + 1}/{nsteps_burn}")
        sampler.run_mcmc(p0, 1)

        steps_percentage += 1/ max(100, nsteps_burn)
        # print("p0 = ", p0)
        # print("sampler = ", sampler)

    sampler.reset()
    sampler.run_mcmc(None, nsteps_prod, progress=True)

    return sampler

def find_bad_data(samples, z, rho, rho_up, rho_low, function):
    Npoints = len(z)
    Nsamps = len(samples)
    probs = np.zeros((Nsamps, Npoints))

    rho_sigma = (rho_up + rho_low) / 2

    for s, params in enumerate(samples):
        func_params = params[:-3]
        Pb, Yb, Vb = params[-3:]

        model_vals = function(z, func_params)

        p_fg = (1 / np.sqrt(2 * np.pi * rho_sigma**2)) * np.exp(-0.5 * ((rho - model_vals) / rho_sigma)**2)
        p_bg = (1 / np.sqrt(2 * np.pi * (Vb + rho_sigma**2))) * np.exp(-0.5 * ((rho - Yb)**2 / (Vb + rho_sigma**2)))

        numerator = Pb * p_bg
        denominator = (1 - Pb) * p_fg + Pb * p_bg

        if random.random() < 0.00001:

            print("Pb = ", Pb)
            print("p_fg = ", p_fg)
            print("p_bg = ", p_bg)

            print("numerator = ", numerator)
            print("denominator = ", denominator)
            print("numerator / denominator = ", numerator / denominator)

            print("mask = ", numerator / denominator > 0.5)


            residuals = rho - model_vals
            print(residuals / rho_sigma)  # This should be ~0 if model fits well

        probs[s, :] = numerator / denominator

    return np.mean(probs, axis=0)


def compute_corner_statistics(samples):
    """
    Compute the median and 1\\σ credible intervals for each parameter.
    """
    stats = {}
    for i in range(samples.shape[1]):  # Loop over each parameter
        param_samples = samples[:, i]
        median = np.percentile(param_samples, 50)  # Median
        lower = np.percentile(param_samples, 16)  # 16th percentile
        upper = np.percentile(param_samples, 84)  # 84th percentile
        stats[f"Param {i+1}"] = {
            "median": median,
            "+1\\σ": upper - median,
            "-1\\σ": median - lower,
        }
    return stats

def marginalize_and_reproduce_function(samples, NUM, zlims):
    z = np.linspace(zlims[0], zlims[1], 100)

    func_params = samples[:, :NUM]
    marginalized_params = np.median(func_params, axis=0)
    stats = compute_corner_statistics(func_params)
    print("stats = ", stats)
    marginalized_params = [stat["median"] for stat in stats.values()]
    print("\n\n****************************************************************************\n\nmarginalized_params = ", marginalized_params)
    print("\n\n****************************************************************************\n\n")

    return z, function(z, marginalized_params)



def modelling_using_David_Hogg_with_uncertainty_in_1D(i, NUM, z, rho, rho_up, rho_low, fittedCoeff=None):
    """
    Main function to run the MCMC and plot results for a subarray.
    """
    # Run MCMC
    sampler = run_mcmc(NUM, z, rho, rho_up, rho_low, fittedCoeff=fittedCoeff)

    # Flatten the chain and remove burn-in samples
    samples = sampler.get_chain(flat=True)

    labels = ['a2', 'a1', 'a0', 'Pb', 'Yb', 'Vb']

    # Corner plot for the subarray
    corner_fig = plt.figure(figsize=(12, 8))
    corner.corner(samples, fig=corner_fig, labels=labels, show_titles=True)
    plt.savefig(f'rhoqso_corner_plot_{i}.png', bbox_inches='tight')
    plt.savefig(f'rhoqso_corner_plot_{i}.pdf', bbox_inches='tight')
    plt.show(block=False)

    # Chain plot for the subarray
    ndim = len(labels)
    chains_fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, sampler.nwalkers))
    for j in range(ndim):
        ax = axes[j]
        # ax.plot(sampler.get_chain()[:, :, j], color=colors[j % len(colors)], alpha=0.3)
        ax.plot(sampler.get_chain()[:, :, j], alpha=0.5)
        ax.set_xlim(0, len(sampler.get_chain()))
        ax.set_ylabel(labels[j])
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.savefig(f'rhoqso_chains_plot_{i}.png', bbox_inches='tight')
    plt.savefig(f'rhoqso_chains_plot_{i}.pdf', bbox_inches='tight')
    plt.show(block=False)

    print("\n\n\n")
    print("Acceptance fraction:", sampler.acceptance_fraction)

    # Check autocorrelation time
    try:
        print("Autocorrelation time:", sampler.get_autocorr_time())
    except Exception as e:
        print("Error calculating autocorrelation time:", e)
    print("\n\n\n")

    # Return the samples for further analysis
    return sampler, samples


def plot_results(samples, main_fig, NUM, z, rho, rho_up, rho_low):
    """
    Plot the results of the MCMC simulation on the main figure.
    """
    plt.figure(main_fig.number)  # Switch to the main figure
    ax = main_fig.gca()

    # Scatter plot of the data
    ax.scatter(z, rho, label='Data', alpha=0.6)
    # ax.errorbar(z, rho, yerr=[rho_up, rho_low], fmt='o', label='Error bars', alpha=0.6)

    # Marginalize and reproduce the function
    z_func, func = marginalize_and_reproduce_function(samples, NUM, (zmin, zmax))
    # print("z_func = ", z_func)
    # print("func = ", func)
    ax.plot(z_func, func, linewidth=2)

    return z, rho, z_func, func


# Main loop for processing subarrays
input_data_file = 'rhoqso_output_data.txt'

all_data = []
with open(input_data_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        data = line.strip().split(',')
        all_data.append(data)

all_data = np.array(all_data).astype(float)

# Find breakpoints in the data
breaks = np.where(np.diff(all_data[:, 0]) <= 0)[0] + 1
print("Breakpoints:", breaks)

# Split the data into subarrays
subarrays = np.split(all_data, breaks)

# Create the main figure for collecting all samples
main_fig = plt.figure(figsize=(12, 8))

scatter_z = []
scatter_rho = []

function_z = []
function_rho = []

sampler_list = []
logrho_up_list = []
logrho_low_list = []

for i, subarray in enumerate(subarrays):
    print(f"\nProcessing Subarray {i + 1}:")
    z, uzerr, lzerr, rho, rho_up, rho_low = [x.flatten() for x in np.split(subarray, 6, axis=1)]

    logrho = np.log10(rho)
    logrho_up = 1 / (np.log(10) * rho) * rho_up
    logrho_low = 1 / (np.log(10) * rho) * rho_low

    print("z = ", z)
    print("rho = ", rho)
    print("rho_up = ", rho_up)
    print("rho_low = ", rho_low)
    print("length of z = ", len(z))
    print("length of rho = ", len(rho))
    print("length of rho_up = ", len(rho_up))

    print("logrho = ", logrho)
    print("logrho_up = ", logrho_up)
    print("logrho_low = ", logrho_low)
    print("length of logrho = ", len(logrho))
    print("length of logrho_up = ", len(logrho_up))
    print("length of logrho_low = ", len(logrho_low))

    # import sys
    # sys.exit(0)
    print("\n\n\n")

    # Fit a polynomial of the desired order to the data
    order = 2  # Change this to control the order of the polynomial
    coefficients = np.polyfit(z, logrho, order)

    # Print the coefficients of the best-fit polynomial
    print(f"\n\n\n\t\t\tBest-fit polynomial coefficients (order {order}): {coefficients}\n\n\n")

    # Evaluate the polynomial at the given z values
    fitted_values = np.polyval(coefficients, z)

    # Plot the original data and the best-fit polynomial
    # plt.figure(figsize=(10, 6))
    # plt.scatter(z, np.log10(rho), label='Data', color='blue', alpha=0.6)
    # plt.plot(z, fitted_values, label=f'Best-fit Polynomial (order {order})', color='red', linewidth=2)
    # plt.xlabel('z')
    # plt.ylabel('log(rho)')
    # plt.title(f'Best-fit Polynomial of Order {order}')
    # plt.legend()
    # plt.grid()
    # plt.show(block=False)

    # Run MCMC and plot corner and chain plots for the subarray
    sampler, samples = modelling_using_David_Hogg_with_uncertainty_in_1D(i, 3, z, logrho, logrho_up, logrho_low, fittedCoeff=coefficients)

    # Plot the results on the main figure
    sz, sr, fz, fr = plot_results(samples, main_fig, 3, z, logrho, logrho_up, logrho_low)

    scatter_z.append(sz)
    scatter_rho.append(sr)
    function_z.append(fz)
    function_rho.append(fr)
    sampler_list.append(sampler)
    logrho_up_list.append(logrho_up)
    logrho_low_list.append(logrho_low)



# Save and show the main figure
plt.figure(main_fig.number)
ax = main_fig.gca()
ax.legend()
ax.set_xlim(zmin, zmax)
ax.set_ylim(-12, 1)
ax.set_xlabel('z')
ax.set_ylabel('log(rho)')
plt.savefig('rhoqso_final_plot.png', bbox_inches='tight')
plt.savefig('rhoqso_final_plot.pdf', bbox_inches='tight')
plt.show(block=False)


plt.close('all')
print("All subarrays processed and results plotted.")

for i in range(len(scatter_z)):
    plt.plot(scatter_z[i], scatter_rho[i], 'o', label=f'Subarray {i + 1} Data', alpha=0.6)
    plt.plot(function_z[i], function_rho[i], label=f'Subarray {i + 1} Function', linewidth=2)
plt.xlabel('z')
plt.ylabel('log(rho)')
plt.xlim(zmin, zmax)
plt.ylim(-22, 1)
plt.legend()
plt.savefig(f'rhoqso_plot.png', bbox_inches='tight')
plt.savefig(f'rhoqso_plot.pdf', bbox_inches='tight')
plt.show(block=False)



def find_MAP(sampler, NUM):
    """
    Find the maximum a posteriori (MAP) estimate of the parameters, 
    after marginalising over the variance parameters.
    """
    print("In find_MAP")
    samples = sampler.get_chain(flat=True)
    print(f"Number of samples: {len(samples)}")

    # Extract the first NUM parameters from the samples
    samples_reduced = samples[:, :NUM]

    # Define the number of bins for the histogram
    bin_count = 50
    bins = [bin_count] * NUM  # Only for the first NUM dimensions

    # Compute the NUM-dimensional histogram for the first NUM parameters
    hist, edges = np.histogramdd(samples_reduced, bins=bins)

    # Find the index of the maximum value in the marginal distribution
    max_index = np.unravel_index(np.argmax(hist), hist.shape)
    print("Index of maximum value in the marginal distribution:")
    print(max_index)

    # Get the corresponding parameter values
    param_values = [edges[i][max_index[i]] for i in range(len(max_index))]
    print("Parameter values corresponding to the maximum value:")
    print(param_values)

    # Return the parameter values
    return param_values

def plot_data(x, y, parameters, xmin, xmax, sigys):
    """
    Plot the data with Gaussian uncertainty for each segment and the MAP line.
    """
    # print("Inside plot_data")

    sigy = (np.abs(sigys[0]) + np.abs(sigys[1])) / 2

    prob_bad_points = find_bad_data(samples, x, y, sigys[0], sigy[1], function)
    print("prob_bad_points = ", prob_bad_points)

    mask = prob_bad_points > 0.5
    print("mask = ", mask)

    # print("sigys = ", sigys)
    # print("sigy = ", sigy)

    plt.scatter(x, y, c='red', label='Data Points', edgecolor='black')
    plt.scatter(x[mask], y[mask], facecolors='none', edgecolors='red', s=100, label='Bad Data Points')
    plt.errorbar(x, y, yerr=sigy, fmt='o', alpha=0.6, capsize=5)
    plt.xlabel('z')
    plt.ylabel('rho')
    plt.title('Fitting for rho with Bad data in dataset')

    x_arr = np.linspace(xmin, xmax, 100)
    y_arr = function(x_arr, parameters)

    plt.plot(x_arr, y_arr, color='black', label='MAP Line')
    plt.xlim(xmin, xmax)
    plt.ylim(-22, 1)

  

for i in range(len(sampler_list)):
    # Get the MAP parameters
    map_params = find_MAP(sampler, NUM=3)
    print("\n\n\n\n\n\n\n\n\nMAP Parameters:", map_params)

    # Plot the data with Gaussian uncertainty for each segment
    plot_data(scatter_z[i], scatter_rho[i], map_params, zmin, zmax, [logrho_up_list[i], logrho_low_list[i]])
# Example usage of find_MAP


filename = "rhoqso_different_final_plot"
plt.legend()


if filename:
    plt.savefig(filename + ".pdf", bbox_inches='tight')
    plt.savefig(filename + ".png", bbox_inches='tight')
# plt.show()
print(f"Plot saved as {filename}.pdf and {filename}.png")