import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import sys
from numpy.polynomial.chebyshev import chebfit
from numpy.polynomial import Chebyshev as T
import time

zmin = 0
zmax = 15

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
        return -np.inf
    return - 100 * Pb

def Vb_penalise(Vb):
    """
    Penalise the Vb parameter.
    """
    if Vb <= 0:
        return -np.inf
    return -np.log10(Vb)

def Yb_penalise(Yb):
    """
    Penalise the Yb parameter.
    """
    if np.abs(Yb) > 1000:
        return -np.inf
    return 0

def function_penalise(params, NUM):
    """
    Penalise the function parameters.
    """
    func_params = params[:NUM]
    # print("func_params = ", func_params)
    if np.any(np.abs(func_params) > 50):
        return -np.inf

    sum = 0
    # for i in range(NUM):
    #     if i == 0 and func_params[i] > 0:
    #         return -np.inf
    #     sum -= 2**-i * func_params[i] * 10

    for i in range(NUM):
        if i == 0 and not (-0.5 < func_params[i] < 0):
            return -np.inf
        if i == 0 or i == NUM - 1:
            sum -= np.log(1 +  np.exp(func_params[i]))
        else:
            sum -= 0

    return sum


def plot_penalaty_function(NUM, ch):
    """
    Plot the penalty function for the parameters.
    """
    if ch == 1:
        x = np.linspace(-2, 2, 10000)
        # y1 = function_penalise([x] * NUM, NUM)

        y2 = [Pb_penalise(x[i]) for i in range(len(x))]
        y3 = [Vb_penalise(x[i]) for i in range(len(x))]
        y4 = [Yb_penalise(x[i]) for i in range(len(x))]

        print("x = ", x)
        print("y2 = ", y2)
        print("y3 = ", y3)
        print("y4 = ", y4)

        print()
        print("len(x) = ", len(x))
        print("len(y2) = ", len(y2))
        print("len(y3) = ", len(y3))
        print("len(y4) = ", len(y4))
        # y2 = Pb_penalise(x)
        # y3 = Vb_penalise(x)
        # y4 = Yb_penalise(x)
        # plt.plot(x, y1, label='Function Penalise')
        plt.plot(x, y2, label='Pb Penalise', linewidth=5)
        plt.plot(x, y3, label='Vb Penalise')
        plt.plot(x, y4, label='Yb Penalise')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        # plt.ylim(-10, 10)
        plt.xlabel('x')
        plt.ylabel('Penalty')
        plt.title('Penalty Function')
        plt.legend()
        plt.grid()
        plt.show()

        plt.close('all')

    if ch == 2:
        x = np.linspace(-50, 5, 100)
        y1 = [function_penalise([x[i], 0, 0], NUM) for i in range(len(x))]
        y2 = [function_penalise([0, x[i], 0], NUM) for i in range(len(x))]
        y3 = [function_penalise([0, 0, x[i]], NUM) for i in range(len(x))]

        print("\nx = ", x)
        print("\ny1 = ", y1)
        print("\ny2 = ", y2)
        print("\ny3 = ", y3)
        print()
        print("len(x) = ", len(x))
        print("len(y1) = ", len(y1))
        print("len(y2) = ", len(y2))
        print("len(y3) = ", len(y3))

        plt.plot(x, y1, label='a2')
        plt.plot(x, y2, label='a1')
        # plt.plot(x, y3, label='a0')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.ylim(-10, 10)
        plt.xlabel('x')
        plt.ylabel('Penalty')
        plt.title('Penalty Function')
        plt.legend()
        plt.grid()
        plt.show()



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

    return Pb_penalise(Pb) + Vb_penalise(Vb) + Yb_penalise(Yb) + function_penalise(params, NUM)


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

    if Pb < 0 or Pb > 1:
        return -np.inf
    if Vb <= 0:
        return -np.inf
    
    rho_sigma = (rho_up + rho_low) / 2
    return np.sum(np.log10((1 - Pb) / np.sqrt(rho_sigma**2) * np.exp(-0.5 * ((rho - function(z, func_params)) / rho_sigma)**2) + Pb / np.sqrt(Vb + rho_sigma**2) * np.exp(-0.5 * ((rho - Yb)**2 / (Vb + rho_sigma**2)))))


def logposterior(params, NUM, z, rho, rho_up, rho_low):
    lp = logprior(params, NUM)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = loglikelihood(params, NUM, z, rho, rho_up, rho_low)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_mcmc(NUM, z, rho, rho_up, rho_low, nwalkers=100, nsteps_burn=2, nsteps_prod=5000):
    """
    Run MCMC to sample the posterior distribution.
    """
    # m, b, Pb, Yb, Vb
    ndim = 3 + NUM

    p0 = np.empty((nwalkers, ndim))

    p0[:, :NUM] = np.random.uniform(-5, 5, size=(nwalkers, NUM))  # function parameters
    p0[:, NUM] = np.random.uniform(0, 1, size=nwalkers)  # Pb
    p0[:, NUM + 1] = np.random.uniform(-10, 10, size=nwalkers)  # Yb
    p0[:, NUM + 2] = np.random.uniform(0, 10, size=nwalkers)  # Vb


    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(NUM, z, rho, rho_up, rho_low))

    # Run the MCMC
    sampler.run_mcmc(p0, nsteps_burn, progress=True)
    # Discard the burn-in samples
    sampler.reset()

    sampler.run_mcmc(None, nsteps_prod, progress=True)

    return sampler

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



def modelling_using_David_Hogg_with_uncertainty_in_1D(i, NUM, z, rho, rho_up, rho_low):
    """
    Main function to run the MCMC and plot results for a subarray.
    """
    # Run MCMC
    sampler = run_mcmc(NUM, z, rho, rho_up, rho_low)

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
    # rho_up = np.log10(rho + rho_up) - rho
    # rho_low = np.log10(rho - rho_low) - rho

    # print("z = ", z)
    # print("rho = ", rho)
    # print("rho_up = ", rho_up)
    # print("rho_low = ", rho_low)
    # print("length of z = ", len(z))
    # print("length of rho = ", len(rho))
    # print("length of rho_up = ", len(rho_up))

    # print("logrho = ", logrho)
    # print("logrho_up = ", logrho_up)
    # print("logrho_low = ", logrho_low)
    # print("length of logrho = ", len(logrho))
    # print("length of logrho_up = ", len(logrho_up))
    # print("length of logrho_low = ", len(logrho_low))

    # import sys
    # sys.exit(0)

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
    sampler, samples = modelling_using_David_Hogg_with_uncertainty_in_1D(i, 3, z, logrho, logrho_up, logrho_low)

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

    # print("sigys = ", sigys)
    # print("sigy = ", sigy)

    plt.scatter(x, y, c='red', label='Data Points', edgecolor='black')
    plt.errorbar(x, y, yerr=sigy, fmt='o', label='Error bars', alpha=0.6, capsize=5)
    plt.xlabel('z')
    plt.ylabel('rho')
    plt.title('Fitting for rho with Bad data in dataset')

    x_arr = np.linspace(xmin, xmax, 100)
    y_arr = function(x_arr, parameters)

    plt.plot(x_arr, y_arr, color='black', label='MAP Line')
    plt.xlim(xmin, xmax)
    plt.ylim(-12, 1)

  

for i in range(len(sampler_list)):
    # Get the MAP parameters
    map_params = find_MAP(sampler, NUM=3)
    print("MAP Parameters:", map_params)

    # Plot the data with Gaussian uncertainty for each segment
    plot_data(scatter_z[i], scatter_rho[i], map_params, zmin, zmax, [logrho_up_list[i], logrho_low_list[i]])
# Example usage of find_MAP


filename = "rhoqso_different_final_plot"

if filename:
    plt.savefig(filename + ".pdf", bbox_inches='tight')
    plt.savefig(filename + ".png", bbox_inches='tight')
# plt.show()
print(f"Plot saved as {filename}.pdf and {filename}.png")