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

    if 0 <= Pb <= 0.75 and Vb > 0:
        if np.all(np.abs(func_params) < 50) and 0 <= Yb <= 1000:
            return -np.log10(1 + Pb) - np.log10(1 + Vb) 
    return -np.inf  # Reject everything else


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

def run_mcmc(NUM, z, rho, rho_up, rho_low, nwalkers=100, nsteps_burn=200, nsteps_prod=20000):
    """
    Run MCMC to sample the posterior distribution.
    """
    # m, b, Pb, Yb, Vb
    ndim = 3 + NUM

    p0 = np.empty((nwalkers, ndim))

    p0[:, :NUM] = np.random.uniform(-5, 5, size=(nwalkers, NUM))  # function parameters
    p0[:, NUM] = np.random.uniform(0, 0.5, size=nwalkers)  # Pb
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


# def plot_results(samples):
#     """
#     Plot the results of the MCMC simulation.
#     """

#     H, xedges, yedges = np.histogram2d(samples[:,1], samples[:,0], bins=500)
#     i,j = np.unravel_index(np.argmax(H), H.shape)
#     b_map = 0.5*(xedges[i]+xedges[i+1])
#     m_map = 0.5*(yedges[j]+yedges[j+1])

#     print("MAP of m:", m_map)
#     print("MAP of b:", b_map)
#     print()

#     H_normalized = H / np.max(H)
#     plt.pcolormesh(xedges, yedges, H_normalized.T, cmap="Greys")

#     H_flat = H.flatten()
#     H_sorted = np.sort(H_flat)
#     cumsum = np.cumsum(H_sorted)
#     cumsum /= cumsum[-1]  # Normalize to [0, 1]

#     # Define percentiles (e.g., 68%, 95%, 99%)
#     levels = [0.25, 0.5, 0.75]
#     contour_levels = sorted(set([H_sorted[np.searchsorted(cumsum, level)] for level in levels]))

#     # Create a meshgrid for contour plotting
#     X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
#     plt.contour(X, Y, H.T, levels=contour_levels, colors="black")

#     # Add labels and title
#     plt.xlabel("b")
#     plt.ylabel("m")
#     plt.xlim(0, 10)
#     plt.ylim()
#     plt.title("2D Histogram with Contours and Density Shading")
#     plt.colorbar(label="Normalized Density")
#     plt.savefig('rhoqso_histogram.png', bbox_inches='tight')
#     plt.savefig('rhoqso_histogram.pdf', bbox_inches='tight')
#     plt.show(block=False)
#     return samples

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
        ax.plot(sampler.get_chain()[:, :, j], color=colors[j % len(colors)], alpha=0.3)
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
    return samples


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

for i, subarray in enumerate(subarrays):
    print(f"\nProcessing Subarray {i + 1}:")
    z, uzerr, lzerr, rho, rho_up, rho_low = [x.flatten() for x in np.split(subarray, 6, axis=1)]

    rho = np.log10(rho)
    rho_up = np.log10(rho_up)
    rho_low = np.log10(rho_low)
    # rho_up = np.log10(rho + rho_up) - rho
    # rho_low = np.log10(rho - rho_low) - rho

    # Fit a polynomial of the desired order to the data
    order = 2  # Change this to control the order of the polynomial
    coefficients = np.polyfit(z, rho, order)

    # Print the coefficients of the best-fit polynomial
    print(f"Best-fit polynomial coefficients (order {order}): {coefficients}")

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
    samples = modelling_using_David_Hogg_with_uncertainty_in_1D(i, 3, z, rho, rho_up, rho_low)

    # Plot the results on the main figure
    sz, sr, fz, fr = plot_results(samples, main_fig, 3, z, rho, rho_up, rho_low)

    scatter_z.append(sz)
    scatter_rho.append(sr)
    function_z.append(fz)
    function_rho.append(fr)

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

# def plot_results(samples, fig, NUM, z, rho, rho_up, rho_low):
#     """
#     Plot the results of the MCMC simulation.
#     """
#     plt.figure(main_fig.number)
#     print("Plotting results...")
#     print("z = ", z)
#     print("rho = ", rho)
#     print("rho_up = ", rho_up)
#     print("rho_low = ", rho_low)
#     print()
#     print("length of z = ", len(z))
#     print("length of rho = ", len(rho))
#     print("length of rho_up = ", len(rho_up))
#     print("length of rho_low = ", len(rho_low))
#     print()
#     plt.scatter(z, rho, 'o', label='Data', fig=fig)
#     plt.errorbar(z, rho, yerr=[rho_up, rho_low], fmt='o', label='Error bars', fig=fig)
#     plt.xlabel('z')
#     plt.ylabel('log(rho)')

#     z, func = marginalize_and_reproduce_function(samples, NUM, (zmin, zmax))
#     plt.plot(z, func, label='Marginalized function', fig=fig)
#     plt.legend()
#     plt.xlim(zmin, zmax)
#     plt.ylim(-12, 1)




# def modelling_using_David_Hogg_with_uncertainity_in_1D(i, NUM, z, rho, rho_up, rho_low):
#     """
#     Main function to run the MCMC and plot results.

#     Add 2D later on, using Exercise 14
#     """
#     # Run MCMC
#     sampler = run_mcmc(NUM, z, rho, rho_up, rho_low)

#     # Flatten the chain and remove burn-in samples
#     samples = sampler.get_chain(flat=True)

#     labels = ['a2', 'a1', 'a0', 'Pb', 'Yb', 'Vb']

#     corner_fig = plt.figure(figsize=(12, 8))

#     corner.corner(samples, fig=corner_fig, labels=labels, show_titles=True)
#     plt.savefig(f'rhoqso_corner_plot_{i}.png', bbox_inches='tight')
#     plt.savefig(f'rhoqso_corner_plot_{i}.pdf', bbox_inches='tight')
#     plt.show(block=False)

#     ndim = len(labels)
#     # Plot the chains for each parameter
#     chains_fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
#     colors = plt.cm.viridis(np.linspace(0, 1, sampler.nwalkers))
#     for j in range(ndim):
#         ax = axes[j]
#         ax.plot(sampler.get_chain()[:, :, j], "k", alpha=0.3)
#         ax.set_xlim(0, len(sampler.get_chain()))
#         ax.set_ylabel(labels[j])
#     axes[-1].set_xlabel("Step number")
#     plt.tight_layout()
#     plt.savefig(f'rhoqso_chains_plot_{i}.png', bbox_inches='tight')
#     plt.savefig(f'rhoqso_chains_plot_{i}.pdf', bbox_inches='tight')
#     plt.show(block=False)

#     print("\n\n\n")
#     print("Acceptance fraction:", sampler.acceptance_fraction)

#     # Check autocorrelation time
#     try:
#         print("Autocorrelation time:", sampler.get_autocorr_time())
#     except Exception as e:
#         print("Error calculating autocorrelation time:", e)
#     print("\n\n\n")

#     # Plot results
#     # plot_results(samples)

#     # Return the samples for further analysis if needed
#     return samples

# input_data_file = 'rhoqso_output_data.txt'

# all_data = []
# with open(input_data_file, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         if line.startswith('#'):
#             continue
#         data = line.strip().split(',')
#         # print(data)
#         all_data.append(data)

# all_data = np.array(all_data)

# all_data = all_data.astype(float)


# breaks = np.where(np.diff(all_data[:, 0]) <= 0)[0] + 1
# print("Breakpoints:", breaks)

# subarrays = np.split(all_data, breaks)

# main_fig = plt.figure(figsize=(12, 8))

# for i, subarray in enumerate(subarrays):
#     print(f"\nSubarray {i + 1}:")
#     # for j, data in enumerate(subarray):
#         # print(f"  {j + 1:>2}: {data}")
#     print("Subarray = ", subarray)
#     # time.sleep(10)
#     z, uzerr, lzerr, rho, rho_up, rho_low = [x.flatten() for x in np.split(subarray, 6, axis=1)]
#     print("z = ", z)
#     print("rho = ", rho)

#     # Fit a polynomial of the desired order to the data
#     order = 2  # Change this to control the order of the polynomial
#     coefficients = np.polyfit(z, np.log10(rho), order)

#     # Print the coefficients of the best-fit polynomial
#     print(f"Best-fit polynomial coefficients (order {order}): {coefficients}")

#     # Evaluate the polynomial at the given z values
#     fitted_values = np.polyval(coefficients, z)

#     # Plot the original data and the best-fit polynomial
#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(z, np.log10(rho), label='Data', color='blue', alpha=0.6)
#     # plt.plot(z, fitted_values, label=f'Best-fit Polynomial (order {order})', color='red', linewidth=2)
#     # plt.xlabel('z')
#     # plt.ylabel('log(rho)')
#     # plt.title(f'Best-fit Polynomial of Order {order}')
#     # plt.legend()
#     # plt.grid()
#     # plt.show(block=False)

#     samples = modelling_using_David_Hogg_with_uncertainity_in_1D(i, 3, z, np.log10(rho).flatten(), np.log10(rho_up).flatten(), np.log10(rho_low).flatten())
#     # Plot the results
#     plot_results(samples, main_fig, 3, z, np.log10(rho).flatten(), np.log10(rho_up).flatten(), np.log10(rho_low).flatten())

# plt.savefig('rhoqso_final_plot.png', bbox_inches='tight')
# plt.savefig('rhoqso_final_plot.pdf', bbox_inches='tight')
# plt.show(block=False)
# print()








# A = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7]

# print("Original array:", A)
# # Convert the list to a NumPy array
# A = np.array(A)

# # Find indices where the ascending order breaks
# breaks = np.where(np.diff(A) <= 0)[0] + 1

# print("Breakpoints:", breaks)
# # Split the array at the breakpoints
# subarrays = np.split(A, breaks)

# # Print the resulting subarrays
# for i, subarray in enumerate(subarrays):
#     print(f"Subarray {i + 1}: {subarray}")