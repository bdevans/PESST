import os
from textwrap import wrap

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats  # gamma

# from .evolution import calculate_fitness, get_random_protein


# NOTE: unused
def test_normal_distribution(mu, sigma):
    """Plot a distribution to test normalality."""
    s = np.random.normal(mu, sigma, 2000)  # generate distribution
    count, bins, ignored = plt.hist(s, 30, density=True)  # plot distribuiton
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                   np.exp(- (bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')
    plt.show()
    return


# # NOTE: Not used
# def plot_fitness_histogram(n_proteins, n_amino_acids, fitness_table):
#     """Generate and plot fitness values for f proteins."""
#     fitnesses = [calculate_fitness(get_random_protein(n_amino_acids, fitness_table), fitness_table)
#                  for p in range(n_proteins)]
#     plt.hist(fitnesses, density=True)  # plot fitnesses as histogram
#     plt.show()
#     return


def plot_threshold_fitness(generation, population, fitnesses, fitness_table, fitness_threshold, save_dir):
    # Store fitness values for each amino in the dataset for the left side of the figure
    (n_amino_acids, n_variants) = fitness_table.shape
    mean_initial_fitness = np.mean(fitness_table.values)  # Average across flattened array
    sigma = np.std(fitness_table.values)
    scale = round((4 * sigma) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # Plot each column of fitness_table as a separate dataseries against 0..N-1
    ax1.plot(fitness_table, "o", color='k', markersize=2)  # np.arange(n_amino_acids)+1,
    ax1.plot([0, n_amino_acids-1], [mean_initial_fitness, mean_initial_fitness],
             'r--', lw=2,
             label="\n".join([r"$\mu_1$ = {:.2f}".format(mean_initial_fitness),
                              r"$\Omega$ = {}".format(fitness_threshold)]))
    ax1.set_ylim(-scale, scale)
    ax1.set_ylabel(r"$\Delta T_m$")
    ax1.set_xlabel("Amino acid position")
    ax1.legend(loc="upper right", fontsize=6.5)
    ax1.set_title(r"Fitness distribution of $\Delta T_m$ matrix", size=8)

    # Find and plot all fitness values in the current generation
    mean_generation_fitness = np.mean(fitnesses)
    # x: proteins within population; y: Fitness for each locus for that protein
    # TODO: Swap colour to a particular protein not locus or make monochrome
    ax2.plot(np.arange(len(population)), fitnesses, "o", markersize=1)  # plot y using x as index array 0..N-1
    ax2.plot([0, len(population)-1], [mean_generation_fitness, mean_generation_fitness],
             'r--', lw=2,
             label="\n".join([r"$\mu_2$ = {:.2f}".format(mean_generation_fitness),
                              r"$\Omega$ = {}".format(fitness_threshold)]))
    ax2.set_ylim(-scale, scale)
    ax2.set_xlabel("Protein")
    ax2.legend(loc="upper right", fontsize=6.5)
    ax2.set_title("\n".join(wrap("Fitness distribution of every sequence in the evolving dataset", 40)), size=8)

    # plt.subplots_adjust(top=0.85)
    fig.suptitle(("Generation {}".format(generation)), fontweight='bold')
    filename = os.path.join(save_dir, "generation_{}.png".format(generation))
    fig.savefig(filename)
    plt.close()


def plot_histogram_of_fitness(disthistfullname, distributions, initial, fitness_threshold):
    plt.figure()
    plt.axis([-10, 8, 0, 0.5])  # generate attractive figure

    # Plot normal distribution of the original fitness space
    mu1distspace = sum(initial) / len(initial)
    plt.hist(initial, 50, density=True, color='k', alpha=0.4)
    plt.title("\n".join(wrap('Fitness distribution of the total fitness space', 60)), fontweight='bold')
    plt.axvline(x=mu1distspace, color="#404040", linestyle=":")

    # Plot normal distribution of the current generation's fitness space
    mu2distspace = sum(distributions) / len(distributions)
    plt.hist(distributions, 50, density=True, color='r', alpha=0.4)
    plt.title("\n".join(wrap('Fitness distribution of the total fitness space vs. changing fitness distribution across every evolving clone', 60)), fontweight='bold')
    plt.axvline(x=mu2distspace, color="#404040", linestyle=":")
    plt.text(4.1, 0.42, "\n".join([r"$\mu_1$ = {:.3}".format(mu1distspace),
                                   r"$\mu_2$ = {:.3}".format(mu2distspace),
                                   r"$\Omega$ = {}".format(fitness_threshold)]))

    plt.savefig(disthistfullname)
    plt.close()


def plot_evolution(history, n_clones, n_amino_acids, initial_fitness, fitness_table, fitness_threshold, plot_omega, plot_epsilon, mu, sigma, mutation_rate, run_path):
    """Plot fitness against generation for all clones.

    This plots after mutation but before replacement so that subthreshold
    proteins are briefly shown to exist.
    """

    # Create array of fitness values with shape (n_generations, n_clones)
    n_generations = len(history) - 1  # First entry is the initial state
    # n_amino_acids = len(initial_protein)
    fitnesses = np.array([[history[g].fitness[c] for c in range(n_clones)]
                          for g in range(n_generations+1)])

    # initial_fitness = calculate_fitness(initial_protein, fitness_table)
    generation_numbers = np.arange(n_generations+1)  # Skip initial generation

    # TODO: Make threshold plotting optional so they can add convergence value instead i.e. epsillon len(protein) * mean(fitness_table)

    plt.figure()
    plt.plot(generation_numbers, fitnesses)
    if plot_omega:  # Add fitness threshold
        plt.plot([0, n_generations], [fitness_threshold, fitness_threshold], 'k-', lw=2, label=r"\Omega")
    if plot_epsilon:  # Add theoretical convergence line
        epsilon = n_amino_acids * np.mean(fitness_table)
        plt.plot([0, n_generations], [epsilon, epsilon], 'k-', lw=2, label=r"\epsilon")
    plt.plot(generation_numbers, np.mean(fitnesses, axis=1), "k--", lw=2)  # Average across clones
    # plt.ylim([fitness_threshold-25, initial_fitness+10])  # not suitable for "low or med" graphs
    # plt.ylim([fitness_threshold-5, ((n_amino_acids+1)*mu)+80]) # for low graphs
    plt.ylim([fitness_threshold-25, initial_fitness+100])  # suitable for med graphs
    plt.xlim([0, n_generations])
    plt.xlabel("Generations", fontweight='bold')
    plt.ylabel("$T_m$", fontweight='bold')
    plt.title("\n".join(wrap("Fitness change for {n_clones} randomly generated "
                             "'superfit' clones of {n_amino_acids} amino acids, "
                             "mutated over {n_generations} generations"
                             .format(n_clones=n_clones, n_amino_acids=n_amino_acids,
                             n_generations=n_generations), 60)), fontweight='bold')
    plt.text(n_generations+15, fitness_threshold-3, r"$\Omega$")
    plt.text(n_generations-1000, initial_fitness+50,
             "\n".join([r"$\mu$ = {}".format(mu),
                        r"$\sigma$ = {}".format(sigma),
                        r"$\delta$ = {}".format(mutation_rate)]))

    # Define dynamic filename
    fitgraphfilename = "fitness_change_over{}generations.png".format(n_generations)
    fitgraphfullname = os.path.join(run_path, "fitnessgraph", fitgraphfilename)
    plt.savefig(fitgraphfullname)
    return


def plot_gamma_distribution(gamma, quartiles, average_medians):
    """Plot the distribution along with the quartiles and medians."""
    kappa, theta, n_iterations, n_samples = (gamma["shape"], gamma["scale"],
                                             gamma["iterations"], gamma["samples"])

    x = np.linspace(0, 6, 1000)
    y = x ** (kappa - 1) * (np.exp(-x / theta)
                            / (stats.gamma(kappa).pdf(x) * theta ** kappa))
    plt.plot(x, y, linewidth=2, color='k', alpha=0)
    plt.fill_between(x, y, where=x > quartiles[0], color='#4c4cff')
    plt.fill_between(x, y, where=x > quartiles[1], color='#7f7fff')
    plt.fill_between(x, y, where=x > quartiles[2], color='#b2b2ff')
    plt.fill_between(x, y, where=x > quartiles[3], color='#e5e5ff')
    plt.axvline(x=average_medians[0], color="#404040", linestyle=":")
    plt.axvline(x=average_medians[1], color="#404040", linestyle=":")
    plt.axvline(x=average_medians[2], color="#404040", linestyle=":")
    plt.axvline(x=average_medians[3], color="#404040", linestyle=":")
    plt.title("\n".join(wrap('Gamma rate categories calculated as the the average of %s median values of 4 equally likely quartiles of %s randomly sampled vaules' % (n_iterations, n_samples), 60)), fontweight='bold', fontsize=10)
    plt.text(5, 0.6, r"$\kappa$ = %s\n$\theta$ = $\frac{1}{\kappa}$" % (kappa))
    plt.show()
    plt.savefig(os.path.join(".", "gamma.png"))