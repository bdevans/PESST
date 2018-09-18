import os
from textwrap import wrap

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats  # gamma

# from .evolution import calculate_fitness, get_random_protein


# NOTE: unused
# def test_normal_distribution(mu, sigma):
#     """Plot a distribution to test normalality."""
#     s = np.random.normal(mu, sigma, 2000)  # generate distribution
#     count, bins, ignored = plt.hist(s, 30, density=True)  # plot distribuiton
#     plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                    np.exp(- (bins - mu)**2 / (2 * sigma**2)),
#              linewidth=2, color='r')
#     plt.show()
#     return


# # NOTE: Not used
# def plot_fitness_histogram(n_proteins, n_amino_acids, fitness_table):
#     """Generate and plot fitness values for f proteins."""
#     fitnesses = [calculate_fitness(get_random_protein(n_amino_acids, fitness_table), fitness_table)
#                  for p in range(n_proteins)]
#     plt.hist(fitnesses, density=True)  # plot fitnesses as histogram
#     plt.show()
#     return


def plot_threshold_fitness(generation, population, fitnesses, fitness_table,
                           fitness_threshold, save_dir):
    # Store fitness values for each amino in the dataset for the left subfigure
    (n_amino_acids, n_variants) = fitness_table.shape
    # Average across flattened array
    mean_initial_fitness = np.mean(fitness_table.values)
    scale = round((4 * np.std(fitness_table.values)) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                   gridspec_kw={'width_ratios': [1, 2]})
    # Plot each column of fitness_table as a separate dataseries against 0..N-1
    ax1.plot(fitness_table, "o", color='k', markersize=1)
    ax1.hlines(mean_initial_fitness, 0, n_amino_acids-1,
               colors="r", linestyles="--", lw=2,
               label=r"$\mu_1$ = {:.2f}".format(mean_initial_fitness))
    if fitness_threshold > -np.inf:
        ax1.hlines(fitness_threshold, 0, n_amino_acids-1,
                   colors="k", linestyles="-", lw=2,
                   label=r"$\Omega$ = {}".format(fitness_threshold))
    ax1.set_ylim(-scale, scale)
    ax1.set_ylabel(r"$\Delta T_m$")
    ax1.set_xlabel("Amino acid position")
    ax1.legend(loc="upper left", fontsize=6.5,)
               # title=r"$\Omega$ = {}".format(fitness_threshold))
    ax1.set_title(r"Fitness distribution of $\Delta T_m$ matrix", size=8)

    # Find and plot all fitness values in the current generation
    mean_fitness = np.mean(fitnesses)
    # x: proteins within population; y: Fitness for each locus for that protein
    # TODO: Swap colour to a particular protein not locus or make monochrome
    ax2.plot(np.arange(len(population)), fitnesses, "o", markersize=1)
    # ax2.plot(np.arange(len(population)), np.sum(fitnesses, axis=1), "*k", markersize=4, label=r"$\mu_p$")
    ax2.hlines(mean_fitness, 0, len(population)-1,
               colors="r", linestyles="--", lw=2,
               label=r"$\mu_2$ = {:.2f}".format(mean_fitness))  # \mu_\phi ?
    if fitness_threshold > -np.inf:
        ax2.hlines(fitness_threshold, 0, len(population)-1,
                   colors="k", linestyles="-", lw=2,
                   label=r"$\Omega$ = {}".format(fitness_threshold))
    ax2.set_ylim(-scale, scale)
    ax2.set_xlabel("Protein")
    ax2.legend(loc="upper left", fontsize=6.5,)
               # title=r"$\Omega$ = {}".format(fitness_threshold))
    # ax2.set_title("\n".join(wrap("Fitness distribution of every sequence in "
    #                              "the evolving dataset", 40)), size=8)
    ax2.set_title("Fitness distribution of every protein in the population",
                  size=8)

    # plt.subplots_adjust(top=0.85)
    fig.suptitle(("Generation {}".format(generation)), fontweight='bold')
    # fig.tight_layout()
    filename = os.path.join(save_dir, "generation_{}.png".format(generation))
    fig.savefig(filename)
    plt.close()


def plot_fitness_space(generation, population, fitnesses, fitness_table,
                       fitness_threshold, save_dir):
    # Store fitness values for each amino in the dataset for the left subfigure
    (n_amino_acids, n_variants) = fitness_table.shape
    # Average across flattened array
    # mean_initial_fitness = np.mean(fitness_table.values)
    scale = round((4 * np.std(fitness_table.values)) + 1)
    T_max = sum(np.amax(fitness_table, axis=1))  # Fittest possible protein

    fig, (ax_arr) = plt.subplots(2, 2, sharey='row', #  sharex='col',
                                 gridspec_kw={'width_ratios': [4, 1],
                                              'height_ratios': [1, 2]})
    # Plot each column of fitness_table as a separate dataseries against 0..N-1
    # ax1.plot(fitness_table, "o", color='k', markersize=1)
    # ax1.hlines(mean_initial_fitness, 0, n_amino_acids-1,
    #            colors="r", linestyles="--", lw=2,
    #            label=r"$\mu_1$ = {:.2f}".format(mean_initial_fitness))
    # if fitness_threshold > -np.inf:
    #     ax1.hlines(fitness_threshold, 0, n_amino_acids-1,
    #                colors="k", linestyles="-", lw=2,
    #                label=r"$\Omega$ = {}".format(fitness_threshold))
    # ax1.set_ylim(-scale, scale)
    # ax1.set_ylabel(r"$\Delta T_m$")
    # ax1.set_xlabel("Amino acid position")
    # ax1.legend(loc="upper left", fontsize=6.5,)
               # title=r"$\Omega$ = {}".format(fitness_threshold))
    # ax1.set_title(r"Fitness distribution of $\Delta T_m$ matrix", size=8)

    # Find and plot all fitness values in the current generation
    mean_fitness = np.mean(fitnesses)
    # x: proteins within population; y: Fitness for each locus for that protein
    # TODO: Swap colour to a particular protein not locus or make monochrome
    ax_arr[0, 0].plot(np.arange(len(population)), np.sum(fitnesses, axis=1), "*k", markersize=4, label=r"$\mu_p$")
    ax_arr[0, 1].hist(np.sum(fitnesses, axis=1), bins=10, align='mid', orientation='horizontal', density=True)
    ax_arr[0, 0].hlines(np.mean(np.sum(fitnesses, axis=1)), 0, len(population)-1,
                        colors="r", linestyles="--", lw=2,
                        label=r"$\mu_\phi$ = {:.2f}".format(mean_fitness))  # \mu_\phi ?
    # epsilon = n_amino_acids * np.mean(fitness_table.values)
    # ax_arr[0, 0].hlines(epsilon, 0, len(population)-1,
    #                     colors="k", linestyles="-", lw=2,
    #                     label=r"$\epsilon$ = {}".format(epsilon))
    if fitness_threshold > -np.inf:
        ax_arr[0, 0].hlines(fitness_threshold, 0, len(population)-1,
                            colors="k", linestyles="-", lw=2,
                            label=r"$\Omega$ = {}".format(fitness_threshold))

    ax_arr[1, 0].plot(np.arange(len(population)), fitnesses, "o", markersize=1)
    ax_arr[1, 1].hist(fitnesses.ravel(), bins=20, align='mid', orientation='horizontal', density=True)

    ax_arr[1, 0].hlines(mean_fitness, 0, len(population)-1,
                        colors="r", linestyles="--", lw=2,
                        label=r"$\mu_p$ = {:.2f}".format(mean_fitness))  # \mu_\phi ?
    if fitness_threshold > -np.inf:
        ax_arr[1, 0].hlines(fitness_threshold, 0, len(population)-1,
                            colors="k", linestyles="-", lw=2,
                            label=r"$\Omega$ = {}".format(fitness_threshold))

    ax_arr[0, 0].set_ylim(None, round(T_max))
    ax_arr[1, 0].set_ylim(-scale, scale)
    ax_arr[1, 0].set_xlabel("Protein")
    ax_arr[1, 1].set_xlabel("Density")
    ax_arr[1, 0].set_ylabel(r"$\Delta T_m$")
    ax_arr[1, 0].legend(loc="upper left", fontsize=6.5,)
               # title=r"$\Omega$ = {}".format(fitness_threshold))
    # ax2.set_title("\n".join(wrap("Fitness distribution of every sequence in "
    #                              "the evolving dataset", 40)), size=8)
    ax_arr[1, 0].set_title("Fitness distribution of every protein in the population",
                  size=8)

    # plt.subplots_adjust(top=0.85)
    fig.suptitle(("Generation {}".format(generation)), fontweight='bold')
    fig.tight_layout()
    filename = os.path.join(save_dir, "fit_dist_gen_{}.png".format(generation))
    fig.savefig(filename)
    plt.close()


def plot_histogram_of_fitness(disthistfullname, distributions, initial,
                              fitness_threshold):
    plt.figure()
    plt.axis([-10, 8, 0, 0.5])  # generate attractive figure

    # Plot normal distribution of the original fitness space
    mu1distspace = sum(initial) / len(initial)
    plt.hist(initial, 50, density=True, color='k', alpha=0.4)
    plt.title("\n".join(wrap('Fitness distribution of the total fitness space',
                             60)), fontweight='bold')
    # plt.axvline(x=mu1distspace, color="#404040", linestyle=":")
    plt.axvline(x=mu1distspace, color="k", linestyle=":",
                label=r"$\mu_1$ = {:.3}".format(mu1distspace))

    # Plot normal distribution of the current generation's fitness space
    mu2distspace = sum(distributions) / len(distributions)
    plt.hist(distributions, 50, density=True, color='r', alpha=0.4)
    plt.title("\n".join(wrap("Fitness distribution of the total fitness space "
                             "vs. changing fitness distribution across every "
                             "evolving clone", 60)), fontweight='bold')
    # plt.axvline(x=mu2distspace, color="#404040", linestyle=":")
    plt.axvline(x=mu2distspace, color="r", linestyle=":",
                label=r"$\mu_1$ = {:.3}".format(mu2distspace))

    if fitness_threshold > -np.inf:
        plt.axvline(x=fitness_threshold, color="k", linestyle="-",
                    label=r"$\Omega$ = {}".format(fitness_threshold))
    plt.legend()
    # plt.text(4.1, 0.4, "\n".join([r"$\mu_1$ = {:.3}".format(mu1distspace),
    #                               r"$\mu_2$ = {:.3}".format(mu2distspace),
    #                               r"$\Omega$ = {}".format(fitness_threshold)]))

    plt.savefig(disthistfullname)
    plt.close()


def plot_evolution(history, fitness_table, fitness_threshold,
                   plot_omega, plot_epsilon,
                   run_path, legend_title=None):
    """Plot fitness against generation for all clones.

    This plots after mutation but before replacement so that subthreshold
    proteins are briefly shown to exist.
    """
    # Create array of fitness values with shape (n_generations, n_clones)
    n_generations = len(history) - 1  # First entry is the initial state
    generation_numbers = np.arange(n_generations+1)  # Skip initial generation
    (n_amino_acids, n_variants) = fitness_table.shape
    n_clones = len(history[0].population)
    fitnesses = np.array([[history[g].fitness[c] for c in range(n_clones)]
                          for g in range(n_generations+1)])

    final_fitnesses = np.array([[history[g].final_fitness[c]
                                 for c in range(n_clones)]
                                for g in range(n_generations+1)])

    plt.figure()
    plt.plot(generation_numbers, fitnesses)

    # Average across clones
    plt.plot(generation_numbers, np.mean(final_fitnesses, axis=1),
             "k:", lw=2, label="Mean")
    plt.xlim([0, n_generations])
    plt.xlabel("Generations", fontweight='bold')
    plt.ylabel("$T_m$", fontweight='bold')
    plt.title("\n".join(wrap("Fitness change for {} randomly generated "
                             "'superfit' clones of {} amino acids, "
                             "mutated over {} generations"
                             .format(n_clones, n_amino_acids, n_generations),
                             60)), fontweight='bold')
    if plot_omega:  # Add fitness threshold
        plt.axhline(fitness_threshold, color="k", lw=2, linestyle="-",
                    label=r"$\Omega$ = {}".format(fitness_threshold))
    if plot_epsilon:  # Add theoretical convergence line
        epsilon = n_amino_acids * np.mean(fitness_table.values)
        plt.axhline(epsilon, color="b", lw=2, linestyle="--",
                    label=r"$\epsilon$ = {:.2f}".format(epsilon))
    plt.legend(title=legend_title)

    # Define dynamic filename
    fitgraphfilename = "fitness_over_{}_generations.png".format(n_generations)
    fitgraphfullname = os.path.join(run_path, "fitnessgraph", fitgraphfilename)
    plt.savefig(fitgraphfullname)
    return


def plot_gamma_distribution(gamma, quartiles, average_medians):
    """Plot the distribution along with the quartiles and medians."""
    kappa, theta, n_iterations, n_samples = (gamma["shape"],
                                             gamma["scale"],
                                             gamma["iterations"],
                                             gamma["samples"])

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
    plt.title("\n".join(wrap("Gamma rate categories calculated as the the "
                             "average of {} median values of 4 equally likely "
                             "quartiles of {} randomly sampled vaules"
                             .format(n_iterations, n_samples), 60)),
              fontweight='bold', fontsize=10)
    plt.text(5, 0.6, r"$\kappa$ = %s\n$\theta$ = $\frac{1}{\kappa}$" % (kappa))
    plt.show()
    plt.savefig(os.path.join(".", "gamma.png"))


def plot_fitness_table(fitness_table, run_path):

    (n_amino_acids, n_variants) = fitness_table.shape
    fig, ax = plt.subplots(figsize=(8, n_amino_acids/5))
    sns.heatmap(fitness_table, center=0, annot=True, fmt=".2f", linewidths=.5,
                cmap="RdBu_r", annot_kws={"size": 5},
                cbar_kws={"label": r"$\Delta T_m$"}, ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Location")
    filename = os.path.join(run_path, "start", "fitness_table.png")
    fig.savefig(filename)
    plt.close()


def plot_LG_matrix(LG_matrix, run_path):

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(LG_matrix, annot=True, fmt=".2f", linewidths=.5,
                square=True, annot_kws={"size": 5},  # vmin=0, vmax=1,
                cbar_kws={"label": r"$\Delta T_m$"}, ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Amino Acid")
    ax.set_title("LG model transition probabilities")
    filename = os.path.join(run_path, "start", "LG_matrix.png")
    fig.savefig(filename)
    plt.close()
