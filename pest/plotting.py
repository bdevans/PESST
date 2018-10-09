import os
from textwrap import wrap

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats  # gamma


def plot_threshold_fitness(generation, population, fitnesses, fitness_table,
                           omega, out_paths):
    # Store fitness values for each amino in the dataset for the left subfigure
    (clone_size, n_amino_acids) = fitness_table.shape
    # Average across flattened array
    mean_initial_fitness = np.mean(fitness_table.values)
    scale = round((4 * np.std(fitness_table.values)) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                   gridspec_kw={'width_ratios': [1, 2]})
    # Plot each column of fitness_table as a separate dataseries against 0..N-1
    ax1.plot(fitness_table, "o", color='k', markersize=1)
    ax1.hlines(mean_initial_fitness, 0, clone_size-1,
               colors="r", linestyles="--", lw=4, zorder=10,
               label=r"$\mu_0$ = {:.2f}".format(mean_initial_fitness))
    if omega > -np.inf:
        ax1.hlines(omega, 0, clone_size-1,
                   colors="k", linestyles="-", lw=4, zorder=10,
                   label=r"$\Omega$ = {}".format(omega))
    ax1.set_ylim(-scale, scale)
    ax1.set_ylabel(r"$\Delta T_m$")
    ax1.set_xlabel("Amino acid position")
    ax1.legend(loc="upper left", fontsize=6.5)
               # title=r"$\Omega$ = {}".format(omega))
    ax1.set_title(r"Fitness distribution of $\Delta T_m$ matrix", size=8)

    # Find and plot all fitness values in the current generation
    mean_fitness = np.mean(fitnesses)
    # x: proteins within population; y: Fitness for each locus for that protein
    # TODO: Swap colour to a particular protein not locus or make monochrome
    ax2.plot(np.arange(len(population)), fitnesses, "o", markersize=1)
    # ax2.plot(np.arange(len(population)), np.sum(fitnesses, axis=1), "*k", markersize=4, label=r"$\mu_p$")
    ax2.hlines(mean_fitness, 0, len(population)-1,
               colors="r", linestyles="--", lw=4, zorder=10,
               label=r"$\mu_p$ = {:.2f}".format(mean_fitness))  # \mu_\phi ?
    if omega > -np.inf:
        ax2.hlines(omega, 0, len(population)-1,
                   colors="k", linestyles="-", lw=4, zorder=10,
                   label=r"$\Omega$ = {}".format(omega))
    ax2.set_ylim(-scale, scale)
    ax2.set_xlabel("Clone")
    ax2.legend(loc="upper left", fontsize=6.5)
               # title=r"$\Omega$ = {}".format(omega))
    # ax2.set_title("\n".join(wrap("Fitness distribution of every sequence in "
    #                              "the evolving dataset", 40)), size=8)
    ax2.set_title("Fitness distribution of every protein in the population",
                  size=8)

    # plt.subplots_adjust(top=0.85)
    fig.suptitle(("Generation {}".format(generation)), fontweight='bold')
    # fig.tight_layout()
    filename = os.path.join(out_paths["figures"], "generation_{}.png".format(generation))
    fig.savefig(filename)
    plt.close()


def plot_fitness_space(generation, population, fitnesses, fitness_table,
                       omega, out_paths):

    col_aa_g = "#3498db"  # Blue
    col_aa_0 = "#95a5a6"  # Green
    col_phi = "#34495e"
    col_mu_phi = "#9b59b6"  # Purple
    col_omega = "#e74c3c"  # "k"
    col_epsilon = "#2ecc71"  # Red

    # Store fitness values for each amino in the dataset for the left subfigure
    (clone_size, n_amino_acids) = fitness_table.shape
    # Average across flattened array
    # mean_initial_fitness = np.mean(fitness_table.values)
    scale = round((4 * np.std(fitness_table.values)) + 1)
    loc = np.mean(fitness_table.values)
    T_max = sum(np.amax(fitness_table, axis=1))  # Fittest possible protein

    fig, (ax_arr) = plt.subplots(2, 2, sharey='row', #  sharex='col',
                                 gridspec_kw={'width_ratios': [4, 1],
                                              'height_ratios': [1, 3]})
    # Plot each column of fitness_table as a separate dataseries against 0..N-1
    # ax1.plot(fitness_table, "o", color='k', markersize=1)
    # ax1.hlines(mean_initial_fitness, 0, clone_size-1,
    #            colors="r", linestyles="--", lw=2,
    #            label=r"$\mu_1$ = {:.2f}".format(mean_initial_fitness))
    # if omega > -np.inf:
    #     ax1.hlines(omega, 0, clone_size-1,
    #                colors="k", linestyles="-", lw=2,
    #                label=r"$\Omega$ = {}".format(omega))
    # ax1.set_ylim(-scale, scale)
    # ax1.set_ylabel(r"$\Delta T_m$")
    # ax1.set_xlabel("Amino acid position")
    # ax1.legend(loc="upper left", fontsize=6.5,)
               # title=r"$\Omega$ = {}".format(omega))
    # ax1.set_title(r"Fitness distribution of $\Delta T_m$ matrix", size=8)

    # Find and plot all fitness values in the current generation
    mean_fitness = np.mean(fitnesses)
    # x: proteins within population; y: Fitness for each locus for that protein
    # TODO: Swap colour to a particular protein not locus or make monochrome
    protein_indicies = np.arange(len(population))
    protein_fitnesses = np.sum(fitnesses, axis=1)
    ax_arr[0, 0].plot(protein_indicies, protein_fitnesses, "*", color=col_phi, markersize=4)  # , label=r"$\mu_p$")
    ax_arr[0, 1].hist(protein_fitnesses, bins=int(np.sqrt(len(population))),
                      align='mid', orientation='horizontal', density=True)
    # plt.setp(ax_arr[0, 1].get_xticklabels(), visible=False)
    ax_arr[0, 1].set_xticks([])
    mean_protein_fitness = np.mean(protein_fitnesses)
    ax_arr[0, 0].hlines(mean_protein_fitness, 0, len(population)-1,
                        colors=col_mu_phi, linestyles="--", lw=3, zorder=10,
                        label=r"$\mu_\phi$ = {:.2f}".format(mean_fitness))  # \mu_\phi ?
    mean_initial_amino_acid_fitness = np.mean(fitness_table.values)
    epsilon = clone_size * mean_initial_amino_acid_fitness
    ax_arr[0, 0].hlines(epsilon, 0, len(population)-1,
                        colors=col_epsilon, linestyles=":", lw=3, zorder=10,
                        label=r"$\epsilon$ = {:.2f}".format(epsilon))
    ncol = 2
    if omega > -np.inf:
        ax_arr[0, 0].hlines(omega, 0, len(population)-1,
                            colors=col_omega, linestyles="-", lw=3, zorder=10,
                            label=r"$\Omega$ = {}".format(omega))
        ncol += 1

    ax_arr[0, 0].set_ylabel(r"$T_m$")
    plt.setp(ax_arr[0, 0].get_xticklabels(), visible=False)
    ax_arr[0, 0].legend(loc="upper left", fontsize=6.5, ncol=ncol)

    ax_arr[1, 0].plot(protein_indicies, fitnesses, "o", color=col_aa_g, markersize=1)
    n, bins, _ = ax_arr[1, 1].hist(fitness_table.values.ravel(),
                                   bins='sqrt',  # int(np.sqrt(fitnesses.size))
                                   color=col_aa_0, alpha=0.4, align='mid',
                                   orientation='horizontal', density=True,
                                   label="Initial dist.")
    ax_arr[1, 1].axhline(y=mean_initial_amino_acid_fitness, color=col_aa_0,
                         linestyle="--", lw=3, zorder=10)
                         # label=r"$\mu_0$ = {:.2f}".format(mean_initial_amino_acid_fitness))
    ax_arr[1, 1].hist(fitnesses.ravel(), bins=bins,  # int(np.sqrt(fitnesses.size))
                      align='mid', color=col_aa_g, alpha=0.4,
                      orientation='horizontal', density=True, label="Present dist.")
    ax_arr[1, 1].axhline(y=mean_fitness, color=col_aa_g, linestyle="--", lw=3, zorder=10)
                         # label=r"$\mu_p$ = {:.2f}".format(mean_fitness))
    ax_arr[1, 1].axhline(y=omega, color=col_omega, linestyle="-", lw=3, zorder=10)
                         # label=r"$\Omega$ = {}".format(omega))
    ax_arr[1, 1].legend(loc="upper left", fontsize=6.5)
    # plt.setp(ax_arr[1, 1].get_xticklabels(), visible=False)
    ax_arr[1, 1].set_xticks([])
    # Set to 1.5*largest original bin count
    # ax_arr[1, 1].set_ylim(0, round(1.5*np.amax(n)))
    ax_arr[1, 1].set_ybound(0, 0.5)

    ax_arr[1, 0].hlines(mean_initial_amino_acid_fitness, 0, len(population)-1,
                        colors=col_aa_0, linestyles="--", lw=3, zorder=10,
                        label=r"$\mu_0$ = {:.2f}".format(mean_initial_amino_acid_fitness))
    ax_arr[1, 0].hlines(mean_fitness, 0, len(population)-1,
                        colors=col_aa_g, linestyles="--", lw=3, zorder=10,
                        label=r"$\mu_p$ = {:.2f}".format(mean_fitness))  # \mu_\phi ?
    if omega > -np.inf:
        ax_arr[1, 0].hlines(omega, 0, len(population)-1,
                            colors=col_omega, linestyles="-", lw=3, zorder=10,
                            label=r"$\Omega$ = {}".format(omega))

    ax_arr[0, 0].set_ylim(None, round(T_max))
    ax_arr[1, 0].set_ylim(loc-scale, loc+scale)
    ax_arr[1, 0].set_xlabel("Clone")
    # ax_arr[1, 1].set_xlabel("Density")
    ax_arr[1, 0].set_ylabel(r"$\Delta T_m$")
    ax_arr[1, 0].legend(loc="upper left", fontsize=6.5, ncol=ncol)
               # title=r"$\Omega$ = {}".format(omega))

    # ax_arr[1, 0].set_title("Fitness distribution of every protein in the population",
    #               size=8)

    # plt.subplots_adjust(top=0.85)
    fig.suptitle(("Generation {}".format(generation)), fontweight='bold')  # TODO Prevent overlap with spines
    fig.set_tight_layout(True)
    filename = os.path.join(out_paths["figures"], "fit_dist_gen_{}.png".format(generation))
    fig.savefig(filename)
    plt.close()
    return (fig, ax_arr)


def plot_histogram_of_fitness(generation, distributions, initial, omega, out_paths):
    plt.figure()
    # plt.axis([-10, 8, 0, 0.5])  # generate attractive figure

    # Plot normal distribution of the original fitness space
    mu1distspace = sum(initial) / len(initial)
    plt.hist(initial, 50, density=True, color='k', alpha=0.4)
    # plt.title("\n".join(wrap('Fitness distribution of the total fitness space',
    #                          60)), fontweight='bold')
    # plt.axvline(x=mu1distspace, color="#404040", linestyle=":")
    plt.axvline(x=mu1distspace, color="k", linestyle=":",
                label=r"$\mu_0$ = {:.3}".format(mu1distspace))

    # Plot normal distribution of the current generation's fitness space
    mu2distspace = sum(distributions) / len(distributions)
    plt.hist(distributions, 50, density=True, color='r', alpha=0.4)
    plt.title("\n".join(wrap("Fitness distribution of the total fitness space "
                             "vs. changing fitness distribution across every "
                             "evolving clone", 60)), fontweight='bold')
    # plt.axvline(x=mu2distspace, color="#404040", linestyle=":")
    plt.axvline(x=mu2distspace, color="r", linestyle=":",
                label=r"$\mu_p$ = {:.3}".format(mu2distspace))

    if omega > -np.inf:
        plt.axvline(x=omega, color="k", linestyle="-",
                    label=r"$\Omega$ = {}".format(omega))
    plt.legend()
    # plt.text(4.1, 0.4, "\n".join([r"$\mu_1$ = {:.3}".format(mu1distspace),
    #                               r"$\mu_2$ = {:.3}".format(mu2distspace),
    #                               r"$\Omega$ = {}".format(omega)]))

    plt.savefig(os.path.join(out_paths["figures"], "histogram_{}.png".format(generation)))
    plt.close()


def plot_evolution(history, fitness_table, omega, plot_omega, plot_epsilon,
                   out_paths, fig_title=True, legend_title=None, xlims=None, ax=None):
    """Plot fitness against generation for all clones.

    This plots after mutation but before replacement so that subthreshold
    proteins are briefly shown to exist.
    """
    col_mu_phi = "#34495e"
    col_omega = "#e74c3c"  # Red
    col_epsilon = '#33a02c'  # "#2ecc71"  # Green
    # Create array of fitness values with shape (n_generations, n_clones)
    n_generations = len(history) - 1  # First entry is the initial state
    generation_numbers = np.arange(n_generations+1)  # Skip initial generation
    (clone_size, n_amino_acids) = fitness_table.shape
    n_clones = len(history[0].population)
    fitnesses = np.array([[history[g].fitness[c] for c in range(n_clones)]
                          for g in range(n_generations+1)])

    final_fitnesses = np.array([[history[g].final_fitness[c]
                                 for c in range(n_clones)]
                                for g in range(n_generations+1)])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    ax.plot(generation_numbers, fitnesses, lw=1)

    # Average across clones
    ax.plot(generation_numbers, np.mean(final_fitnesses, axis=1),
            "--", color=col_mu_phi, lw=3, zorder=20, label=r"$\mu_\phi$")
    if xlims is not None:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim([-5, n_generations+5])
    ax.set_xlabel("Generation")  # , fontweight='bold')
    ax.set_ylabel("$T_m$", fontweight='bold')
    if fig_title:
        ax.set_title("\n".join(wrap("Stability change for {} randomly generated "
                                    "clones of {} amino acids, "
                                    "mutated over {} generations"
                                    .format(n_clones, clone_size, n_generations), 60)), fontweight='bold')
    if plot_omega:  # Add fitness threshold
        ax.axhline(omega, color=col_omega, lw=3, linestyle="-", zorder=10,
                   label=r"$\Omega$ = {}".format(omega))
    if plot_epsilon:  # Add theoretical convergence line
        epsilon = clone_size * np.mean(fitness_table.values)
        ax.axhline(epsilon, color=col_epsilon, lw=3, linestyle=":", zorder=10,
                   label=r"$\epsilon$ = {:.2f}".format(epsilon))
    ax.legend(title=legend_title)

    if ax is None:
        evo_fig = "stability_evolution_over_{}_generations.png".format(n_generations)
        fig.savefig(os.path.join(out_paths["figures"], evo_fig))
    return ax


def plot_gamma_distribution(gamma, samples, quartiles, average_medians, out_paths):
    """Plot the distribution along with the quartiles and medians."""
    kappa, theta, n_iterations, n_samples = (gamma["shape"],
                                             gamma["scale"],
                                             gamma["iterations"],
                                             gamma["samples"])

    x = np.linspace(0, 6, 1000)
    # y = x**(kappa - 1) * (np.exp(-x / theta)
    #                         / (stats.gamma(kappa).pdf(x, kappa)))  # * theta ** kappa))
    y = stats.gamma.pdf(x, kappa, scale=theta)
    plt.plot(x, y, linewidth=2, color='k', alpha=0,
             label="\n".join([r"$\kappa$ = {:.2f}".format(kappa),
                              r"$\theta$ = {:.2f}".format(theta)]))
                              #r"$\theta$ = $\frac{{}}{{}}$".format(1, kappa)]))
    plt.hist(samples, bins=int(np.sqrt(len(samples))), range=(0, 6),
             density=True, color='g', histtype='step')
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
    plt.legend()
    plt.savefig(os.path.join(out_paths["initial"], "gamma.png"))


def plot_fitness_table(fitness_table, out_paths):

    (clone_size, n_amino_acids) = fitness_table.shape
    fig, ax = plt.subplots(figsize=(8, clone_size/5))
    sns.heatmap(fitness_table, center=0, annot=True, fmt=".2f", linewidths=.5,
                cmap="RdBu_r", annot_kws={"size": 5},
                cbar_kws={"label": r"$\Delta T_m$"}, ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Location")
    filename = os.path.join(out_paths["initial"], "fitness_table.png")
    fig.savefig(filename)
    plt.close()


def plot_LG_matrix(LG_matrix, out_paths):

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(LG_matrix, annot=True, fmt=".2f", linewidths=.5, cmap="cubehelix",
                square=True, annot_kws={"size": 5},  # vmin=0, vmax=1,
                cbar_kws={"label": r"$\Delta T_m$"}, ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Amino Acid")
    ax.set_title("LG model transition probabilities")
    filename = os.path.join(out_paths["initial"], "LG_matrix.png")
    fig.savefig(filename)
    plt.close()


def plot_phi_fitness_table(generation, phi_fitness_table, clims, out_paths):
    r"""Plot a heatmap of changes in stability (\Delta T_m) for each amino acid
    in each protein."""
    (n_proteins, clone_size) = phi_fitness_table.shape
    fig, ax = plt.subplots(figsize=(clone_size/6, n_proteins/8))
    sns.heatmap(phi_fitness_table, center=0, annot=False, fmt=".2f",
                linewidths=.5, cmap="RdBu_r", annot_kws={"size": 5},
                cbar_kws={"label": r"$\Delta T_m$"}, vmin=clims[0], vmax=clims[1], ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Location")
    ax.set_ylabel("Clone")
    filename = os.path.join(out_paths["figures"],
                            "phi_fitness_table_{}.png".format(generation))
    ax.set_title("Generation {}".format(generation), fontweight="bold")
    fig.savefig(filename)
    plt.close()
