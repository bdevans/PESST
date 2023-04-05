import os
import copy
import warnings
from textwrap import fill

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats  # gamma


# pal = sns.color_palette("Paired"); print(pal.as_hex())
# ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
# '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
default_colours = {"aa_0": "#b2df8a",  # "#fdbf6f",
                   "aa_0_mu": "#33a02c",  # "#ff7f00",
                   "aa_g": "#a6cee3",
                   "aa_g_mu": "#1f78b4",
                   "phi": "#9b59b6",
                   "phi_0_mu": "#fb9a99",
                   "phi_mu": "#34495e",
                   "epsilon": "#fdbf6f", #"#ff7f00",  # "#33a02c",
                   "omega": "#e74c3c"}

def remove_inner_ticklabels(fig):
    for ax in fig.axes:
        try:
            ax.label_outer()
        except:
            pass


def plot_amino_acid_evolution(history, epsilon_r, out_paths, 
                              fig_title=True, legend_title=None, 
                              xlims=None, colours=None, ax=None):
    """Plot stability against generation for all amino acids.
    """
    if colours is None:
        colours = default_colours

    # Create array of stability values with shape (n_generations, n_clones)
    n_generations = len(history) - 1  # First entry is the initial state
    generation_numbers = np.arange(n_generations+1)  # Skip initial generation
    # (clone_size, n_amino_acids) = stability_table.shape
    # (n_clones, _) = history[0].stabilities.shape
    n_residues = history[0].stabilities.size  # == stability_table.shape

    stabilities = np.array([history[g].stabilities.ravel()
                            for g in range(n_generations+1)])
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    # sns.lineplot(generation_numbers, stabiltiies, estimator=None, lw=1)
    ax.plot(generation_numbers, stabilities,
            color=colours["aa_g"], alpha=0.2, lw=1)
    # color=sns.desaturate(colours["phi"], 0.3), lw=1)

    # Average across clones
    ax.plot(generation_numbers, np.mean(stabilities, axis=1), "--",
            color=colours["aa_g_mu"], lw=3, zorder=20, label=r"$\mu_{r}$")  # residues
    if xlims is not None:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim([-0.05*n_generations, 1.05*n_generations])
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$\Delta \Delta G_e$ (kcal/mol)")  # , fontweight='bold')
    if fig_title:
        ax.set_title(fill(f"Stability change for {n_residues} amino acids, "
                          f"mutated over {n_generations} generations", 60),
                     fontweight='bold')
    if epsilon_r is not None:  # Add theoretical convergence line
        # epsilon_r == mean_stability_0 == np.mean(stability_table.values)
        ax.axhline(epsilon_r, color=colours["aa_0_mu"], lw=3, linestyle="--",
                   zorder=20, label=rf"$\epsilon_r$ = {epsilon_r:.2f}")

    legend = ax.legend(title=legend_title, loc="upper right", frameon=True, fancybox=True, framealpha=0.7)
    legend.set_zorder(100)

    if fig is not None:
        evo_fig = f"residue_evolution_over_{n_generations}_generations.png"
        fig.savefig(os.path.join(out_paths["figures"], evo_fig))
    return ax


def plot_evolution(history, stability_table, omega, plot_omega, plot_epsilon,
                   out_paths, fig_title=True, legend_title=None, legend_loc="upper right",
                   xlims=None, colours=None, ax=None):
    """Plot stability against generation for all clones.

    This plots after mutation but before replacement so that subthreshold
    proteins are briefly shown to exist.
    """
    if colours is None:
        colours = default_colours

    # Create array of stability values with shape (n_generations, n_clones)
    # n_generations = list(history)[-1]
    # n_steps = len(history) - 1  # First entry is the initial state
    # generation_numbers = np.arange(0, n_generations, n_steps)  # Skip initial generation
    generation_numbers = list(history)
    n_generations = generation_numbers[-1]
    # print(generation_numbers)
    # print(list(history))
    (clone_size, n_amino_acids) = stability_table.shape
    (n_clones, _) = history[0].stabilities.shape

    stabilities = np.array([np.sum(history[g].stabilities, axis=1)
                            for g in generation_numbers])
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    # sns.lineplot(generation_numbers, stabiltiies, estimator=None, lw=1)
    ax.plot(generation_numbers, stabilities, color=colours["phi"], alpha=0.2, lw=1, zorder=30)
            # color=sns.desaturate(colours["phi"], 0.3), lw=1)

    # Average across clones
    ax.plot(generation_numbers, np.mean(stabilities, axis=1), "--",
            color=colours["phi_mu"], lw=3, zorder=40, label=r"$\mu_\phi$")
    if xlims is not None:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim([-0.05*n_generations, 1.05*n_generations])
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$\Delta G_e$ (kcal/mol)") #, fontweight='bold')
    if fig_title:
        ax.set_title(fill(f"Stability change for {n_clones} clones "
                        f"of {clone_size} amino acids, mutated "
                        f"over {n_generations} generations", 60),
                     fontweight='bold')
    if plot_omega:  # Add stability threshold
        ax.axhline(omega, color=colours["omega"], lw=3, linestyle="-",
                   zorder=20, label=rf"$\Omega$ = {omega}")
    if plot_epsilon:  # Add theoretical convergence line
        epsilon = clone_size * np.mean(stability_table.values)
        ax.axhline(epsilon, color=colours["epsilon"], lw=3, linestyle=":",
                   zorder=20, label=rf"$\epsilon$ = {epsilon:.2f}")

    legend = ax.legend(title=legend_title, loc=legend_loc, frameon=True, fancybox=True, framealpha=0.7)
    legend.set_zorder(100)

    if fig is not None:
        evo_fig = f"stability_evolution_over_{n_generations}_generations.png"
        fig.savefig(os.path.join(out_paths["figures"], evo_fig))
    return ax


def plot_amino_acid_stabilities(aa_stabilities, epsilon_r=None,
                                colours=None, ax=None):

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if colours is None:
        colours = default_colours

    (n_clones, clone_size) = aa_stabilities.shape
    protein_indicies = np.arange(1, n_clones+1)
    mean_stability = np.mean(aa_stabilities)

    ax.plot(protein_indicies, aa_stabilities,
            "o", color=colours["aa_g"], markersize=1)
    # ax.hlines(epsilon_r, 0, n_clones-1,
    #           colors=colours["aa_0_mu"], linestyles="--", lw=3, zorder=20,
    #           label=r"$\mu_0$ = {:.2f}".format(epsilon_r))
    if epsilon_r is not None:  # Add theoretical convergence line
        ax.axhline(epsilon_r, color=colours["aa_0_mu"], lw=3, linestyle="--",
                   zorder=20, label=rf"$\epsilon_r$ = {epsilon_r:.2f}")
    # ax.hlines(mean_stability, 0, n_clones-1,
    #           colors=colours["aa_g_mu"], linestyles="--", lw=3, zorder=20,
    #           label=rf"$\mu_p$ = {mean_stability:.2f}")
    ax.axhline(mean_stability, color=colours["aa_g_mu"], linestyle="--", lw=3,
               zorder=20, label=rf"$\mu_p$ = {mean_stability:.2f}")
    ncol = 2
    # if omega > -np.inf:
    #     ax.hlines(omega, 0, n_clones-1,
    #               colors=colours["omega"], linestyles="-", lw=3, zorder=20,
    #               label=r"$\Omega$ = {}".format(omega))
    #     ncol += 1

    ax.set_xlabel("Clone")
    ax.set_ylabel(r"$\Delta \Delta G_e$ (kcal/mol)")
    legend = ax.legend(loc="upper left", fontsize=6.5, ncol=ncol, frameon=True, fancybox=True, framealpha=0.7)
    legend.set_zorder(100)
    ax.set_title("Stability distribution of every amino acid in the population", size=8)

    return ax


def plot_initial_amino_acid_stabilities(stability_table, omega, colours=None, ax=None):
# TODO: def plot_locus_stabilities(generation, stability_table, omega, colours=None, ax=None):

    fig = None
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 12)

    if colours is None:
        colours = default_colours

    clone_size = stability_table.shape[0]
    mean_stability_0 = np.mean(stability_table.values)
    locus_indicies = np.arange(1, clone_size+1)

    # Plot each column of stability_table as a separate dataseries against 0..N-1
    ax.plot(locus_indicies, stability_table.values, "o", color=colours["aa_0"], markersize=1)
    # ax.hlines(mean_stability_0, 1, clone_size,
    #           colors=colours["aa_0_mu"], linestyles="--", lw=3, zorder=20,
    #           label=rf"$\mu_0$ = {mean_stability_0:.2f}")
    ax.axhline(mean_stability_0, color=colours["aa_0_mu"], linestyle="--", lw=3, zorder=20,
               label=rf"$\mu_0$ = {mean_stability_0:.2f}")

    # if omega > -np.inf:
    #     ax.hlines(omega, 0, clone_size-1,
    #               colors=colours["omega"], linestyles="-", lw=3, zorder=20,
    #               label=r"$\Omega$ = {}".format(omega))

    ax.set_xlabel("Amino acid position")
    ax.set_ylabel(r"$\Delta \Delta G_e$ (kcal/mol)")
    legend = ax.legend(loc="upper left", fontsize=6.5, frameon=True, fancybox=True, framealpha=0.7)
    legend.set_zorder(100)
    ax.set_title(
        r"Stabiltiy distribution of $\Delta \Delta G_e$ (kcal/mol) matrix", size=8)
    return ax


def plot_generation_stability(generation, stabilities, stability_table, out_paths):
    """Plot stability tables for a generation."""

    clims = (np.floor(np.amin(stability_table.values)),
             np.ceil(np.amax(stability_table.values)))
    plot_phi_stability_table(generation, stabilities, clims, out_paths)


def plot_stability_distributions(generation, stabilities, stability_table, initial_stabilities=None, alpha=0.8, density_cap=None, colours=None, ax=None):

    # Set density_cap=None to remove y-axis limits
    fig = None
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 12)

    if colours is None:
        colours = default_colours

    mean_eps_stability = np.mean(stability_table.ravel())

    # sns.histplot(stability_table.ravel(),
    #                 color=colours['epsilon'], alpha=alpha,
    #                 stat='density', ax=ax,
    #                 label="\n".join([r"$\epsilon$ distribution", rf"$\mu_r^{{\epsilon}}$ = {mean_eps_stability:.2f}"]))
    sns.kdeplot(stability_table.ravel(),
                color=colours['epsilon'], alpha=alpha, fill=True, ax=ax,
                    label="\n".join([r"$\epsilon$ distribution", rf"$\mu_r^{{\epsilon}}$ = {mean_eps_stability:.2f}"]))
    # for g in generations:
        # sns.histplot(y=history[g].stabilities.values.ravel(),
        #             color=colours['aa_0'], alpha=alpha,
        #             stat='density', ax=ax,
        #             label=f"Generation {g} distribution")
    mean_gen_stability = np.mean(stabilities.ravel())

    # sns.histplot(stabilities.ravel(),
    #             color=colours['aa_g'], alpha=alpha,
    #             stat='density', ax=ax,
    #              label="\n".join([f"Generation {generation:,}", rf"$\mu_r^{{(g)}}$ = {mean_gen_stability:.2f}"]))
    sns.kdeplot(stabilities.ravel(),
                color=colours['aa_g'], alpha=alpha, fill=True, ax=ax,
                label="\n".join([f"Generation {generation:,}", rf"$\mu_r^{{(g)}}$ = {mean_gen_stability:.2f}"]))

    if initial_stabilities is not None:
        mean_initial_stability = np.mean(initial_stabilities.ravel())

        # sns.histplot(initial_stabilities.ravel(),
        #              color=colours['aa_0'], alpha=alpha,
        #              stat='density', ax=ax,
        #              label="\n".join(["Generation 0", rf"$\mu_r^{{(0)}}$ = {mean_initial_stability:.2f}"]))
        sns.kdeplot(initial_stabilities.ravel(),
                     color=colours['aa_0'], alpha=alpha, fill=True, ax=ax,
                     label="\n".join(["Generation 0", rf"$\mu_r^{{(0)}}$ = {mean_initial_stability:.2f}"]))

    
    ax.axvline(x=mean_eps_stability, color=colours["epsilon"],
               linestyle="--", lw=3, zorder=20,)
            #    label=rf"$\epsilon_r$ = {mean_eps_stability:.2f}")
    # means = [np.mean(history[g].stabilities.values) for g in generations]
    
    ax.axvline(x=mean_gen_stability, color=colours["aa_g_mu"],
               linestyle="--", lw=3, zorder=20,)
            #    label=rf"$\mu_r^{{(g)}}$ = {mean_gen_stability:.2f}")
    if initial_stabilities is not None:
        ax.axvline(x=mean_initial_stability, color=colours["aa_0_mu"],
                   linestyle="--", lw=3, zorder=20,)
                #    label=rf"$\mu_r^{{(0)}}$ = {mean_initial_stability:.2f}")

    legend = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.7, fontsize=8)  # fontsize=6.5,
    legend.set_zorder(100)

    ax.set_xlabel(r"$\Delta \Delta G_e$ (kcal/mol)")
    ax.set_ylabel("Density")
    ax.set_ybound(0, density_cap)

    # if fig is not None:
    #     fig.savefig(os.path.join(out_paths["figures"],
    #                              f"histogram_G{generation}.png"))
    return ax

def plot_stability_histograms(generation, aa_stabilities, stability_table, #omega,
                              out_paths, orient='vertical', density_cap=0.7,
                              colours=None, ax=None):

    # Set density_cap=None to remove y-axis limits
    fig = None
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 12)

    if colours is None:
        colours = default_colours

    mean_stability_0 = np.mean(stability_table.values)
    mean_stability = np.mean(aa_stabilities)

    # Plot initial distribution
    stats_0 = "\n".join([rf"$\sigma$ = {np.std(stability_table.values):.2f}",
                         f"skew = {stats.skew(stability_table.values, axis=None):.2f}",
                         f"kurtosis = {stats.kurtosis(stability_table.values, axis=None):.2f}"])
    # n, bins, _ = ax.hist(stability_table.values.ravel(), bins='sqrt',
    #                      align='mid', orientation=orient,
    #                      color=colours["aa_0"], alpha=0.8, density=True,
    #                      label="\n".join([r"$\epsilon$ distribution", stats_0]))
    if orient.lower() == 'vertical':
        # NOTE: The terminology is reversed
        vertical = False
        sns.histplot(stability_table.values.ravel(),
                     color=colours['aa_0'], alpha=0.8,
                     stat='density', ax=ax,
                     label="\n".join([r"$\epsilon$ distribution", stats_0]))
    else:
        vertical = True
        sns.histplot(y=stability_table.values.ravel(),
                     color=colours['aa_0'], alpha=0.8,
                     stat='density', ax=ax,
                     label="\n".join([r"$\epsilon$ distribution", stats_0]))
    # sns.distplot(stability_table.values.ravel(), #bins='sqrt', 
    #              color=colours['aa_0'], hist_kws={"alpha": 0.8}, 
    #              norm_hist=True, kde=False, ax=ax, vertical=vertical,
    #              label="\n".join([r"$\epsilon$ distribution", stats_0]))

    # Plot initial distribution mean
    label = rf"$\epsilon_r$ = {mean_stability_0:.2f}"
    if orient == 'vertical':
        ax.axvline(x=mean_stability_0, color=colours["aa_0_mu"],
                   linestyle="--", lw=3, zorder=20, label=label)
    elif orient == 'horizontal':
        ax.axhline(y=mean_stability_0, color=colours["aa_0_mu"],
                   linestyle="--", lw=3, zorder=20, label=label)

    # Plot current distribution
    # TODO: Make "Intial/Present Distribution" bold or titles of extra legends
    stats_g = "\n".join([rf"$\sigma$ = {np.std(aa_stabilities):.2f}",
                         f"skew = {stats.skew(aa_stabilities, axis=None):.2f}",
                         f"kurtosis = {stats.kurtosis(aa_stabilities, axis=None):.2f}"])
    # ax.hist(aa_stabilities.ravel(), bins=bins,
    #         align='mid', color=colours["aa_g"], alpha=0.8,
    #         orientation=orient, density=True,
    #         label=f"Present distribution\n{stats_g}")
    if orient.lower() == 'vertical':
        sns.histplot(aa_stabilities.ravel(),
                     color=colours['aa_g'], alpha=0.8,
                     stat='density', ax=ax,
                     label=f"Present distribution\n{stats_g}")
    else:
        sns.histplot(y=aa_stabilities.ravel(),
                     color=colours['aa_g'], alpha=0.8,
                     stat='density', ax=ax,
                     label=f"Present distribution\n{stats_g}")
    # sns.distplot(aa_stabilities.ravel(), #bins='sqrt', 
    #              color=colours['aa_g'], hist_kws={"alpha": 0.8}, 
    #              norm_hist=True, kde=False, ax=ax, vertical=vertical,
    #              label=f"Present distribution\n{stats_g}")
    # Plot current distribution mean
    label = rf"$\mu_r$ = {mean_stability:.2f}"
    if orient == 'vertical':
        ax.axvline(x=mean_stability, color=colours["aa_g_mu"],
                   linestyle="--", lw=3, zorder=20, label=label)
    elif orient == 'horizontal':
        ax.axhline(y=mean_stability, color=colours["aa_g_mu"],
                   linestyle="--", lw=3, zorder=20, label=label)

    # # Plot Omega
    # if omega > -np.inf:
    #     label = r"$\Omega$ = {}".format(omega)
    #     if orient == 'vertical':
    #         ax.axvline(x=omega, color=colours["omega"],
    #                    linestyle="-", lw=3, zorder=10, label=label)
    #     elif orient == 'horizontal':
    #         ax.axhline(y=omega, color=colours["omega"],
    #                    linestyle="-", lw=3, zorder=10, label=label)
    # Add legend
    legend = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.7, fontsize=8)  # fontsize=6.5,
    legend.set_zorder(100)

    if orient == 'vertical':
        ax.set_xlabel(r"$\Delta \Delta G_e$ (kcal/mol)")
        ax.set_ylabel("Distribution density")
        ax.set_ybound(0, density_cap)
    elif orient == 'horizontal':
        ax.set_ylabel(r"$\Delta \Delta G_e$ (kcal/mol)")
        ax.set_xlabel("Distribution density")
        ax.set_xbound(0, density_cap)

    # Set to 1.5*largest original bin count
    # ax.set_ylim(0, round(1.5*np.amax(n)))
    if fig is not None:
        ax.set_title(fill("Initial stability distribution vs. "
                        "changing stability distribution across "
                        "every evolving clone", 60),
                     fontweight='bold')
        fig.savefig(os.path.join(out_paths["figures"],
                                 f"histogram_G{generation}.png"))
    return ax


def plot_protein_stabilities(aa_stabilities, omega, epsilon, plot_epsilon, plot_mean=True,
                             colours=None, ax=None):

    fig = None
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 12)

    if colours is None:
        colours = default_colours

    (n_clones, clone_size) = aa_stabilities.shape
    protein_stabilities = np.sum(aa_stabilities, axis=1)
    protein_indicies = np.arange(1, n_clones+1)
    mean_protein_stability = np.mean(protein_stabilities)

    ax.plot(protein_indicies, protein_stabilities, "*",
            color=colours["phi"], markersize=4)

    if plot_mean:
        # ax.hlines(mean_protein_stability, 1, n_clones,
        #           colors=colours["phi_mu"], linestyles="--", lw=3, zorder=20,
        #           label=rf"$\mu_\phi$ = {mean_protein_stability:.2f}")
        ax.axhline(mean_protein_stability,
                color=colours["phi_mu"], linestyle="--", lw=3, zorder=20,
                label=rf"$\mu_\phi$ = {mean_protein_stability:.2f}")

    if plot_epsilon:
        # ax.hlines(epsilon, 1, n_clones,
        #           colors=colours["epsilon"], linestyles=":", lw=3, zorder=20,
        #           label=rf"$\epsilon_\phi$ = {epsilon:.2f}")
        ax.axhline(epsilon,
                    color=colours["epsilon"], linestyle=":", lw=3, zorder=20,
                    label=rf"$\epsilon_\phi$ = {epsilon:.2f}")
    ncol = 2
    if omega > -np.inf:
        # ax.hlines(omega, 1, n_clones,
        #           colors=colours["omega"], linestyles="-", lw=3, zorder=20,
        #           label=rf"$\Omega$ = {omega}")
        ax.axhline(omega,
                  color=colours["omega"], linestyle="-", lw=3, zorder=20,
                  label=rf"$\Omega$ = {omega}")
        ncol += 1

    ax.set_xlabel("Clone")
    ax.set_ylabel(r"$\Delta G_e$ (kcal/mol)")
    legend = ax.legend(loc="upper left", ncol=ncol, frameon=True, fancybox=True, framealpha=0.7)  # fontsize=6.5, 
    legend.set_zorder(100)
    return ax



def plot_all_stabilities(generation, history, stability_table, omega,
                         plot_omega, plot_epsilon, n_generations, out_paths, 
                         colours=None):

    if colours is None:
        colours = default_colours

    pad_factor = 0.1
    density_cap = 0.7

    aa_stabilities = history[generation].stabilities
    (n_clones, clone_size) = aa_stabilities.shape
    (clone_size, n_amino_acids) = stability_table.shape
    mean_stability_0 = np.mean(stability_table.values)
    epsilon = clone_size * mean_stability_0

    fig = plt.figure(figsize=(12, 8))  # (width, height)
    # https://matplotlib.org/users/gridspec.html
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=3,
                               width_ratios=[1, 1.618, 1],  # [Left, Middle, Right]
                               height_ratios=[2, 2])  # [Top, Bottom]
    gs.update(top=0.95, wspace=0.10, hspace=0.12)  # Leave room for the title

    # Plot initial distribution
    ax_aa_0 = plt.subplot(gs[1, 0])
    plot_initial_amino_acid_stabilities(stability_table, omega,
                                        colours=colours, ax=ax_aa_0)
    # Calculate \Delta \Delta G_e (amino acid) stability plotting bounds
    ymin, ymax = np.floor(np.amin(stability_table.values)), np.ceil(np.amax(stability_table.values))
    pad = pad_factor * abs(ymax - ymin)  # * 0.5
    ymax = np.ceil(ymax + pad)
    ymin = np.floor(ymin - pad)
    ax_aa_0.set_ylim(ymin, ymax)
    # plt.setp(ax_aa_0.get_yticklabels(), visible=False)
    # ax_aa_0.set_ylabel(None)
    ax_aa_0.set_title(None)
    ax_aa_0.legend_.remove()

    # Plot current generation's amino acid stabilities
    ax_aa_g = plt.subplot(gs[1, 1], sharey=ax_aa_0)
    plot_amino_acid_stabilities(aa_stabilities, mean_stability_0,
                                colours=colours, ax=ax_aa_g)
    plt.setp(ax_aa_g.get_yticklabels(), visible=False)
    ax_aa_g.set_ylabel(None)
    ax_aa_g.set_title(None)
    ax_aa_g.legend_.remove()

    # Plot marginal stability distributions
    ax_hist = plt.subplot(gs[1, -1], sharey=ax_aa_0)
    plot_stability_histograms(generation, aa_stabilities, stability_table, #omega,
                              out_paths, orient='horizontal', density_cap=density_cap,
                              colours=colours, ax=ax_hist)
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.set_ylabel(None)
    ax_hist.set_title(None)


    # Calculate \Delta G_e (protein) stability plotting bounds
    initial_protein_stabilities = np.sum(history[0].stabilities, axis=1)
    # NOTE: All clones in the initial population are currently identical
    DGe_max = sum(np.amax(stability_table, axis=1))  # Most unstable possible protein
    # DGe_min = sum(np.amin(stability_table, axis=1))  # Least unstable possible protein
    min_s0 = min(initial_protein_stabilities)
    max_s0 = max(initial_protein_stabilities)
    max_values = [epsilon, max_s0]  # , DGe_max * 0.1]
    if plot_omega and omega < np.inf:
        max_values.append(omega)
    ymax = max(max_values)
    # print(ymax, DGe_max)
    # if np.argmax(max_values) == 0:  # need to extend the range beyond epsilon
    #     ymax = DGe_max * 0.1
    min_values = [min_s0]
    if plot_omega and omega > -np.inf:
        min_values.append(omega)
    if plot_epsilon:
        min_values.append(epsilon)
    ymin = min(min_values)
    pad = pad_factor * abs(ymax - ymin)
    ymax = np.ceil(ymax + pad)
    ymin = np.floor(ymin - pad)

    protein_stabilities = np.sum(aa_stabilities, axis=1)
    mean_protein_stability = np.mean(protein_stabilities)

    # Plot initial protein stabilities
    ax_phi_0 = plt.subplot(gs[0, 0])
    plot_protein_stabilities(history[0].stabilities, omega, epsilon, plot_epsilon,
                             colours=colours, ax=ax_phi_0)
    plt.setp(ax_phi_0.get_xticklabels(), visible=False)
    ax_phi_0.set_xlabel(None)
    ax_phi_0.set_title(None)
    ax_phi_0.legend_.remove()
    # plt.setp(ax_phi_0.get_yticklabels(), visible=False)
    # ax_phi_0.set_ylabel(None)
    ax_phi_0.set_ylim(ymin, ymax)

    # Plot current generation's protein stabilities
    ax_phi = plt.subplot(gs[0, 1], sharex=ax_aa_g, sharey=ax_phi_0)
    plot_protein_stabilities(aa_stabilities, omega, epsilon, plot_epsilon,
                             colours=colours, ax=ax_phi)
    plt.setp(ax_phi.get_xticklabels(), visible=False)
    ax_phi.set_xlabel(None)
    ax_phi.set_title(None)
    ax_phi.legend_.remove()
    plt.setp(ax_phi.get_yticklabels(), visible=False)
    ax_phi.set_ylabel(None)

    # Plot marginal protein stability distributions
    ax_phi_hist = plt.subplot(gs[0, -1], sharex=ax_hist, sharey=ax_phi_0)
    # bins = np.linspace(ymin, ymax, int(np.sqrt(n_clones)))
    # ax_phi_hist.set_ylim(ymin, ymax)
    # ax_phi_hist.hist(protein_stabilities.ravel(), bins='sqrt', # bins, # 'sqrt',#int(np.sqrt(n_clones)),
    #                  color=colours['phi'], alpha=0.8,
    #                  align='mid', orientation='horizontal', density=True)
    sns.histplot(y=protein_stabilities.ravel(),
                 color=colours['phi'], alpha=0.8, stat='density', ax=ax_phi_hist)
    plt.setp(ax_phi_hist.get_xticklabels(), visible=False)
    plt.setp(ax_phi_hist.get_yticklabels(), visible=False)
    ax_phi_hist.set_xlabel(None)
    ax_phi_hist.set_ylabel(None)
    ax_phi_hist.set_xbound(0, density_cap)

    ax_phi_hist.axhline(y=mean_protein_stability, color=colours['phi_mu'],
                        linestyle="--", lw=3, zorder=20,
                        label=rf"$\mu_\phi$ = {mean_protein_stability:.2f}")
    mean_protein_stability_0 = np.mean(initial_protein_stabilities)
    ax_phi_hist.axhline(y=mean_protein_stability_0, color=colours['phi_0_mu'],
                        linestyle="-", lw=3, zorder=20,
                        label=rf"$\mu_{{\phi_0}}$ = {mean_protein_stability_0:.2f}")
    if plot_epsilon:
        ax_phi_hist.axhline(epsilon,
                  color=colours["epsilon"], linestyle=":", lw=3, zorder=20,
                  label=rf"$\epsilon$ = {epsilon:.2f}")
    
    if plot_omega:
        ax_phi_hist.axhline(y=omega, color=colours['omega'], linestyle="-", lw=3, zorder=20,
                            label=rf"$\Omega$ = {omega}")

    legend = ax_phi_hist.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.7)  # fontsize=6.5, 
    legend.set_zorder(100)

    # Add title and save
    # plt.subplots_adjust(top=0.85)
    fig.suptitle((f"Generation {generation}"), fontweight='bold')
    # fig.set_tight_layout(True)
    filename = os.path.join(out_paths["figures"], f"stabilities_G{generation}.png")
    fig.savefig(filename)
    plt.close()
    return (fig, [ax_phi_0, ax_phi, ax_phi_hist, 
                  ax_aa_0, ax_aa_g, ax_hist])


def plot_traces(generation, history, stability_table, omega,
                plot_omega, plot_epsilon, n_generations, out_paths, 
                plot_initial=False, colours=None):

    if colours is None:
        colours = default_colours

    pad_factor = 0.1
    density_cap = 0.7
    plot_evo_legend = False

    aa_stabilities = history[generation].stabilities
    (n_clones, clone_size) = aa_stabilities.shape
    (clone_size, n_amino_acids) = stability_table.shape
    epsilon_r = np.mean(stability_table.values)
    epsilon = clone_size * epsilon_r

    fig = plt.figure(figsize=(14, 8))  # (width, height)
    if plot_initial:
        ncols = 4
        width_ratios = [0.618, 1.618, 1, 0.618]
    else:
        ncols = 3
        width_ratios = [1.618, 1, 0.618]
    # https://matplotlib.org/users/gridspec.html
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=ncols,
                               # [Left, Middle, Right]
                               width_ratios=width_ratios,
                               height_ratios=[2, 2])  #  [Top, Bottom]
    # gs.update(top=0.95, wspace=0.08, hspace=0.08, left=0.05, right=0.95)  # Leave room for the title

    ax_botton_left = plt.subplot(gs[1, 0])
    ax_top_left = plt.subplot(gs[0, 0], sharex=ax_botton_left)

    if plot_initial:
        # Plot initial distribution: \Delta \Delta G_e vs locus
        ax_aa_0 = ax_botton_left
        plot_initial_amino_acid_stabilities(stability_table, omega,
                                            colours=colours, ax=ax_aa_0)
        ax_aa_0.set_title(None)
        ax_aa_0.legend_.remove()

    # Calculate \Delta \Delta G_e (amino acid) stability plotting bounds
    ymin = np.floor(np.amin(stability_table.values))
    ymax = np.ceil(np.amax(stability_table.values))
    pad = pad_factor * abs(ymax - ymin)  # * 0.5
    ymax = np.ceil(ymax + pad)
    ymin = np.floor(ymin - pad)
    ax_botton_left.set_ylim(ymin, ymax)
    # plt.setp(ax_aa_0.get_yticklabels(), visible=False)
    # ax_aa_0.set_ylabel(None)
    # ax_botton_left.set_title(None)
    # ax_botton_left.legend_.remove()

    # Plot stability trace (amino acid evoutionary history): \Delta \Delta G_e vs generation
    if plot_initial:
        ax_evo_aa = plt.subplot(gs[1, 1], sharey=ax_aa_0)
    else:
        ax_evo_aa = ax_botton_left
    gen_xlims = (-5, n_generations+5)
    plot_amino_acid_evolution(history, epsilon_r, out_paths,
                              fig_title=False, xlims=gen_xlims, colours=colours,
                              ax=ax_evo_aa)
    # Add a marker to show the current mean stability
    ax_evo_aa.plot(len(history)-1, np.mean(aa_stabilities), '*',
                    color=colours["aa_g_mu"], markersize=10)
    if not plot_evo_legend:
        ax_evo_aa.legend_.remove()
    
    if plot_initial:
        plt.setp(ax_evo_aa.get_yticklabels(), visible=False)
        ax_evo_aa.set_ylabel(None)

    # Plot current generation's amino acid stabilities: \Delta \Delta G_e vs clone
    # ax_aa_g = plt.subplot(gs[1, 2], sharey=ax_aa_0)
    if plot_initial:
        col = 2
    else:
        col = 1
    ax_aa_g = plt.subplot(gs[1, col], sharey=ax_botton_left)
    # if plot_initial:
    #     plot_amino_acid_stabilities(history[0].stabilities, epsilon_r,
    #                                 colours={"aa_g": colours["aa_0"], 
    #                                          "aa_g_mu": colours["aa_0_mu"]}, 
    #                                 ax=ax_aa_g)
    plot_amino_acid_stabilities(aa_stabilities, epsilon_r,
                                colours=colours, ax=ax_aa_g)
    plt.setp(ax_aa_g.get_yticklabels(), visible=False)
    ax_aa_g.set_ylabel(None)
    ax_aa_g.set_title(None)
    ax_aa_g.legend_.remove()

    # Plot marginal stability distributions: \Delta \Delta G_e vs density
    ax_hist = plt.subplot(gs[1, -1], sharey=ax_botton_left)
    plot_stability_histograms(generation, aa_stabilities, stability_table, #omega,
                              out_paths, orient='horizontal', density_cap=density_cap,
                              colours=colours, ax=ax_hist)
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.set_ylabel(None)
    ax_hist.set_title(None)


    # Calculate \Delta G_e (protein) stability plotting bounds
    initial_protein_stabilities = np.sum(history[0].stabilities, axis=1)
    # NOTE: All clones in the initial population are currently identical
    # Most unstable possible protein
    DGe_max = sum(np.amax(stability_table, axis=1))
    # DGe_min = sum(np.amin(stability_table, axis=1))  # Least unstable possible protein
    min_s0 = min(initial_protein_stabilities)
    max_s0 = max(initial_protein_stabilities)
    max_values = [epsilon, max_s0]  # , DGe_max * 0.1]
    if plot_omega and omega < np.inf:
        max_values.append(omega)
    ymax = max(max_values)
    # print(ymax, DGe_max)
    # if np.argmax(max_values) == 0:  # need to extend the range beyond epsilon
    #     ymax = DGe_max * 0.1
    min_values = [min_s0]
    if plot_omega and omega > -np.inf:
        min_values.append(omega)
    if plot_epsilon:
        min_values.append(epsilon)
    ymin = min(min_values)
    pad = pad_factor * abs(ymax - ymin)
    ymax = np.ceil(ymax + pad)
    ymin = np.floor(ymin - pad)

    protein_stabilities = np.sum(aa_stabilities, axis=1)
    mean_protein_stability = np.mean(protein_stabilities)

    if plot_initial:
        # Plot initial protein stabilities: \Delta G_e vs locus
        # ax_phi_0 = plt.subplot(gs[0, 0])
        ax_phi_0 = ax_top_left
        epsilon_r_colours = copy.copy(colours)
        epsilon_r_colours["phi"] = "#ff7f00"
        plot_protein_stabilities(stability_table, omega, epsilon, plot_epsilon,
                                plot_mean=False,  # Plot epsilon only
                                colours=epsilon_r_colours, ax=ax_phi_0)
        plt.setp(ax_phi_0.get_xticklabels(), visible=False)
        ax_phi_0.set_xlabel(None)
        ax_phi_0.set_title(None)
        ax_phi_0.legend_.remove()
        # plt.setp(ax_phi_0.get_yticklabels(), visible=False)
        # ax_phi_0.set_ylabel(None)
        # ax_phi_0.set_ylim(ymin, ymax)

    ax_top_left.set_ylim(ymin, ymax)
    if not plot_initial:
        ax_top_left.set_ylabel(r"$\Delta G_e$ (kcal/mol)")

    # Plot protein evolutionary history: \Delta G_e vs generation
    if plot_initial:
        ax_evo_phi = plt.subplot(gs[0, 1], sharex=ax_evo_aa, sharey=ax_phi_0)
    else:
        ax_evo_phi = ax_top_left
    plot_evolution(history, stability_table, omega, plot_omega, plot_epsilon,
                   out_paths, fig_title=False, xlims=gen_xlims, colours=colours,
                   ax=ax_evo_phi)
    # if not plot_initial:
    #     ax_evo_phi.set_ylabel(r"$\Delta G_e$ (kcal/mol)")
    # Add a marker to show the current mean stability
    ax_evo_phi.plot(len(history)-1, mean_protein_stability, '*',
                    color=colours["phi_mu"], markersize=10)  # ,
                # label=r"$\mu_\phi$ = {:.2f}".format(mean_protein_stability))
    if plot_evo_legend:
        handles, labels = ax_evo_phi.get_legend_handles_labels()
        mu_text_index = labels.index(r"$\mu_\phi$")
        new_label = rf"$\mu_\phi$ = {mean_protein_stability:.2f}"
        ax_evo_phi.legend_.get_texts()[mu_text_index].set_text(f"{new_label: <16}")
    else:
        ax_evo_phi.legend_.remove()
    if plot_initial:
        plt.setp(ax_evo_phi.get_yticklabels(), visible=False)
        ax_evo_phi.set_ylabel(None)
    # NOTE: ax.set_xticklabels([]) removes ticks entirely
    plt.setp(ax_evo_phi.get_xticklabels(), visible=False)
    ax_evo_phi.set_xlabel(None)


    # Plot current generation's protein stabilities: \Delta G_e vs clone
    if plot_initial:
        col = 2
    else:
        col = 1
    ax_phi = plt.subplot(gs[0, col], sharex=ax_aa_g, sharey=ax_top_left)
    plot_protein_stabilities(aa_stabilities, omega, epsilon, plot_epsilon,
                             colours=colours, ax=ax_phi)
    plt.setp(ax_phi.get_xticklabels(), visible=False)
    ax_phi.set_xlabel(None)
    ax_phi.set_title(None)
    ax_phi.legend_.remove()
    plt.setp(ax_phi.get_yticklabels(), visible=False)
    ax_phi.set_ylabel(None)

    # Plot marginal protein stability distributions: \Delta G_e vs density
    ax_phi_hist = plt.subplot(gs[0, -1], sharex=ax_hist, sharey=ax_top_left)
    # bins = np.linspace(ymin, ymax, int(np.sqrt(n_clones)))
    # ax_phi_hist.set_ylim(ymin, ymax)
    # ax_phi_hist.hist(protein_stabilities.ravel(), bins='sqrt',  # bins, # 'sqrt',#int(np.sqrt(n_clones)),
    #                  color=colours['phi'], alpha=0.8,
    #                  align='mid', orientation='horizontal', density=True, stacked=True)
    # sns.distplot(protein_stabilities.ravel(), #bins='sqrt', 
    #              color=colours['phi'], kde=False, hist_kws={"alpha": 0.8}, 
    #              vertical=True, norm_hist=True, ax=ax_phi_hist)
    sns.histplot(y=protein_stabilities.ravel(),
                 color=colours['phi'], alpha=0.8, stat='density', ax=ax_phi_hist)
    plt.setp(ax_phi_hist.get_xticklabels(), visible=False)
    plt.setp(ax_phi_hist.get_yticklabels(), visible=False)
    ax_phi_hist.set_xlabel(None)
    ax_phi_hist.set_ylabel(None)
    ax_phi_hist.set_xbound(0, density_cap)

    ax_phi_hist.axhline(y=mean_protein_stability, color=colours['phi_mu'],
                        linestyle="--", lw=3, zorder=10,
                        label=rf"$\mu_\phi$ = {mean_protein_stability:.2f}")
    mean_protein_stability_0 = np.mean(initial_protein_stabilities)
    ax_phi_hist.axhline(y=mean_protein_stability_0, color=colours['phi_0_mu'],
                        linestyle="-", lw=3, zorder=10,
                        label=rf"$\mu_{{\phi_0}}$ = {mean_protein_stability_0:.2f}")
    if plot_epsilon:
        ax_phi_hist.axhline(epsilon,
                            color=colours["epsilon"], linestyle=":", lw=3, zorder=20,
                            label=rf"$\epsilon_\phi$ = {epsilon:.2f}")

    if plot_omega:
        ax_phi_hist.axhline(y=omega, color=colours['omega'], linestyle="-", lw=3, zorder=20,
                            label=rf"$\Omega$ = {omega}")

    legend = ax_phi_hist.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.7)  # fontsize=6.5, 
    legend.set_zorder(100)

    # Add title and save
    # plt.subplots_adjust(top=0.85)
    # gs.tight_layout(fig, rect=(0, 0, 1, 1))
    fig.suptitle((f"Generation {generation}"), fontweight='bold')
    # fig.set_tight_layout(True)
    gs.tight_layout(fig, rect=(0.02, 0.02, 0.98, 0.95))  # Leave room for the title
    filename = os.path.join(
        out_paths["figures"], f"traces_G{generation}.png")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Creating legend with loc="best" can be slow with large amounts of data.')
        fig.savefig(filename)
    plt.close()
    if plot_initial:
        return (fig, [ax_phi_0, ax_evo_phi, ax_phi, ax_phi_hist,
                      ax_aa_0, ax_evo_aa, ax_aa_g, ax_hist])
    else:
        return (fig, [ax_evo_phi, ax_phi, ax_phi_hist,
                      ax_evo_aa, ax_aa_g, ax_hist])
    # return fig  # TODO: Check this contains all the axes


def plot_simulation(generation, history, stability_table, omega,
                    plot_omega, plot_epsilon, n_generations, out_paths, 
                    colours=None):

    if colours is None:
        colours = default_colours

    pad_factor = 0.1
    density_cap = 0.7

    aa_stabilities = history[generation].stabilities
    (n_clones, clone_size) = aa_stabilities.shape
    (clone_size, n_amino_acids) = stability_table.shape
    epsilon_r = np.mean(stability_table.values)
    epsilon = clone_size * epsilon_r

    fig = plt.figure(figsize=(12, 9))  # (width, height)
    # https://matplotlib.org/users/gridspec.html
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=3,
                               width_ratios=[1.618, 1, 1],  # [Left, Middle, Right]
                               height_ratios=[2, 2])  # [Top, Bottom]
    gs.update(top=0.95, wspace=0.08, hspace=0.16)  # Leave room for the title

    # Plot current generation's amino acid stabilities
    ax_aa_g = plt.subplot(gs[1, 0])
    plot_amino_acid_stabilities(aa_stabilities, epsilon_r,
                                colours=colours, ax=ax_aa_g)
    # Calculate \Delta \Delta G_e (amino acid) stability plotting bounds
    ymin, ymax = np.floor(np.amin(stability_table.values)), np.ceil(np.amax(stability_table.values))
    pad = pad_factor * abs(ymax - ymin)  # * 0.5
    ymax = np.ceil(ymax + pad)
    ymin = np.floor(ymin - pad)
    ax_aa_g.set_ylim(ymin, ymax)
    ax_aa_g.set_title(None)
    ax_aa_g.legend_.remove()

    # Plot initial distribution
    ax_aa_0 = plt.subplot(gs[1, 1], sharey=ax_aa_g)
    plot_initial_amino_acid_stabilities(stability_table, omega,
                                        colours=colours, ax=ax_aa_0)
    plt.setp(ax_aa_0.get_yticklabels(), visible=False)
    ax_aa_0.set_ylabel(None)
    ax_aa_0.set_title(None)
    ax_aa_0.legend_.remove()

    # Plot marginal stability distributions
    ax_hist = plt.subplot(gs[1, -1], sharey=ax_aa_g)
    plot_stability_histograms(generation, aa_stabilities, stability_table, #omega,
                              out_paths, orient='horizontal', density_cap=density_cap,
                              colours=colours, ax=ax_hist)
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.set_ylabel(None)
    ax_hist.set_title(None)

    # Find and plot all stability values in the current generation
    # x: proteins within population; y: stability for each locus for that protein
    ax_phi = plt.subplot(gs[0, 0], sharex=ax_aa_g)
    plot_protein_stabilities(aa_stabilities, omega, epsilon, plot_epsilon,
                             colours=colours, ax=ax_phi)
    plt.setp(ax_phi.get_xticklabels(), visible=False)
    ax_phi.set_xlabel(None)
    ax_phi.set_title(None)
    ax_phi.legend_.remove()

    # Calculate \Delta G_e (protein) stability plotting bounds
    initial_protein_stabilities = np.sum(history[0].stabilities, axis=1)
    # NOTE: All clones in the initial population are currently identical
    # This could be extended to the first few generations
    DGe_max = sum(np.amax(stability_table, axis=1))  # Most unstable possible protein
    # DGe_min = sum(np.amin(stability_table, axis=1))  # Least unstable possible protein
    min_s0 = min(initial_protein_stabilities)
    max_s0 = max(initial_protein_stabilities)
    max_values = [epsilon, max_s0]  # , DGe_max * 0.1]
    if plot_omega and omega < np.inf:
        max_values.append(omega)
    ymax = max(max_values)
    # print(ymax, DGe_max)
    # if np.argmax(max_values) == 0:  # need to extend the range beyond epsilon
    #     ymax = DGe_max * 0.1
    min_values = [min_s0]
    if plot_omega and omega > -np.inf:
        min_values.append(omega)
    if plot_epsilon:
        min_values.append(epsilon)
    ymin = min(min_values)
    pad = pad_factor * abs(ymax - ymin)
    ymax = np.ceil(ymax + pad)
    ymin = np.floor(ymin - pad)
    ax_phi.set_ylim(ymin, ymax)

    # Plot evolutionary history
    protein_stabilities = np.sum(aa_stabilities, axis=1)
    mean_protein_stability = np.mean(protein_stabilities)

    ax_evo = plt.subplot(gs[0, 1:], sharey=ax_phi)
    xlims = (-5, n_generations+5)
    plot_evolution(history, stability_table, omega, plot_omega, plot_epsilon,
                   out_paths, fig_title=False, xlims=xlims, colours=colours,
                   ax=ax_evo)
    # Add a marker to show the current mean stability
    ax_evo.plot(len(history)-1, mean_protein_stability, '*',
                color=colours["phi_mu"], markersize=10)  # ,
                # label=r"$\mu_\phi$ = {:.2f}".format(mean_protein_stability))
    handles, labels = ax_evo.get_legend_handles_labels()
    mu_text_index = labels.index(r"$\mu_\phi$")
    new_label = rf"$\mu_\phi$ = {mean_protein_stability:.2f}"
    ax_evo.legend_.get_texts()[mu_text_index].set_text(f"{new_label: <16}")
    plt.setp(ax_evo.get_yticklabels(), visible=False)
    # NOTE: ax.set_xticklabels([]) removes ticks entirely
    ax_evo.set_ylabel(None)

    # Add title and save
    # plt.subplots_adjust(top=0.85)
    fig.suptitle((f"Generation {generation}"), fontweight='bold')
    # fig.set_tight_layout(True)
    filename = os.path.join(out_paths["figures"],
                            f"pesst_G{generation}.png")
    fig.savefig(filename)
    plt.close()
    return (fig, [ax_phi, ax_evo, 
                  ax_aa_g, ax_aa_0, ax_hist])


def plot_gamma_distribution(gamma, samples, quartiles, average_medians,
                            out_paths, ax=None):
    """Plot the distribution along with the quartiles and medians."""

    fig = None
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 12)

    kappa, theta, n_iterations, n_samples = (gamma["shape"],
                                             gamma["scale"],
                                             gamma["iterations"],
                                             gamma["samples"])

    x = np.linspace(0, 6, 1000)
    # y = x**(kappa - 1) * (np.exp(-x / theta)
    #                         / (stats.gamma(kappa).pdf(x, kappa)))  # * theta ** kappa))
    y = stats.gamma.pdf(x, kappa, scale=theta)
    ax.plot(x, y, linewidth=2, color='k', alpha=0,
            label="\n".join([rf"$\kappa$ = {kappa:.2f}",
                             rf"$\theta$ = {theta:.2f}"]))
    ax.hist(samples, bins=int(np.sqrt(len(samples))), range=(0, 6),
            density=True, color='g', histtype='step')
    ax.fill_between(x, y, where=x > quartiles[0], color='#4c4cff')
    ax.fill_between(x, y, where=x > quartiles[1], color='#7f7fff')
    ax.fill_between(x, y, where=x > quartiles[2], color='#b2b2ff')
    ax.fill_between(x, y, where=x > quartiles[3], color='#e5e5ff')
    ax.axvline(x=average_medians[0], color="#404040", linestyle=":")
    ax.axvline(x=average_medians[1], color="#404040", linestyle=":")
    ax.axvline(x=average_medians[2], color="#404040", linestyle=":")
    ax.axvline(x=average_medians[3], color="#404040", linestyle=":")
    ax.set_title(fill("Gamma rate categories calculated as the the "
                    f"average of {n_iterations} median values of "
                    f"4 equally likely quartiles of {n_samples:,} "
                    "randomly sampled vaules", 60),
                 fontweight='bold', fontsize=10)
    legend = ax.legend()
    legend.set_zorder(100)

    if fig:
        fig.savefig(os.path.join(out_paths["initial"], "gamma.png"))
        plt.close()


def plot_stability_table(stability_table, out_paths):

    (clone_size, n_amino_acids) = stability_table.shape
    fraction = 0.1
    pad = 0.05
    cpi = 4  # cells per inch
    w_leg = (fraction + pad) * n_amino_acids / cpi  # legend width
    fig, ax = plt.subplots(figsize=(w_leg + n_amino_acids/cpi, clone_size/(cpi+3)))
    sns.heatmap(stability_table, center=0, annot=True, fmt=".2f", linewidths=.5,
                cmap="RdBu_r", annot_kws={"size": 5},
                cbar_kws={
                    "label": r"$\Delta \Delta G_e$ (kcal/mol)", "fraction": fraction, "pad": pad},
                ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Location")
    ax.set_title(
        r"Amino Acid stability contributions ($\Delta \Delta G_e$ (kcal/mol))")
    fig.set_tight_layout(True)
    filename = os.path.join(out_paths["initial"], "stability_table.png")
    fig.savefig(filename)
    plt.close()


def plot_LG_matrix(LG_matrix, out_paths):

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(LG_matrix, annot=True, fmt=".2f", linewidths=.5, cmap="cubehelix_r",
                square=True, annot_kws={"size": 5},  # vmin=0, vmax=1,
                ax=ax)  # cbar_kws={"label": "Transition probabiltiy"},
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Amino Acid")
    ax.set_title("LG model transition probabilities")
    fig.set_tight_layout(True)
    filename = os.path.join(out_paths["initial"], "LG_matrix.png")
    fig.savefig(filename)
    plt.close()


def plot_phi_stability_table(generation, phi_stability_table, clims, out_paths):
    r"""Plot a heatmap of changes in stability (\Delta \Delta G_e (kcal/mol))
    for each amino acid in each protein."""
    (n_proteins, clone_size) = phi_stability_table.shape
    fraction = 0.1
    pad = 0.05
    cpi = 8  # cells per inch
    w_leg = (fraction + pad) * clone_size / (cpi+2)  # legend width
    # width, height = clone_size/(cpi+2), n_proteins/cpi
    fig, ax = plt.subplots(figsize=(w_leg + clone_size/(cpi+2), n_proteins/cpi))
    sns.heatmap(phi_stability_table, center=0, annot=False, fmt=".2f",
                linewidths=.5, cmap="RdBu_r", annot_kws={"size": 5},
                cbar_kws={
                    "label": r"$\Delta \Delta G_e$ (kcal/mol)", "fraction": fraction, "pad": pad},
                vmin=clims[0], vmax=clims[1], ax=ax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel("Location")
    ax.set_ylabel("Clone")
    filename = os.path.join(out_paths["figures"],
                            f"phi_stability_table_G{generation}.png")
    ax.set_title(f"Generation {generation}", fontweight="bold")
    fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close()





# TODO: Replace references to these old plots in the code where individual figures are still required

# def plot_threshold_stability(generation, population, stabilities, stability_table,
#                            omega, out_paths):
#     # Store stability values for each amino in the dataset for the left subfigure
#     (clone_size, n_amino_acids) = stability_table.shape
#     # Average across flattened array
#     mean_initial_stability = np.mean(stability_table.values)
#     scale = round((4 * np.std(stability_table.values)) + 1)

#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
#                                    gridspec_kw={'width_ratios': [1, 2]})
#     # Plot each column of stability_table as a separate dataseries against 0..N-1
#     ax1.plot(stability_table, "o", color='k', markersize=1)
#     ax1.hlines(mean_initial_stability, 0, clone_size-1,
#                colors="r", linestyles="--", lw=4, zorder=10,
#                label=r"$\mu_0$ = {:.2f}".format(mean_initial_stability))
#     if omega > -np.inf:
#         ax1.hlines(omega, 0, clone_size-1,
#                    colors="k", linestyles="-", lw=4, zorder=10,
#                    label=r"$\Omega$ = {}".format(omega))
#     ax1.set_ylim(-scale, scale)
#     ax1.set_ylabel(r"$\Delta T_m$")
#     ax1.set_xlabel("Amino acid position")
#     ax1.legend(loc="upper left", fontsize=6.5)
#                # title=r"$\Omega$ = {}".format(omega))
#     ax1.set_title(r"stability distribution of $\Delta T_m$ matrix", size=8)

#     # Find and plot all stability values in the current generation
#     mean_stability = np.mean(stabilities)
#     # x: proteins within population; y: stability for each locus for that protein
#     # TODO: Swap colour to a particular protein not locus or make monochrome
#     ax2.plot(np.arange(len(population)), stabilities, "o", markersize=1)
#     # ax2.plot(np.arange(len(population)), np.sum(stabilities, axis=1), "*k", markersize=4, label=r"$\mu_p$")
#     ax2.hlines(mean_stability, 0, len(population)-1,
#                colors="r", linestyles="--", lw=4, zorder=10,
#                label=r"$\mu_p$ = {:.2f}".format(mean_stability))  # \mu_\phi ?
#     if omega > -np.inf:
#         ax2.hlines(omega, 0, len(population)-1,
#                    colors="k", linestyles="-", lw=4, zorder=10,
#                    label=r"$\Omega$ = {}".format(omega))
#     ax2.set_ylim(-scale, scale)
#     ax2.set_xlabel("Clone")
#     ax2.legend(loc="upper left", fontsize=6.5)
#                # title=r"$\Omega$ = {}".format(omega))
#     # ax2.set_title("\n".join(wrap("stability distribution of every sequence in "
#     #                              "the evolving dataset", 40)), size=8)
#     ax2.set_title("stability distribution of every protein in the population",
#                   size=8)

#     # plt.subplots_adjust(top=0.85)
#     fig.suptitle(("Generation {}".format(generation)), fontweight='bold')
#     # fig.tight_layout()
#     filename = os.path.join(out_paths["figures"], "OLD_generation_{}.png".format(generation))
#     fig.savefig(filename)
#     plt.close()


# def plot_stability_space(generation, population, stabilities, stability_table,
#                        omega, out_paths):

#     col_aa_g = "#3498db"  # Blue
#     col_aa_0 = "#95a5a6"  # Green
#     col_phi = "#34495e"
#     col_mu_phi = "#9b59b6"  # Purple
#     col_omega = "#e74c3c"  # "k"
#     col_epsilon = "#2ecc71"  # Red

#     # Store stability values for each amino in the dataset for the left subfigure
#     (clone_size, n_amino_acids) = stability_table.shape
#     # Average across flattened array
#     # mean_initial_stability = np.mean(stability_table.values)
#     scale = round((4 * np.std(stability_table.values)) + 1)
#     loc = np.mean(stability_table.values)
#     T_max = sum(np.amax(stability_table, axis=1))  # Most stable protein possible

#     fig, (ax_arr) = plt.subplots(2, 2, sharey='row', #  sharex='col',
#                                  gridspec_kw={'width_ratios': [4, 1],
#                                               'height_ratios': [1, 3]})
#     # Plot each column of stability_table as a separate dataseries against 0..N-1
#     # ax1.plot(stability_table, "o", color='k', markersize=1)
#     # ax1.hlines(mean_initial_stability, 0, clone_size-1,
#     #            colors="r", linestyles="--", lw=2,
#     #            label=r"$\mu_1$ = {:.2f}".format(mean_initial_stability))
#     # if omega > -np.inf:
#     #     ax1.hlines(omega, 0, clone_size-1,
#     #                colors="k", linestyles="-", lw=2,
#     #                label=r"$\Omega$ = {}".format(omega))
#     # ax1.set_ylim(-scale, scale)
#     # ax1.set_ylabel(r"$\Delta T_m$")
#     # ax1.set_xlabel("Amino acid position")
#     # ax1.legend(loc="upper left", fontsize=6.5,)
#                # title=r"$\Omega$ = {}".format(omega))
#     # ax1.set_title(r"stability distribution of $\Delta T_m$ matrix", size=8)

#     # Find and plot all stability values in the current generation
#     mean_stability = np.mean(stabilities)
#     # x: proteins within population; y: stability for each locus for that protein
#     # TODO: Swap colour to a particular protein not locus or make monochrome
#     protein_indicies = np.arange(len(population))
#     protein_stabilities = np.sum(stabilities, axis=1)
#     ax_arr[0, 0].plot(protein_indicies, protein_stabilities, "*", color=col_phi, markersize=4)  # , label=r"$\mu_p$")
#     ax_arr[0, 1].hist(protein_stabilities, bins=int(np.sqrt(len(population))),
#                       align='mid', orientation='horizontal', density=True)
#     # plt.setp(ax_arr[0, 1].get_xticklabels(), visible=False)
#     ax_arr[0, 1].set_xticks([])
#     mean_protein_stability = np.mean(protein_stabilities)
#     ax_arr[0, 0].hlines(mean_protein_stability, 0, len(population)-1,
#                         colors=col_mu_phi, linestyles="--", lw=3, zorder=10,
#                         label=r"$\mu_\phi$ = {:.2f}".format(mean_stability))  # \mu_\phi ?
#     mean_initial_amino_acid_stability = np.mean(stability_table.values)
#     epsilon = clone_size * mean_initial_amino_acid_stability
#     ax_arr[0, 0].hlines(epsilon, 0, len(population)-1,
#                         colors=col_epsilon, linestyles=":", lw=3, zorder=10,
#                         label=r"$\epsilon$ = {:.2f}".format(epsilon))
#     ncol = 2
#     if omega > -np.inf:
#         ax_arr[0, 0].hlines(omega, 0, len(population)-1,
#                             colors=col_omega, linestyles="-", lw=3, zorder=10,
#                             label=r"$\Omega$ = {}".format(omega))
#         ncol += 1

#     ax_arr[0, 0].set_ylabel(r"$T_m$")
#     plt.setp(ax_arr[0, 0].get_xticklabels(), visible=False)
#     ax_arr[0, 0].legend(loc="upper left", fontsize=6.5, ncol=ncol)

#     ax_arr[1, 0].plot(protein_indicies, stabilities, "o", color=col_aa_g, markersize=1)
#     n, bins, _ = ax_arr[1, 1].hist(stability_table.values.ravel(),
#                                    bins='sqrt',  # int(np.sqrt(stabilities.size))
#                                    color=col_aa_0, alpha=0.4, align='mid',
#                                    orientation='horizontal', density=True,
#                                    label="Initial dist.")
#     ax_arr[1, 1].axhline(y=mean_initial_amino_acid_stability, color=col_aa_0,
#                          linestyle="--", lw=3, zorder=10)
#                          # label=r"$\mu_0$ = {:.2f}".format(mean_initial_amino_acid_stability))
#     ax_arr[1, 1].hist(stabilities.ravel(), bins=bins,  # int(np.sqrt(stabilities.size))
#                       align='mid', color=col_aa_g, alpha=0.4,
#                       orientation='horizontal', density=True, label="Present dist.")
#     ax_arr[1, 1].axhline(y=mean_stability, color=col_aa_g, linestyle="--", lw=3, zorder=10)
#                          # label=r"$\mu_p$ = {:.2f}".format(mean_stability))
#     ax_arr[1, 1].axhline(y=omega, color=col_omega, linestyle="-", lw=3, zorder=10)
#                          # label=r"$\Omega$ = {}".format(omega))
#     ax_arr[1, 1].legend(loc="upper left", fontsize=6.5)
#     # plt.setp(ax_arr[1, 1].get_xticklabels(), visible=False)
#     ax_arr[1, 1].set_xticks([])
#     # Set to 1.5*largest original bin count
#     # ax_arr[1, 1].set_ylim(0, round(1.5*np.amax(n)))
#     ax_arr[1, 1].set_ybound(0, 0.5)

#     ax_arr[1, 0].hlines(mean_initial_amino_acid_stability, 0, len(population)-1,
#                         colors=col_aa_0, linestyles="--", lw=3, zorder=10,
#                         label=r"$\mu_0$ = {:.2f}".format(mean_initial_amino_acid_stability))
#     ax_arr[1, 0].hlines(mean_stability, 0, len(population)-1,
#                         colors=col_aa_g, linestyles="--", lw=3, zorder=10,
#                         label=r"$\mu_p$ = {:.2f}".format(mean_stability))  # \mu_\phi ?
#     if omega > -np.inf:
#         ax_arr[1, 0].hlines(omega, 0, len(population)-1,
#                             colors=col_omega, linestyles="-", lw=3, zorder=10,
#                             label=r"$\Omega$ = {}".format(omega))

#     ax_arr[0, 0].set_ylim(None, round(T_max))
#     ax_arr[1, 0].set_ylim(loc-scale, loc+scale)
#     ax_arr[1, 0].set_xlabel("Clone")
#     # ax_arr[1, 1].set_xlabel("Density")
#     ax_arr[1, 0].set_ylabel(r"$\Delta T_m$")
#     ax_arr[1, 0].legend(loc="upper left", fontsize=6.5, ncol=ncol)
#                # title=r"$\Omega$ = {}".format(omega))

#     # ax_arr[1, 0].set_title("stability distribution of every protein in the population",
#     #               size=8)

#     # plt.subplots_adjust(top=0.85)
#     fig.suptitle(("Generation {}".format(generation)), fontweight='bold')  # TODO Prevent overlap with spines
#     fig.set_tight_layout(True)
#     filename = os.path.join(out_paths["figures"], "OLD_stable_dist_gen_{}.png".format(generation))
#     fig.savefig(filename)
#     plt.close()
#     return (fig, ax_arr)


# def plot_histogram_of_stability(generation, distributions, initial, omega, out_paths):
#     plt.figure()
#     # plt.axis([-10, 8, 0, 0.5])  # generate attractive figure

#     # Plot normal distribution of the original stability space
#     mu1distspace = sum(initial) / len(initial)
#     plt.hist(initial, 50, density=True, color='k', alpha=0.4)
#     # plt.title("\n".join(wrap('stability distribution of the total stability space',
#     #                          60)), fontweight='bold')
#     # plt.axvline(x=mu1distspace, color="#404040", linestyle=":")
#     plt.axvline(x=mu1distspace, color="k", linestyle=":",
#                 label=r"$\mu_0$ = {:.3}".format(mu1distspace))

#     # Plot normal distribution of the current generation's stability space
#     mu2distspace = sum(distributions) / len(distributions)
#     plt.hist(distributions, 50, density=True, color='r', alpha=0.4)
#     plt.title("\n".join(wrap("stability distribution of the total stability space "
#                              "vs. changing stability distribution across every "
#                              "evolving clone", 60)), fontweight='bold')
#     # plt.axvline(x=mu2distspace, color="#404040", linestyle=":")
#     plt.axvline(x=mu2distspace, color="r", linestyle=":",
#                 label=r"$\mu_p$ = {:.3}".format(mu2distspace))

#     if omega > -np.inf:
#         plt.axvline(x=omega, color="k", linestyle="-",
#                     label=r"$\Omega$ = {}".format(omega))
#     plt.legend()
#     # plt.text(4.1, 0.4, "\n".join([r"$\mu_1$ = {:.3}".format(mu1distspace),
#     #                               r"$\mu_2$ = {:.3}".format(mu2distspace),
#     #                               r"$\Omega$ = {}".format(omega)]))

#     plt.savefig(os.path.join(out_paths["figures"], "OLD_histogram_{}.png".format(generation)))
#     plt.close()
