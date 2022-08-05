import copy
import os
import sys
# from pprint import pprint
from collections import namedtuple
import warnings
import random
# from random import sample, choice, shuffle  # TODO: Consolidate with numpy

import numpy as np
# from numpy.random import normal, uniform, gamma
import scipy as sp
from scipy import stats
import pandas as pd
# import matplotlib as mpl
from matplotlib import pyplot as plt  # TODO: Move into plotting.py
from tqdm import trange

from .dataio import (create_output_folders, save_settings, write_tree,
                     write_roots, write_initial_protein, write_stability_table,
                     write_fasta_alignment, write_final_fasta, load_LG_matrix,
                     write_histogram_statistics, append_ks_statistics,
                     create_gif, save_history, Generation)
from .plotting import (plot_simulation, plot_evolution, plot_gamma_distribution,
                       plot_generation_stability, plot_stability_histograms,
                       plot_all_stabilities, plot_stability_table,
                       plot_LG_matrix, plot_traces)
from .utilities import print_protein


def get_stability_table(clone_size, amino_acids, distributions):
    """Generate a dataframe describing the stability contributions for each
    amino acid (columns) in each position (rows) of the generated protein.
    """

    assert len(distributions)

    if isinstance(distributions, (tuple, list)) \
        and isinstance(distributions[0], (int, float, complex)):
        # Boilerplate code to wrap a single set of parameters in a list
        distributions = [distributions]

    # values = np.zeros(shape=(clone_size, len(amino_acids)))
    values = []
    remaining_residues = clone_size
    for d, params in enumerate(distributions):
        if isinstance(params, dict):
            assert 'mu' in params
            assert 'sigma' in params
            mu, sigma = params['mu'], params['sigma']
            skew = params.get("skew", 0)
            proportion = params.get("proportion", 1/len(distributions))
        elif isinstance(params, (list, tuple)):
            if len(params) == 2:
                (mu, sigma) = params
                skew = 0
                proportion = 1/len(distributions)
            elif len(params) == 3:
                (mu, sigma, skew) = params
                proportion = 1/len(distributions)
            elif len(params) == 4:
                (mu, sigma, skew, proportion) = params
            else:
                raise RuntimeError(
                    f'Unexpected number of parameters passed: {len(params)}'
                    'Expected: 2 <= len(params) < 5')
        else:
            raise RuntimeError("Unexpected data format for distributions parameters!"
                               f"Expected dictionary, tuple or list (got {type(params)})!")
        if d < len(distributions)-1:
            n_residues = int(np.round(proportion * clone_size))
            remaining_residues -= n_residues
        else:  # Last distribution
            n_residues = remaining_residues
        print(f'Distribution {d}: mu={mu}, sigma={sigma}, skew={skew}, '
              f'proportion={proportion:.2f} ==> n_residues={n_residues}')
        values.append(sp.stats.skewnorm.rvs(skew, loc=mu, scale=sigma,
                                            size=(n_residues, len(amino_acids))))

    values = np.concatenate(values, axis=0)  # vstack
    assert (clone_size, len(amino_acids)) == values.shape, f"Unexpect shape: {values.shape}"
    # Shuffle rows in place
    np.random.shuffle(values)
    return pd.DataFrame(values, columns=amino_acids)


def clone_protein(protein, n_clones):
    """Generate a dictionary containing n_clones of an initial protein."""
    return {l: copy.deepcopy(protein) for l in range(n_clones)}


def get_allowed_sites(clone_size, n_invariants):
    """Select invariant sites in the initially generated protein and return
    allowed values.
    """
    variant_sites = list(range(1, clone_size))  # keys for mutable sites
    # Randomly define invariant sites (without replacement)
    invariant_sites = random.sample(variant_sites, n_invariants)
    invariant_sites.insert(0, 0)  # First aa is always anchored (Methionine)
    # Remove the invariant sites from allowed values
    for a in invariant_sites[1:]:
        variant_sites.remove(a)
    # invariant = [variant_sites.pop(random.randrange(len(variant_sites)))
    #              for r in range(n_invariants)]
    Sites = namedtuple('Sites', ['invariant', 'variant'])
    # Return a namedtuple with anchors and available sites
    return Sites(invariant=invariant_sites, variant=variant_sites)


def gamma_ray(clone_size, sites, gamma, record, out_paths):
    """Generate a set of gamma rate categories.

    Does so by sampling many times from a gamma distribution.
    Tests of the trade-off between computing time and variance led me to set
    this to 10,000 samples from the distribution.
    Computes quartiles from the data with equal likelihood by finding bounds
    for quartiles and collecting values between bounds.
    Finds discrete rate values by taking the median of sorted collected values.
    Then iterates for a predefined set of runs, recording the median values.
    Tests of the tradeoff between computing time and variance led me to set
    this to 100 independent runs (1 million total samples from distribution).
    """

    kappa, theta, n_iterations, n_samples = (gamma["shape"],
                                             gamma["scale"],
                                             gamma["iterations"],
                                             gamma["samples"])

    medians = np.zeros(shape=(n_iterations, 4))

    for i in range(n_iterations):
        # Draw n_samples from the gamma distribution
        # samples = np.random.gamma(kappa, theta, n_samples)
        samples = sp.stats.gamma.rvs(kappa, scale=theta, size=n_samples)
        # Define quartiles in that data with equal probability
        quartiles = np.percentile(samples, (0, 25, 50, 75, 100),
                                  interpolation='midpoint')
        # Find the median of each quartile
        medians[i, :], _, _ = sp.stats.binned_statistic(samples, samples,
                                                        statistic='median',
                                                        bins=quartiles)
    # Calculate average of medians across iterations
    average_medians = np.mean(medians, axis=0)

    # TODO: Move plotting out
    if record["figures"]:
        # Plot the gamma distribution as a check
        plot_gamma_distribution(gamma, samples, quartiles, average_medians, out_paths)

    gamma_categories = np.random.choice(average_medians, size=clone_size)
    gamma_categories[sites.invariant] = 0
    return gamma_categories / sum(gamma_categories)  # p_mutation


def mutate_protein(protein, p_mutation, LG_matrix):
    """Mutate a residue to another residue based on the LG matrix."""
    mutant = copy.deepcopy(protein)  # Necessary!
    location = np.random.choice(len(mutant), p=p_mutation)
    amino_acid = mutant[location]
    p_transition = LG_matrix.loc[amino_acid]
    LG_residues = LG_matrix.columns.values  # .tolist()
    mutant[location] = np.random.choice(LG_residues, p=p_transition)
    return mutant


def get_amino_acid_stabilities(protein, stability_table):
    """Collate the individual stability contributions of each amino acid in a
    protein."""
    return [stability_table.loc[loc, aa] for loc, aa in enumerate(protein)]


def calculate_stability(protein, stability_table):
    """Calculate stability of a protein given the sequence and stability values."""
    return sum(get_amino_acid_stabilities(protein, stability_table))


def get_random_protein(clone_size, stability_table, start_amino_acid="M"):
    """Generate an original starting protein clone_size long with a start
    amino acid set to methionine.
    """
    protein = random.choices(stability_table.columns.values.tolist(),
                             k=clone_size-1)
    # protein = [random.choice(RESIDUES) for _ in range(clone_size)] # < 3.6
    protein.insert(0, start_amino_acid)  # Start with methionine
    # TODO: Convert to strings preventing in-place modification
    # return ''.join(protein)
    return protein


def twist_protein(protein, mutation_sites, stability_table):
    """Randomly change one variant site and calculate new stability."""
    mutant = protein[:]
    amino_acids = stability_table.columns.values.tolist()
    for ai in mutation_sites:
        mutant[ai] = random.choice(amino_acids)
    stability = calculate_stability(mutant, stability_table)
    return (mutant, stability)


def get_stable_protein(stability_start, clone_size, sites, stability_table, omega):
    """Generate a protein of a specified stability.

    Pass a string to make either a 'high', 'mid', 'low' or 'marginal'
    stability protein. Alternatively, pass a tuple or list where the target
    stability is specified as (lower_bound, upper_bound).

    This function currently locks in invariant sites before finding the most
    stable sites, meaning the invariant sites are simply sampled from the
    normal distribution, and not from the superstable distribution.

    Generating the protein in this manner avoids bias towards increased
    stability that could be generated by the invariant sites.
    """
    tail_length = 3  # Number of most extreme mutations to consider
    current_protein = get_random_protein(clone_size, stability_table)
    amino_acids = stability_table.columns.values

    if isinstance(stability_start, str):  # not medium
        # NOTE: With (\Delta) \Delta G_e, positive is bad (destabilising) and negative is good (stabilising)

        if stability_start.lower() == 'high':  # generate super stable protein
            ranks = list(range(tail_length))  # [0, 1, 2]  # Three lowest (least destabilising)

        elif stability_start.lower() == 'mid':
            trim_length = tail_length - 1  # TODO: Remove - this is inelegant but needed to reproduce Nic's results
            ranks = list(range(trim_length, len(amino_acids) - trim_length))  # All except the tails

        elif stability_start.lower() == 'low':  # generate super unstable protein
            # TODO: Adjust threshold or allow building proteins above threshold
            ranks = list(range(-1, -1 - tail_length, -1))  # [-1, -2, -3]  # Three highest (most destabilising)

        elif stability_start.lower() == 'marginal':
            return get_bounded_stability_protein((None, omega), current_protein,
                                                 sites, stability_table)

        else:
            warnings.warn(
                f"Unknown string passed {stability_start=}."
                "Valid options are 'high', 'mid', 'low' or 'marginal'."
                "Alternatively a 2-tuple specifying (lower_bound, upper_bound)"
                "may be passed."
                )
            sys.exit(1)

        # Nested list of candidates amino acids for each position
        pool = [["M"] * tail_length]
        for locus in range(1, len(stability_table.index)):  # range(clone_size):
            if locus in sites.invariant:
                pool.append([current_protein[locus]] * tail_length)
            else:
                sorted_aa = amino_acids[stability_table.loc[locus].argsort()]
                # sorted_aa = stability_table.sort_values(stability_table.loc[locus], axis=1).columns.tolist()
                pool.append([sorted_aa[rank] for rank in ranks])
        protein = []
        for candidates in pool:
            protein.append(random.choice(candidates))

    else:  # Generate protein within stability bounds.
        assert isinstance(stability_start, (tuple, list)) and len(stability_start) == 2
        protein = get_bounded_stability_protein(stability_start, current_protein,
                                                sites, stability_table)

    return protein


def get_bounded_stability_protein(stability_bounds, current_protein, sites, stability_table,
                                   n_variant_sites=5, maximum_attempts=100):
    """Take starting protein sequence, mutate 5 residues until the protein is
    closer to the stability bounds, then choose 5 new residues and continue.
    If it cannot make a more stable protein with the 5 residues it's mutating
    it reverts back to the previous state and picks 5 new residues."""

    current_stability = calculate_stability(current_protein, stability_table)

    lower_bound, upper_bound = stability_bounds
    if lower_bound is None:
        lower_bound = -np.inf
    if upper_bound is None:
        upper_bound = np.inf
    # TODO: Also pass omega to check against?
    assert lower_bound < upper_bound
    # TODO: This is slightly different to the original algorithm (below)
    # stability = current_stability
    # counter = 0
    # while not lower_bound < stability < upper_bound and counter <= maximum_attempts:
    #     # Mutate the new protein (sample without replacement)
    #     chosen_sites = random.sample(sites.variant, n_variant_sites)
    #     (protein, stability) = twist_protein(current_protein, chosen_sites,
    #                                        stability_table)
    #     counter += 1

    while not lower_bound < current_stability < upper_bound:
        # Mutate the new protein (sample without replacement)
        chosen_sites = random.sample(sites.variant, n_variant_sites)
        (protein, stability) = twist_protein(current_protein, chosen_sites, stability_table)
        counter = 0
        # while not lower_bound < stability < upper_bound and counter <= maximum_attempts:
        #     # Revert to initial protein
        #     # ...
        #     # Continue to mutate until better than current_protein
        #     (protein, stability) = twist_protein(current_protein, chosen_sites, stability_table)
        #     counter += 1

        if current_stability < lower_bound:  # setting lower bounds of medium stability
            while stability < current_stability and counter <= maximum_attempts:
                # Continue to mutate until better than current_protein
                (protein, stability) = twist_protein(current_protein, chosen_sites, stability_table)
                counter += 1

        elif current_stability > upper_bound:  # set upper bounds of medium stability
            while stability > current_stability and counter <= maximum_attempts:
                # Continue to mutate until better than current_protein
                (protein, stability) = twist_protein(current_protein, chosen_sites, stability_table)
                counter += 1

        current_protein, current_stability = protein, calculate_stability(protein, stability_table)

    return current_protein


def record_generation_statistics(generation, population, sites,
                                 stability_table, out_paths):
                                # omega, p_mutation, record, out_paths):
    """Record the stability of every protein in the generation and store them in
    dictionary. Optionally generate data and figures about stability.
    """

    # # Build distribution of stability values existing in evolving protein
    # stabilities = get_phi_stability_table(population, stability_table)

    # # TODO: Move plot out
    # if record["figures"]:
    #     clims = (np.floor(np.amin(stability_table.values)),
    #             np.ceil(np.amax(stability_table.values)))
    #     plot_phi_stability_table(generation, stabilities, clims, out_paths)

    # if record["residues"]:
    #     # save_dir = os.path.join(out_paths["figures"], "stabilitydotmatrix")
    #     plot_threshold_stability(generation, population, stabilities,
    #                            stability_table, omega, out_paths)
    #     plot_stability_space(generation, population, stabilities, stability_table,
    #                        omega, out_paths)

    # if record["statistics"]:
    # Record 5 statistical tests on the protein stability space
    stats_file_name = f"normal_distribution_tests_G{generation}.md"
    if generation == 0:
        # stats_file_name = "normal_distribution_statistics_stability_space.md"
        distributions = stability_table.values
    else:
        # stats_file_name = f"normal_distribution_statistics_generation{generation}.md"
        # Build distribution of stability values excluding invariant sites
        distributions = get_phi_stability_table(population, stability_table,
                                                exclude_invariants=True,
                                                variant_sites=sites.variant)
                                                # record["invariants"], sites.variant)
    stats_full_name = os.path.join(out_paths["statistics"], stats_file_name)
    write_histogram_statistics(stats_full_name, distributions)
    if generation > 0:
        append_ks_statistics(stats_full_name, distributions.ravel(),
                             stability_table.values.ravel())

    # if record["histograms"] and record["figures"]:
    #     # TODO: Move out
    #     # disthistfilename = "generation_{}.png".format(generation)
    #     # disthistfullname = os.path.join(out_paths["figures"], "histograms", disthistfilename)
    #     # plot_histogram_of_stability(generation, stabilities.ravel(),
    #     #                           stability_table.values.ravel(), omega, out_paths)
    #     plot_stability_histograms(generation, stabilities, stability_table,
    #                               omega, out_paths)


def get_phi_stability_table(population, stability_table, exclude_invariants=False,
                            variant_sites=None):
    """Build a stability table for given generation's population.

    The array has one row for each protein in the population and the stability
    value (\Delta \Delta G_e) for each amino acid in its position.
    """
    dist_clone_stability = []
    # Find and plot all stability values in the current generation
    for pi, protein in population.items():

        if exclude_invariants:  # record["invariants"]:
            stabilities = [stability_table.loc[loc, amino_acid]
                           for loc, amino_acid in enumerate(protein)
                           if loc in variant_sites]
        else:
            stabilities = get_amino_acid_stabilities(protein, stability_table)

        dist_clone_stability.append(stabilities)  # Becomes a new row
    return np.asarray(dist_clone_stability)


def calculate_population_stabilities(population, stability_table):
    """Calculate the total stability of every protein in a population."""
    return np.sum(get_phi_stability_table(population, stability_table), axis=1)


def create_tree(n_proteins, n_roots):
    """Create a dictionary of lists of indicies for roots and branches."""
    protein_keys = list(range(n_proteins))
    tree = {}  # Dictionary of keys for roots and branches in the population
    # Randomly sample without replacement n_roots items from clonelist
    root_keys = random.sample(protein_keys, n_roots)
    for r in root_keys:
        protein_keys.remove(r)
    # root_keys = [protein_keys.pop(random.randrange(len(protein_keys)))
    #              for r in range(n_roots)]
    tree["roots"] = root_keys
    tree["branches"] = [protein_keys]  # lists of non-roots
    return tree


def calc_bifurcation_interval(n_generations, n_clones, n_roots):
    """Calculate the bifurcation interval (in generations) given that there
    must be at least three proteins left in each branch after bifurcation."""
    # Original iterative algorithm
    # pool = n_clones - n_roots
    # n_phases = 1  # n_bifurations + 1
    # while pool > 5:  # stop when there are 3, 4 or 5 leaves per branch
    #     pool //= 2  # Floor division
    #     n_phases += 1
    # return int(n_generations/n_phases)  # number of generations per bifurcation

    # Reasoning
    # I_B := floor(G / n_phases)
    # n_phases := n_B + 1
    # 3 <= floor(pool / 2**n_B) < 6
    # floor(pool / 2**(n_B-1)) = 6
    # floor(pool / 2**n_B) = 3
    # pool / 3 = 2**n_B
    # log2(pool / 3) = n_B
    # n_B = floor(log2(pool / 3))
    # pool := n_clones - n_roots
    # n_phases = floor(log2(n_clones - n_roots / 3)) + 1
    # I_B = floor(G / floor(log2((n_clones - n_roots) / 3)) + 1)
    return int(n_generations / (1 + int(np.log2(n_clones-n_roots)-np.log2(3))))


def bifurcate_branches(branches):
    new_bifurcations = []  # temporary store for new bifurcations
    for branch in branches:  # bifuricate each set of leaves
        random.shuffle(branch)
        midpoint = int(len(branch)/2)
        new_bifurcations.append(branch[:midpoint])
        new_bifurcations.append(branch[midpoint:])
    return new_bifurcations[:]


def select_from_pool(protein_index, candidates, stabilities, omega):
    """Filter out the original protein and those above the instability
    threshold."""
    pool = [c for c in candidates
            if c != protein_index and stabilities[c] <= omega]
    if len(pool) > 0:
        new_protein_index = random.choice(pool)
    else:
        new_protein_index = None
    return new_protein_index


def replace_protein(protein_index, tree, stabilities, omega):

    if protein_index in tree["roots"]:
        new_index = select_from_pool(protein_index, tree["roots"],
                                     stabilities, omega)
    else:  # Protein is in one of the branches
        for branch in tree["branches"]:
            if protein_index in branch:
                new_index = select_from_pool(protein_index, branch,
                                             stabilities, omega)
    return new_index


# TODO: Refactor for efficiency?
# def mutate_population(current_generation, n_mutations_per_gen, tree,
#                       variant_sites, p_mutation, LG_matrix,
#                       stability_table, omega):
#     """Mutate a set of sequences based on the LG+I+G model of amino acid
#     substitution.
#     """
#     # NOTE: This could be removed for speed after checking it is not used later
#     next_generation = copy.deepcopy(current_generation)
#     for q in range(n_mutations_per_gen):  # impliment gamma
#
#         stabilities = calculate_generation_stability(next_generation, stability_table)
#         counter = 0
#         successful_mutation = False
#         while not successful_mutation:  # Mutate until all branches have stable proteins
#             successful_mutation = True
#             # Pick random key, clone to make a random generation
#             pi, protein = random.choice(list(next_generation.items()))
#             # Mutate the copy with the randomly chosen residue
#             mutant = mutate_protein(protein, p_mutation, LG_matrix)
#
#             # next_generation[pi] = mutant  # update with new sequence
#             stability = calculate_stability(mutant, stability_table)
#
#             if stability < omega:  # if stability is less than threshold clone a random sequence in its place.
#                 mutant_index = replace_protein(pi, tree, stabilities,
#                                                omega)
#                 # If no suitable clones are available, re-mutate the generation and start again
#                 if mutant_index is None:
#                     successful_mutation = False
#                     stabilities[pi] = stability  # Save last unsuccessful stability
#                     # break  # out of loop over stabilities
#                 else:
#                     next_generation[pi] = next_generation[mutant_index]  # swap out unstable clone for stable clone
#             counter += 1
#             if counter == 100:
#                 raise Exception("Unable to mutate population: "
#                                 "maximum tries exceeded!\n"
#                                 "The mutation rate is too high, mu is too low "
#                                 "or sigma is too small.")
#
#     return next_generation, stabilities


def mutate_population(current_generation, n_mutations_per_gen, tree,
                      p_mutation, LG_matrix, stability_table, omega):
    """Mutate a set of sequences based on the LG+I+G model of amino acid
    substitution.
    """
    counter = 0
    successful_mutation = False

    while not successful_mutation:  # Mutate until all branches are stable
        successful_mutation = True
        next_generation = copy.deepcopy(current_generation)
        for q in range(n_mutations_per_gen):  # impliment gamma
            # Pick random key, clone to make a random generation
            pi, protein = random.choice(list(next_generation.items()))
            # Mutate the copy with the randomly chosen residue
            mutant = mutate_protein(protein, p_mutation, LG_matrix)
            next_generation[pi] = mutant  # update with new sequence

        stabilities = calculate_population_stabilities(next_generation,
                                                       stability_table)

        for pi, stability in enumerate(stabilities):
            if stability > omega:  # clone a random sequence if unstable
                mutant_index = replace_protein(pi, tree, stabilities, omega)
                # If no suitable clones are available, re-mutate and try again
                if mutant_index is None:
                    successful_mutation = False
                    break  # out of loop over stabilities
                else:  # swap out unstable clone for stable clone
                    next_generation[pi] = next_generation[mutant_index]

        counter += 1
        if counter == 100:
            raise Exception("Unable to mutate population: "
                            "maximum tries exceeded!\n"
                            "The mutation rate is too high, mu is too low "
                            "or sigma is too small.")

    return next_generation


def kill_proteins(population, tree, death_rate, stability_table, omega):
    n_clones = len(population)
    # mortals = random.sample(range(n_clones), int(n_clones*death_rate))
    clones = np.arange(n_clones)
    condemned = clones[np.where(np.random.rand(n_clones) < death_rate)]
    # Recalculate stabilities after all mutations
    stabilities = calculate_population_stabilities(population, stability_table)
    for pi in condemned:
        new_index = replace_protein(pi, tree, stabilities, omega)
        if new_index is None:  # Should never happen
            warnings.warn(f"Unable to kill protein {pi}!")
            raise Exception("No suitable candidates on branch!")
        else:
            population[pi] = population[new_index]  # Replace dead protein
    return population


def evolve(n_generations, population, stability_table, omega, sites,
           p_mutation, mutation_rate, death_rate, tree, LG_matrix,
           plot_omega, plot_epsilon, record, out_paths):
    """Generation generator - mutate a protein for a defined number of
    generations according to an LG matrix and gamma distribution.
    """

    n_clones = len(population)
    clone_size = len(population[0])
    n_mutations_per_gen = int(n_clones * clone_size * mutation_rate)
    n_roots = len(tree["roots"])

    # Calculate number of bifurications per generation.
    n_gens_per_bifurcation = calc_bifurcation_interval(n_generations, n_clones,
                                                       n_roots)

    # population = copy.deepcopy(initial_population)  # current generation
    phi_stabilities = get_phi_stability_table(population, stability_table)
    # Record initial population
    if record["statistics"]:
        record_generation_statistics(0, population, sites, stability_table, out_paths)
    if record["figures"]:
        # Build distribution of stability values existing in evolving protein
        stabilities = get_phi_stability_table(population, stability_table)
        plot_generation_stability(0, stabilities, stability_table, out_paths)
        if record["histograms"]:
            plot_stability_histograms(0, stabilities, stability_table, out_paths)
    write_fasta_alignment(0, population, out_paths)

    # Store each generation along with its stability
    # Generation = namedtuple('Generation', ['population', 'stabilities'])
    # Create a list of generations and add initial population and stability
    # history = [Generation(population=population, stabilities=phi_stabilities)]
    history = {0: Generation(population=population, stabilities=phi_stabilities)}
    if record["data"]:
        save_history(0, history, out_paths)

    if record["figures"]:
        # TODO: Refactor plot_omega, plot_epsilon
        plot_simulation(0, history, stability_table, omega, plot_omega,
                        plot_epsilon, n_generations, out_paths)
        plot_all_stabilities(0, history, stability_table, omega,
                             plot_omega, plot_epsilon, n_generations, out_paths)
        plot_traces(0, history, stability_table, omega,
                    plot_omega, plot_epsilon, n_generations, out_paths)
        plt.close('all')  # TODO: Move into plotting.py

    for gen in trange(1, n_generations+1):  # run evolution for 1:n_generations

        # Bifuricate in even generation numbers so every branch on tree has
        # 3 leaves that have been evolving by the last generation
        if gen % n_gens_per_bifurcation == 0 and len(tree["branches"][0]) > 3:
            tree["branches"] = bifurcate_branches(tree["branches"])
            # Write out bifurcations
            # tree_log_file = os.path.join(run_path, "tree", "tree.txt")
            write_tree(gen, tree, out_paths)

        # Mutate population
        next_generation = mutate_population(population, n_mutations_per_gen,
                                            tree, p_mutation, LG_matrix,
                                            stability_table, omega)

        # Allow sequences to die and be replacecd at a predefined rate
        if death_rate > 0:
            next_generation = kill_proteins(next_generation, tree, death_rate,
                                            stability_table, omega)

        phi_stabilities = get_phi_stability_table(population, stability_table)
        # The population becomes next_generation only if bifurcations (and deaths) were successful
        population = next_generation
        # Record intermediate stabilities to show existence of unstable proteins
        # history.append(Generation(population=population, stabilities=phi_stabilities))
        history[gen] = Generation(population=population, stabilities=phi_stabilities)

        # Write fasta every record["fasta_rate"] generations
        if gen % record["fasta_rate"] == 0:
            write_fasta_alignment(gen, population, out_paths)
        # Record population details at the end of processing
        if gen % record["rate"] == 0:
            if record["statistics"]:
                record_generation_statistics(gen, population, sites, stability_table, out_paths)
            if record["figures"]:
                # Build distribution of stability values existing in evolving protein
                stabilities = get_phi_stability_table(population, stability_table)
                plot_generation_stability(gen, stabilities, stability_table, out_paths)
                if record["histograms"]:
                    plot_stability_histograms(gen, stabilities, stability_table, out_paths)
                plot_simulation(gen, history, stability_table, omega,
                                plot_omega, plot_epsilon, n_generations, out_paths)
                plot_all_stabilities(gen, history, stability_table, omega,
                                     plot_omega, plot_epsilon, n_generations, out_paths)
                plot_traces(gen, history, stability_table, omega,
                            plot_omega, plot_epsilon, n_generations, out_paths)
            if record["data"]:
                save_history(gen, history, out_paths)
            plt.close('all')  # TODO: Move into plotting.py

    write_final_fasta(population, tree, out_paths)
    return history


def pesst(n_generations=2000, stability_start='high', omega=0,
          mu=0, sigma=2.5, skew=0, distributions='Tokuriki',
          n_clones=52, n_roots=4, clone_size=100, p_invariant=0.1,
          mutation_rate=0.001, death_rate=0.02,
          gamma_kwargs=None, record_kwargs=None, output_dir=None, seed=None):
    """The main function to intialise and run PESST evolutionary simulations."""

    print('PESST started...')
    # Validate arguments
    assert 1 < clone_size
    assert 2 < n_roots < n_clones
    assert 0.0 <= mutation_rate <= 1.0
    assert 0.0 <= death_rate <= 1.0
    assert 0.0 <= p_invariant < 1.0  # Must be less than 1 for evolution
    n_invariants = int(p_invariant * clone_size)
    assert 0 <= n_invariants < clone_size

    # NOTE: With (\Delta) \Delta G_e, positive is bad (destabilising)
    # and negative is good (stabilising)

    if distributions is None:
        # distributions = [(mu, sigma, skew, 1)]
        distributions = [{"mu": mu, "sigma": sigma, "skew": skew, "proportion": 1}]
    elif isinstance(distributions, str) and distributions.capitalize() == 'Tokuriki':
        # Calculate the fraction of surface (P1) and core (1-P1) residues 
        # according to: Tokuriki et al. 2007. doi:10.1016/j.jmb.2007.03.069
        P1 = 1.13 - (0.3 * np.log10(clone_size))
        P1 = np.clip(P1, 0, 1)  # Ensure 0 <= P1 <= 1
        n_surface = int(np.floor(P1 * clone_size))
        n_core = clone_size - n_surface
        # Create a list of distributions
        # Distribution = namedtuple('Distribution', ['mu', 'sigma', 'skew', 'proportion'])
        # distributions = [Distribution(mu=0.54, sigma=0.98, skew=0, proportion=P1),
        #                  Distribution(mu=2.05, sigma=1.91, skew=0, proportion=1-P1)]
        distributions = [{"mu": 0.54, "sigma": 0.98, "skew": 0, "proportion": P1},
                         {"mu": 2.05, "sigma": 1.91, "skew": 0, "proportion": 1-P1}]
    else:
        assert isinstance(distributions, (list, tuple))

    # TODO: Add rerun flag to load settings (and seed)
    # settings = json.load(sf)

    # Instability threshold
    if omega is None:
        omega = np.inf

    # TODO: switch from random to np.random for proper seeding
    if seed is None:
        # Get a random integer to seed both
        seed = random.randrange(2**32)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
    np.random.seed(seed)
    random.seed(seed)

    # Default keyword arguments
    # TODO: Refactor kwargs like seaborn? e.g.
    # n_boot = kwargs.get("n_boot", 10000)
    # https://github.com/mwaskom/seaborn/blob/3a3ec75befab52c02650c62772a90f8c23046038/seaborn/algorithms.py

    # gamma = {"shape": 1.9,
    #          "scale": 1/1.9,  # theta = 1/beta NOTE: 1/gamma_shape
    #          "iterations": 50,
    #          "samples": 10000}
    # gamma = {}
    # if gamma_kwargs is not None:
    #     gamma.update(gamma_kwargs)
    # Populate gamma with default values for each key if not already given
    gamma_kwargs.setdefault('shape', 1.9)
    gamma_kwargs.setdefault('scale', 1/1.9)  # theta = 1/beta NOTE: 1/gamma_shape
    gamma_kwargs.setdefault('iterations', 50)
    gamma_kwargs.setdefault('samples', 10000)
    gamma = gamma_kwargs  # TODO: Remove after renaming args and below

    # TODO: Put run_path (and subdirs) in record dict
    # record = {"rate": 50,
    #           "fasta_rate": 50,
    #           "residues": False,
    #           "statistics": True,
    #           "histograms": True,
    #           "data": True,
    #           "gif": True}
    # if record_kwargs is not None:
    #     record.update(record_kwargs)
    record_kwargs.setdefault('rate', 50)
    record_kwargs.setdefault('fasta_rate', 50)
    # record_kwargs.setdefault('residues', False)
    record_kwargs.setdefault('statistics', True)
    record_kwargs.setdefault('histograms', True)
    record_kwargs.setdefault('data', True)
    record_kwargs.setdefault('figures', True)
    record_kwargs.setdefault('gif', True)
    record_kwargs.setdefault('gif_rate', 0.25)
    record = record_kwargs  # TODO: Remove after renaming args and below

    # Create output folder and subfolders
    # PWD = os.path.dirname(__file__)
    out_paths = create_output_folders(output_dir)
    # run_path = out_paths["results"]
    settings_kwargs = {"n_generations": n_generations,
                       "stability_start": stability_start,
                       "omega": omega,
                       "distributions": distributions,
                       "n_clones": n_clones,
                       "clone_size": clone_size,
                       "mutation_rate": mutation_rate,
                       "p_invariant": p_invariant,
                       "death_rate": death_rate,
                       "n_roots": n_roots,
                       "seed": seed,
                       "gamma": gamma,
                       "record": record}
    save_settings(settings_kwargs, out_paths)  # record run settings
    
    LG_matrix = load_LG_matrix()  # Load LG matrix
    if record["figures"]:
        plot_LG_matrix(LG_matrix, out_paths)

    print('Creating amino acid stability distribution...')
    # Make stability table of \Delta \Delta G_e values
    stability_table = get_stability_table(clone_size, LG_matrix.columns, distributions)
    write_stability_table(stability_table, out_paths)

    if isinstance(stability_start, str) and stability_start.lower() == "low":
        print("NOTE: With 'low' starting stability Omega is ignored.")
        # warnings.warn("With 'low' starting stability selected Omega is ignored.")
                      # "If the run fails, please check your stability threshold,"
                      # "omega, is low enough: {}".format(omega))
        plot_omega, plot_epsilon = False, True
        omega = np.inf
    else:
        epsilon = clone_size * np.mean(stability_table.values)  # mu
        if omega > epsilon:
            plot_omega, plot_epsilon = True, True
        else:
            plot_omega, plot_epsilon = True, False
    if record["figures"]:
        plot_stability_table(stability_table, out_paths)

    # NOTE: These could be calculated after fixing sites
    # min_stability = np.sum(np.amin(stability_table, axis=1))
    # mean_stability = np.sum(np.mean(stability_table, axis=1))  # Approximate
    # max_stability = np.sum(np.amax(stability_table, axis=1))
    # print(f"Protein stability bounds: ({min_stability:.3f}, {max_stability:.3f}), "
    #       f"[mean = {mean_stability:.3f}]")

    # Generate variant/invariant sites
    # TODO: return boolean array where True is variant
    sites = get_allowed_sites(clone_size, n_invariants)
    # Generate mutation probabilities for every site
    p_mutation = gamma_ray(clone_size, sites, gamma, record, out_paths)  # TODO: Move plotting out

    # Generate a protein of specified stability taking into account the invariant
    # sites created (calling variables in this order stops the evolutionary
    # process being biased by superstable invariant sites.)
    # phi
    initial_protein = get_stable_protein(stability_start, clone_size, sites,
                                         stability_table, omega)
    write_initial_protein(initial_protein, out_paths)  # Record initial protein
    initial_stability = calculate_stability(initial_protein, stability_table)

    # Calculate bounds on the protein stability space and an approximate mean
    max_protein = copy.deepcopy(initial_protein)
    max_residues = stability_table.idxmax(axis=1)
    min_protein = copy.deepcopy(initial_protein)
    min_residues = stability_table.idxmin(axis=1)
    mean_stabilities = get_amino_acid_stabilities(initial_protein, stability_table)
    # Account for invariant sites
    for vloc in sites.variant:
        max_protein[vloc] = max_residues[vloc]
        min_protein[vloc] = min_residues[vloc]
        mean_stabilities[vloc] = np.mean(stability_table.loc[vloc])
    min_stability = calculate_stability(min_protein, stability_table)
    mean_stability = sum(mean_stabilities)
    max_stability = calculate_stability(max_protein, stability_table)

    # min_stabilities = np.amin(stability_table, axis=1)
    # for iloc in sites.invariant:
    #     min_stabilities[iloc] = stability_table.loc[iloc, initial_protein[iloc]]
    # min_stability = np.sum(min_stabilities)
    # initial_stabiliies = get_amino_acid_stabilities(initial_protein, stability_table)
    # is_variant = np.zeros_like(initial_protein, dtype=bool)
    # is_variant[sites.variant] = True
    # is_invariant = ~is_variant
    # min_stability = np.sum(np.amin(stability_table, axis=1)[is_variant], 
    #                        initial_stabiliies[is_invariant])
    # min_stability = np.sum(np.where(is_variant, np.amin(stability_table, axis=1), initial_stabiliies))
    # max_stability = np.sum(np.where(is_variant, np.amax(stability_table, axis=1), initial_stabiliies))
    # mean_stability = np.sum(np.where(is_variant, np.mean(stability_table, axis=1), initial_stabiliies))

    print(f"Protein stability bounds: ({min_stability:.3f}, {max_stability:.3f}), "
          f"[mean = {mean_stability:.3f}]")
    print(f"Initial protein stability = {initial_stability:.3f} [Omega = {omega}]")
    print_protein(initial_protein)

    # assert omega < max_stability  # Ensure that the possibile (instabilities) span Omega
    assert min_stability < omega
    if initial_stability > omega:
        raise Exception(f"The instability threshold (Omega = {omega}) is too low "
                        f"for the distribution (mean = {mean_stability:.3f})!")

    initial_population = clone_protein(initial_protein, n_clones)  # copy
    # Population = namedtuple('Population', ['proteins', 'sites', 'stabilities'])

    # Generate list of clone keys for bifurication
    tree = create_tree(n_clones, n_roots)
    write_roots(tree["roots"], out_paths)
    write_tree(0, tree, out_paths)

    print(f'Evolving {n_clones} proteins for {n_generations} generations...')
    history = evolve(n_generations, initial_population, stability_table, omega,
                     sites, p_mutation, mutation_rate, death_rate, tree,
                     LG_matrix, plot_omega, plot_epsilon, record, out_paths)

    if record["figures"]:
        # legend_title = "; ".join([r"$\mu$ = {}".format(mu),
        #                           r"$\sigma$ = {}".format(sigma),
        #                           "skew = {}".format(skew),
        #                           r"$\delta$ = {}".format(mutation_rate)])
        plot_evolution(history, stability_table, omega, plot_omega, plot_epsilon,
                    out_paths)  # , legend_title=legend_title)

        # Create animations
        if record["gif"]:
            recorded_generations = list(range(0, n_generations+1, record["rate"]))
            figures = ["pesst", "phi_stability_table", "stabilities", "traces"]
            # if record["residues"]:
            #     figures.extend(["OLD_stable_dist_gen", "OLD_generation"])
            if record["histograms"]:
                figures.append("histogram")
            for fig_base in figures:
                path_root = os.path.join(out_paths["figures"], fig_base)
                filenames = [f"{path_root}_G{gen}.png"
                            for gen in recorded_generations]
                create_gif(filenames, out_paths, duration=record["gif_rate"])

            # if record["statistics"]:
            #     path_root = os.path.join(out_paths["figures"], "generation_")
            #     filenames = [path_root+"{}.png".format(gen)
            #                  for gen in record_generations]
            #     create_gif(filenames, duration=0.25)
    return history, out_paths
