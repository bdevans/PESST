import copy
import os
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
from tqdm import trange

from .dataio import (create_output_folders, save_settings, write_tree,
                     write_roots, write_initial_protein, write_stability_table,
                     write_fasta_alignment, write_final_fasta, load_LG_matrix,
                     write_histogram_statistics, append_ks_statistics,
                     create_gif, save_history)
from .plotting import (plot_stability, plot_evolution, plot_gamma_distribution,
                       plot_threshold_fitness, plot_stability_histograms,
                       plot_fitness_space, plot_stability_table, plot_LG_matrix,
                       plot_phi_fitness_table)


def get_stability_table(clone_size, mu, sigma, skew, amino_acids):
    """Generate a dictionary describing list of fitness values at each position
    of the generated protein.
    """
    values = sp.stats.skewnorm.rvs(skew, loc=mu, scale=sigma,
                                   size=(clone_size, len(amino_acids)))
    fitness_table = pd.DataFrame(values, columns=amino_acids)
    return fitness_table


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


def gamma_ray(clone_size, sites, gamma, out_paths):
    """Generate a set of gamma rate categories.

    Does so by sampling many times from a gamma distribution.
    Tests of the trade-off between computing time and variance led me to set
    this to 10000 samples from the distribution.
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
        # TODO: Remove warnings filter when scipy is updated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            # Find the median of each quartile
            medians[i, :], _, _ = sp.stats.binned_statistic(samples, samples,
                                                            statistic='median',
                                                            bins=quartiles)
    # Calculate average of medians across iterations
    average_medians = np.mean(medians, axis=0)

    # Replot the gamma distributuion as a check
    plot_gamma_distribution(gamma, samples, quartiles, average_medians, out_paths)

    gamma_categories = np.random.choice(average_medians, size=clone_size)
    gamma_categories[sites.invariant] = 0
    return gamma_categories/sum(gamma_categories)  # p_mutation


def mutate_protein(protein, p_mutation, LG_matrix):
    """Mutate a residue to another residue based on the LG matrix."""
    mutant = copy.deepcopy(protein)  # Necessary!
    location = np.random.choice(len(mutant), p=p_mutation)
    amino_acid = mutant[location]
    p_transition = LG_matrix.loc[amino_acid]
    LG_residues = LG_matrix.columns.values  # .tolist()
    mutant[location] = np.random.choice(LG_residues, p=p_transition)
    return mutant


def get_amino_acid_stabilities(protein, fitness_table):
    """Collate the individual stability contributions of each amino acid in a
    protein."""
    return [fitness_table.loc[loc, aa] for loc, aa in enumerate(protein)]


def calculate_stability(protein, fitness_table):
    """Calculate stability of a protein given the sequence and fitness values."""
    return sum(get_amino_acid_stabilities(protein, fitness_table))


def get_random_protein(clone_size, fitness_table, start_amino_acid="M"):
    """Generate an original starting protein clone_size long with a start
    amino acid set to methionine.
    """
    protein = random.choices(fitness_table.columns.values.tolist(),
                             k=clone_size-1)
    # protein = [random.choice(RESIDUES) for _ in range(clone_size)] # < 3.6
    protein.insert(0, start_amino_acid)  # Start with methionine
    # TODO: Convert to strings preventing in-place modification
    # return ''.join(protein)
    return protein


def twist_protein(protein, mutation_sites, fitness_table):
    """Randomly change one variant site and calculate new fitness."""
    mutant = protein[:]
    amino_acids = fitness_table.columns.values.tolist()
    for ai in mutation_sites:
        mutant[ai] = random.choice(amino_acids)
    fitness = calculate_stability(mutant, fitness_table)
    return (mutant, fitness)


def get_fit_protein(stability_start, clone_size, sites, stability_table):
    """Generate a protein of a specified stability.

    Make either a 'high', 'low' or medium stability protein where the stability
    is specified by a tuple of (lower_bound, upper_bound).

    This function currently locks in invariant sites before finding the most
    stable sites, meaning the invariant sites are simply sampled from the
    normal distribution, and not from the superfit distribution.

    Generating the protein in this manner avoids bias towards increased
    stability that could be generated by the invariant sites.
    """
    initial_protein = get_random_protein(clone_size, stability_table)

    if isinstance(stability_start, str):  # not medium
        if stability_start.lower() == 'low':  # generate super unstable protein
            # TODO: Adjust threshold or allow building proteins above threshold
            sequence = [0, 1, 2]  # Three lowest

        elif stability_start.lower() == 'high':  # generate super stable protein
            sequence = [-1, -2, -3]  # Three highest

        pool = [["M"] * 3]
        for ai in range(1, len(stability_table.index)):  # range(clone_size):
            if ai in sites.invariant:
                pool.append([initial_protein[ai]] * 3)
            else:
                amino_acids = stability_table.columns.values
                sorted_aa = amino_acids[stability_table.loc[ai].argsort()]
                # sorted_aa = stability_table.sort_values(stability_table.loc[ai], axis=1).columns.tolist()
                pool.append([sorted_aa[rank] for rank in sequence])
        protein = []
        for candidates in pool:
            protein.append(random.choice(candidates))

    # Generate medium stability protein.
    # Take starting protein sequence, mutate 5 residues until the protein is
    # closer to the stability bounds, then choose 5 new residues and continue.
    # If it cannot make a more stable protein with the 5 residues it's mutating
    # it reverts back to the previous state and picks 5 new residues.
    else:  # elif stability_start == 'medium':
        n_variant_sites = 5
        initial_stability = calculate_stability(initial_protein, stability_table)

        lower_bound, upper_bound = stability_start
        if lower_bound is None:
            lower_bound = -np.inf
        if upper_bound is None:
            upper_bound = np.inf
        # TODO: Also pass omega to check against?
        assert lower_bound < upper_bound
        # TODO: This is slightly different to the original algorithm (below)
        stability = initial_stability
        protein = initial_protein
        sites_count = 0
        # lowest_found, highest_found = -np.inf, np.inf
        # range = upper_bound - lower_bound
        # error = abs(initial_stability - lower_bound + range/2)  # Calculate distance

        while not lower_bound < stability < upper_bound and sites_count <= 100:
            # Mutate the new protein (sample without replacement)
            chosen_sites = random.sample(sites.variant, n_variant_sites)
            print("Trying set of sites: {}".format(sites_count))
            twist_count = 0
            prev_protein = protein
            while not lower_bound < stability < upper_bound and twist_count < 100:
                (protein, stability) = twist_protein(initial_protein,
                                                     chosen_sites,
                                                     stability_table)
                twist_count += 1
            sites_count += 1

        while not lower_bound < initial_stability < upper_bound:
            # Mutate the new protein (sample without replacement)
            chosen_sites = random.sample(sites.variant, n_variant_sites)
            (protein, stability) = twist_protein(initial_protein, chosen_sites, stability_table)
            counter = 0
            # while not lower_bound < stability < upper_bound and counter <= 100:
            #     # Revert to initial protein
            #     # ...
            #     # Continue to mutate until better than initial_protein
            #     (protein, stability) = twist_protein(initial_protein, chosen_sites, stability_table)
            #     counter += 1
            previous_stability = initial_stability

            if initial_stability < lower_bound:  # setting lower bounds of medium stability
                while stability < previous_stability and counter <= 100:
                    # Continue to mutate until better than initial_protein
                    previous_stability = stability
                    (protein, stability) = twist_protein(initial_protein, chosen_sites, stability_table)
                    counter += 1

            elif initial_stability > upper_bound:  # set upper bounds of medium stability
                while stability > previous_stability and counter <= 100:
                    previous_stability = stability
                    # Continue to mutate until better than initial_protein
                    (protein, stability) = twist_protein(initial_protein, chosen_sites, stability_table)
                    counter += 1

            initial_protein = protein
            initial_stability = calculate_stability(protein, stability_table)
        protein = initial_protein

    return protein


def record_generation_stability(generation, population, sites, fitness_table,
                                omega, p_mutation, record, out_paths):
    """Record the fitness of every protein in the generation and store them in
    dictionary. Optionally generate data and figures about fitness.
    """

    # Build distribution of fitness values existing in evolving protein
    stabilities = get_phi_stability_table(population, fitness_table)

    clims = (np.floor(np.amin(fitness_table.values)),
             np.ceil(np.amax(fitness_table.values)))
    plot_phi_fitness_table(generation, stabilities, clims, out_paths)

    if record["residues"]:
        # save_dir = os.path.join(out_paths["figures"], "fitnessdotmatrix")
        plot_threshold_fitness(generation, population, stabilities,
                               fitness_table, omega, out_paths)
        plot_fitness_space(generation, population, stabilities, fitness_table,
                           omega, out_paths)

    if record["statistics"]:
        # Record 5 statistical tests on the protein fitness space
        if generation == 0:
            stats_file_name = "normal_distribution_statistics_fitness_space.md"
            distributions = fitness_table.values
        else:
            stats_file_name = "normal_distribution_statistics_generation{}.md".format(generation)
            # Build distribution of stability values excluding invariant sites
            distributions = get_phi_stability_table(population, fitness_table,
                                                    exclude_invariants=True,
                                                    variant_sites=sites.variant)
                                                    # record["invariants"], sites.variant)
        stats_full_name = os.path.join(out_paths["statistics"], stats_file_name)
        write_histogram_statistics(stats_full_name, distributions)
        if generation > 0:
            append_ks_statistics(stats_full_name, distributions.ravel(),
                                 fitness_table.values.ravel())

    if record["histograms"]:
        # disthistfilename = "generation_{}.png".format(generation)
        # disthistfullname = os.path.join(out_paths["figures"], "histograms", disthistfilename)
        # plot_histogram_of_fitness(generation, stabilities.ravel(),
        #                           fitness_table.values.ravel(), omega, out_paths)
        plot_stability_histograms(generation, stabilities, fitness_table,
                                  omega, out_paths)


def get_phi_stability_table(population, fitness_table, exclude_invariants=False,
                            variant_sites=None):
    """Build a fitness table for given generation's population.

    The array has one row for each protein in the population and the fitness
    value for each amino acid in its position.
    """
    dist_clone_fitness = []
    # Find and plot all fitness values in the current generation
    for pi, protein in list(population.items()):

        if exclude_invariants:  # record["invariants"]:
            stabilities = [fitness_table.loc[loc, amino_acid]
                           for loc, amino_acid in enumerate(protein)
                           if loc in variant_sites]
        else:
            stabilities = get_amino_acid_stabilities(protein, fitness_table)

        dist_clone_fitness.append(stabilities)  # Becomes a new row
    return np.asarray(dist_clone_fitness)


def calculate_population_fitness(population, fitness_table):
    """Calculate the total stability of every protein in a population."""
    return np.sum(get_phi_stability_table(population, fitness_table), axis=1)


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


def select_from_pool(protein_index, candidates, fitnesses, omega):
    """Filter out original protein and those below the fitness threshold."""
    pool = [c for c in candidates
            if c != protein_index and fitnesses[c] >= omega]
    if len(pool) > 0:
        new_protein_index = random.choice(pool)
    else:
        new_protein_index = None
    return new_protein_index


def replace_protein(protein_index, tree, fitnesses, omega):

    if protein_index in tree["roots"]:
        new_index = select_from_pool(protein_index, tree["roots"], fitnesses,
                                     omega)
    else:  # Protein is in one of the branches
        for branch in tree["branches"]:
            if protein_index in branch:
                new_index = select_from_pool(protein_index, branch, fitnesses,
                                             omega)
    return new_index


# TODO: Refactor for efficiency?
# def mutate_population(current_generation, n_mutations_per_gen, tree,
#                       variant_sites, p_mutation, LG_matrix,
#                       fitness_table, omega):
#     """Mutate a set of sequences based on the LG+I+G model of amino acid
#     substitution.
#     """
#     # NOTE: This could be removed for speed after checking it is not used later
#     next_generation = copy.deepcopy(current_generation)
#     for q in range(n_mutations_per_gen):  # impliment gamma
#
#         fitnesses = calculate_generation_fitness(next_generation, fitness_table)
#         counter = 0
#         successful_mutation = False
#         while not successful_mutation:  # Mutate until all branches have fit proteins
#             successful_mutation = True
#             # Pick random key, clone to make a random generation
#             pi, protein = random.choice(list(next_generation.items()))
#             # Mutate the copy with the randomly chosen residue
#             mutant = mutate_protein(protein, p_mutation, LG_matrix)
#
#             # next_generation[pi] = mutant  # update with new sequence
#             fitness = calculate_stability(mutant, fitness_table)
#
#             if fitness < omega:  # if fitness is less than threshold clone a random sequence in its place.
#                 mutant_index = replace_protein(pi, tree, fitnesses,
#                                                omega)
#                 # If no suitable clones are available, re-mutate the generation and start again
#                 if mutant_index is None:
#                     successful_mutation = False
#                     fitnesses[pi] = fitness  # Save last unsuccessful fitness
#                     # break  # out of loop over fitnesses
#                 else:
#                     next_generation[pi] = next_generation[mutant_index]  # swap out unfit clone for fit clone
#             counter += 1
#             if counter == 100:
#                 raise Exception("Unable to mutate population: "
#                                 "maximum tries exceeded!\n"
#                                 "The mutation rate is too high, mu is too low "
#                                 "or sigma is too small.")
#
#     return next_generation, fitnesses


def mutate_population(current_generation, n_mutations_per_gen, tree,
                      p_mutation, LG_matrix, fitness_table, omega):
    """Mutate a set of sequences based on the LG+I+G model of amino acid
    substitution.
    """
    counter = 0
    successful_mutation = False

    while not successful_mutation:  # Mutate until all branches are fit
        successful_mutation = True
        next_generation = copy.deepcopy(current_generation)
        for q in range(n_mutations_per_gen):  # impliment gamma
            # Pick random key, clone to make a random generation
            pi, protein = random.choice(list(next_generation.items()))
            # Mutate the copy with the randomly chosen residue
            mutant = mutate_protein(protein, p_mutation, LG_matrix)
            next_generation[pi] = mutant  # update with new sequence

        stabilities = calculate_population_fitness(next_generation,
                                                   fitness_table)

        for pi in range(len(stabilities)):
            if stabilities[pi] < omega:  # clone a random sequence
                mutant_index = replace_protein(pi, tree, stabilities, omega)
                # If no suitable clones are available, re-mutate and try again
                if mutant_index is None:
                    successful_mutation = False
                    break  # out of loop over stabilities
                else:  # swap out unfit clone for fit clone
                    next_generation[pi] = next_generation[mutant_index]

        counter += 1
        if counter == 100:
            raise Exception("Unable to mutate population: "
                            "maximum tries exceeded!\n"
                            "The mutation rate is too high, mu is too low "
                            "or sigma is too small.")

    return next_generation


def kill_proteins(population, tree, death_rate, fitness_table, omega):
    n_clones = len(population)
    # mortals = random.sample(range(n_clones), int(n_clones*death_rate))
    clones = np.arange(n_clones)
    condemned = clones[np.where(np.random.rand(n_clones) < death_rate)]
    # Recalculate stabilities after all mutations
    stabilities = calculate_population_fitness(population, fitness_table)
    for pi in condemned:
        new_index = replace_protein(pi, tree, stabilities, omega)
        if new_index is None:  # Should never happen
            warnings.warn("Unable to kill protein {}!".format(pi))
            raise Exception("No suitable candidates on branch!")
        else:
            population[pi] = population[new_index]  # Replace dead protein
    return population


def evolve(n_generations, population, fitness_table, omega, sites,
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
    phi_stabilities = get_phi_stability_table(population, fitness_table)
    # Record initial population
    record_generation_stability(0, population, sites, fitness_table, omega,
                                p_mutation, record, out_paths)
    write_fasta_alignment(0, population, out_paths)

    # Store each generation along with its fitness
    Generation = namedtuple('Generation', ['population', 'stabilities'])
    # Create a list of generations and add initial population and fitness
    history = [Generation(population=population, stabilities=phi_stabilities)]
    if record["data"]:
        save_history(0, history, out_paths)

    # TODO: Refactor plot_omega, plot_epsilon
    plot_stability(0, history, fitness_table, omega, plot_omega, plot_epsilon,
                   n_generations, out_paths)

    for gen in trange(n_generations):  # run evolution for n_generations

        # Bifuricate in even generation numbers so every branch on tree has
        # 3 leaves that have been evolving by the last generation
        if gen > 0 and (gen+1) % n_gens_per_bifurcation == 0 \
                   and len(tree["branches"][0]) > 3:
            tree["branches"] = bifurcate_branches(tree["branches"])
            # Write out bifurcations
            # tree_log_file = os.path.join(run_path, "tree", "tree.txt")
            write_tree(gen+1, tree, out_paths)

        # Mutate population
        next_generation = mutate_population(population, n_mutations_per_gen,
                                            tree, p_mutation, LG_matrix,
                                            fitness_table, omega)

        # Allow sequences to die and be replacecd at a predefined rate
        if death_rate > 0:
            next_generation = kill_proteins(next_generation, tree, death_rate,
                                            fitness_table, omega)

        phi_stabilities = get_phi_stability_table(population, fitness_table)
        # The population becomes next_generation only if bifurcations (and deaths) were successful
        population = next_generation
        # Record intermediate fitnesses to show existence of unfit proteins
        history.append(Generation(population=population, stabilities=phi_stabilities))

        # Write fasta every record["fasta_rate"] generations
        if (gen+1) % record["fasta_rate"] == 0:
            write_fasta_alignment(gen+1, population, out_paths)
        # Record population details at the end of processing
        if (gen+1) % record["rate"] == 0:
            record_generation_stability(gen+1, population, sites, fitness_table,
                                        omega, p_mutation, record, out_paths)

            plot_stability(gen+1, history, fitness_table, omega,
                           plot_omega, plot_epsilon, n_generations, out_paths)
            if record["data"]:
                save_history(gen+1, history, out_paths)

    write_final_fasta(population, tree, out_paths)
    return history


def pesst(n_generations=2000, stability_start='high', omega=0,
          mu=0, sigma=2.5, skew=0,
          n_clones=52, n_roots=4, clone_size=100, p_invariant=0.1,
          mutation_rate=0.001, death_rate=0.02,
          gamma_kwargs=None, record_kwargs=None, output_dir=None, seed=None):

    # Validate arguments
    assert 1 < clone_size
    assert 2 < n_roots < n_clones
    assert 0.0 <= mutation_rate <= 1.0
    assert 0.0 <= death_rate <= 1.0
    assert 0.0 <= p_invariant < 1.0  # Must be less than 1 for evolution
    n_invariants = int(p_invariant * clone_size)
    assert 0 <= n_invariants < clone_size

    # TODO: Add rerun flag to load settings (and seed)
    # settings = json.load(sf)
    if omega is None:
        omega = -np.inf

    # TODO: switch from random to np.random for proper seeding
    if seed is None:
        # Get a random integer to seed both
        seed = random.randrange(2**32)
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
    record_kwargs.setdefault('residues', False)
    record_kwargs.setdefault('statistics', True)
    record_kwargs.setdefault('histograms', True)
    record_kwargs.setdefault('data', True)
    record_kwargs.setdefault('gif', True)
    record = record_kwargs  # TODO: Remove after renaming args and below

    # Create output folder and subfolders
    # PWD = os.path.dirname(__file__)
    out_paths = create_output_folders(output_dir)
    # run_path = out_paths["results"]
    settings_kwargs = {"n_generations": n_generations,
                       "stability_start": stability_start,
                       "omega": omega,
                       "mu": mu,
                       "sigma": sigma,
                       "skew": skew,
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
    plot_LG_matrix(LG_matrix, out_paths)
    # Make fitness table of Delta T_m values
    fitness_table = get_stability_table(clone_size, mu, sigma, skew, LG_matrix.columns)
    if isinstance(stability_start, str) and stability_start.lower() == "low":
        warnings.warn("With 'low' starting fitness selected Omega is ignored.")
                      # "If the run fails, please check your fitness threshold,"
                      # "omega, is low enough: {}".format(omega))
        plot_omega, plot_epsilon = False, True
        omega = -np.inf
    else:
        # epsilon = clone_size * np.mean(fitness_table.values)
        epsilon = clone_size * np.mean(fitness_table.values)  # mu
        if omega < epsilon:
            plot_omega, plot_epsilon = True, True
        else:
            plot_omega, plot_epsilon = True, False
    write_stability_table(fitness_table, out_paths)
    plot_stability_table(fitness_table, out_paths)
    T_max = sum(np.amax(fitness_table, axis=1))  # Fittest possible protein
    assert omega < T_max

    # Generate variant/invariant sites
    # TODO: return boolean array where True is variant
    sites = get_allowed_sites(clone_size, n_invariants)
    # Generate mutation probabilities for every site
    p_mutation = gamma_ray(clone_size, sites, gamma, out_paths)  # TODO: Move plotting out

    # Generate a protein of specified fitness taking into account the invariant
    # sites created (calling variables in this order stops the evolutionary
    # process being biased by superfit invariant sites.)
    # phi
    initial_protein = get_fit_protein(stability_start, clone_size, sites,
                                      fitness_table)
    # print_protein(initial_protein)
    write_initial_protein(initial_protein, out_paths)  # Record initial protein
    initial_fitness = calculate_stability(initial_protein, fitness_table)
    if initial_fitness < omega:
        raise Exception("The fitness threshold is too high!")
    # assert initial_fitness < T_max

    initial_population = clone_protein(initial_protein, n_clones)  # copy
    # Population = namedtuple('Population', ['proteins', 'sites', 'fitnesses'])

    # Generate list of clone keys for bifurication
    tree = create_tree(n_clones, n_roots)
    write_roots(tree["roots"], out_paths)
    write_tree(0, tree, out_paths)

    history = evolve(n_generations, initial_population, fitness_table, omega,
                     sites, p_mutation, mutation_rate, death_rate, tree,
                     LG_matrix, plot_omega, plot_epsilon, record, out_paths)

    # legend_title = "; ".join([r"$\mu$ = {}".format(mu),
    #                           r"$\sigma$ = {}".format(sigma),
    #                           "skew = {}".format(skew),
    #                           r"$\delta$ = {}".format(mutation_rate)])
    plot_evolution(history, fitness_table, omega, plot_omega, plot_epsilon,
                   out_paths)  # , legend_title=legend_title)

    # Create animations
    if record["gif"]:
        recorded_generations = list(range(0, n_generations+1, record["rate"]))
        figures = ["pesst_gen", "phi_fitness_table"]
        if record["residues"]:
            figures.extend(["fit_dist_gen", "generation"])
        if record["histograms"]:
            figures.append("histogram")
        for fig_base in figures:
            path_root = os.path.join(out_paths["figures"], fig_base)
            filenames = [path_root+"_{}.png".format(gen)
                         for gen in recorded_generations]
            create_gif(filenames, duration=0.25)

        # if record["statistics"]:
        #     path_root = os.path.join(out_paths["figures"], "generation_")
        #     filenames = [path_root+"{}.png".format(gen)
        #                  for gen in record_generations]
        #     create_gif(filenames, duration=0.25)
    return history, out_paths