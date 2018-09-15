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

from .dataio import (create_output_folders, write_settings_file, load_LG_matrix,
                     write_roots, write_initial_protein, write_protein_fitness,
                     write_fasta_alignment, write_final_fasta,
                     write_histogram_statistics, append_ks_statistics)
from .plotting import (plot_evolution, plot_gamma_distribution,
                       plot_threshold_fitness, plot_histogram_of_fitness)
# from .utilities import print_protein

# TODO: Automatically extract from LG_matrix - DONE
# Define RESIDUES matching the ordering of LG_matrix
# RESIDUES_NAME = "ARNDCQEGHILKMFPSTWYV"  # Strings are immutable
# RESIDUES_INDEX = {aa: ai for ai, aa in enumerate(RESIDUES_NAME)}  # Faster than calling .index()
# Residues = namedtuple('Residues', ['name', 'index'])
# RESIDUES = Residues(name=RESIDUES_NAME, index=RESIDUES_INDEX)


def get_fitness_table(n_amino_acids, mu, sigma, LG_matrix):
    """Generate a dictionary describing list of fitness values at each position
    of the generated protein.
    """
    n_variants = len(LG_matrix.columns)
    values = np.random.normal(mu, sigma, size=(n_amino_acids, n_variants))
    fitness_table = pd.DataFrame(values, columns=LG_matrix.columns)
    return fitness_table


def clone_protein(protein, n_clones):
    """Generate a dictionary containing n_clones of generated protein
    - this contains the evolving dataset.
    """
    return {l: copy.deepcopy(protein) for l in range(n_clones)}


def get_allowed_sites(n_amino_acids, n_anchors):
    """Select invariant sites in the initially generated protein and return
    allowed values.
    """
    # TODO: Methionine is always the first anchor, so change to range(n_amino_acids) and add one to n_anchors definition
    allowed_values = list(range(1, n_amino_acids))  # keys for mutable sites
    # Randomly define invariant sites (without replacement)
    anchored_sequences = random.sample(allowed_values, n_anchors)
    anchored_sequences.insert(0, 0)  # First aa is always anchored (Methionine)
    # Remove the invariant sites from allowed values
    for a in anchored_sequences[1:]:
        allowed_values.remove(a)
    # anchored_sequences = [allowed_values.pop(random.randrange(len(allowed_values))) for r in range(n_anchors)]
    Sites = namedtuple('Sites', ['invariant', 'variant'])
    # Return a namedtuple with anchors and available sites
    return Sites(invariant=anchored_sequences, variant=allowed_values)


def gamma_ray(n_amino_acids, sites, gamma):
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

    kappa, theta, n_iterations, n_samples = (gamma["shape"], gamma["scale"],
                                             gamma["iterations"], gamma["samples"])

    medians = np.zeros(shape=(n_iterations, 4))

    for i in range(n_iterations):
        # Draw n_samples from the gamma distribution
        samples = np.random.gamma(kappa, theta, n_samples)
        # Define quartiles in that data with equal probability
        quartiles = np.percentile(samples, (0, 25, 50, 75, 100),
                                  interpolation='midpoint')
        # Find the median of each quartile
        medians[i, :], _, _ = sp.stats.binned_statistic(samples, samples,
                                                        statistic='median',
                                                        bins=quartiles)
    # Calculate average of medians across iterations
    average_medians = np.mean(medians, axis=0)

    # Replot the gamma distributuion as a check
    if False:
        plot_gamma_distribution(gamma, quartiles, average_medians)

    gamma_categories = np.random.choice(average_medians, size=n_amino_acids)
    gamma_categories[sites.invariant] = 0
    return gamma_categories/sum(gamma_categories)  # p_location


def mutate_protein(protein, p_location, LG_matrix):
    """Mutate a residue to another residue based on the LG matrix."""
    mutant = copy.deepcopy(protein)  # Necessary!
    location = np.random.choice(len(mutant), p=p_location)
    amino_acid = mutant[location]
    p_transition = LG_matrix.loc[amino_acid]
    LG_residues = LG_matrix.columns.values  # .tolist()
    mutant[location] = np.random.choice(LG_residues, p=p_transition)
    return mutant


def calculate_fitness(protein, fitness_table):
    """Calculate fitness of a protein given the sequence and fitness values."""
    protein_fitness = [fitness_table.loc[ai, amino_acid]  # TODO: Indexing
                       for ai, amino_acid in enumerate(protein)]
    return sum(protein_fitness)


def get_random_protein(n_amino_acids, fitness_table, start_amino_acid="M"):
    """Generate an original starting protein n_amino_acids long with a start
    amino acid set to methionine.
    """
    protein = random.choices(fitness_table.columns.values.tolist(), k=n_amino_acids-1)  # RESIDUES.name
    # protein = [random.choice(RESIDUES) for _ in range(n_amino_acids)]  # ver < 3.6
    protein.insert(0, start_amino_acid)  # Start with methionine
    # TODO: Convert to strings preventing in-place modification
    # return ''.join(protein)
    return protein


def twist_protein(protein, mutation_sites, fitness_table):
    mutant = protein[:]  # copy.deepcopy(start_protein)
    amino_acids = fitness_table.columns.values.tolist()
    for ai in mutation_sites:
        mutant[ai] = random.choice(amino_acids)  # RESIDUES.name
    fitness = calculate_fitness(mutant, fitness_table)
    return (mutant, fitness)


def get_fit_protein(fitness_level, n_amino_acids, sites, fitness_table, fitness_threshold):
    """Generate a protein of a specified fitness.

    Make either a superfit protein, a superunfit protein or a 'medium
    fitness' protein with fitness just higher than the fitness threshold.

    This function currently locks in invariant sites before finding the fittest
    sites, meaning the invariant sites are simply sampled from the normal
    distribution, and not from the superfit distribution.

    Generating the protein in this manner avoids bias towards increased fitness
    that could be generated by the invariant sites.
    """
    initial_protein = get_random_protein(n_amino_acids, fitness_table)

    if fitness_level != 'medium':
        if fitness_level == 'low':  # generate unfit protein
            # TODO: Adjust threshold in this case or unable to build proteins above threshold
            sequence = [0, 1, 2]  # Three lowest

        elif fitness_level == 'high':  # generate superfit protein
            sequence = [-1, -2, -3]  # Three highest

        aminos = [["M"] * 3]
        for ai in range(1, len(fitness_table.index)):  # range(n_amino_acids):
            if ai in sites.invariant:
                aminos.append([initial_protein[ai]] * 3)
            else:
                # i_sorted = np.argsort(fitness_table.loc[ai])
                # # aminos.append([RESIDUES[fitness_table[ai, i_sorted[start]]]
                # #                for start in sequence])
                # aminos.append([RESIDUES.name[i_sorted[start]] for start in sequence])
                # sorted_aa = sorted(fitness_table[ai], key=lambda k: fitness_table[ai][k])
                # DataFrame
                # sorted_aa = RESIDUES.name[fitness_table.loc[ai].argsort()]
                sorted_aa = fitness_table.columns.values[fitness_table.loc[ai].argsort()]
                # sorted_aa = fitness_table.sort_values(fitness_table.loc[ai], axis=1).columns.tolist()
                aminos.append([sorted_aa[rank] for rank in sequence])
        protein = []
        # Generate a superunfit protein by randomly picking one of the 3 most fit amino acids at each position
        for candidates in aminos:
            protein.append(random.choice(candidates))

    # Generate medium fitness protein. This module is a little buggy. It takes
    # the starting protein sequence, mutates 5 residues until the protein is
    # fitter, then chooses 5 new residues and continues.
    # If it cannot make a fitter protein with the 5 residues its mutating it
    # reverts back to the previous state and picks 5 new residues.
    elif fitness_level == 'medium':
        # TODO: Parameterise medium fitness bounds as arguments
        n_variants = 5
        # start_protein = initial_protein  # Copies the external initial_protein
        initial_fitness = calculate_fitness(initial_protein, fitness_table)
        # NOTE: This chooses from anchored_sequences whereas the other conditions exclude them FIXED
        protein = initial_protein[:]  # copy.deepcopy(initial_protein)
        # TODO: This is slightly different to the original algorithm (below) but should give similar results
        # fitness = initial_fitness
        # counter = 0
        # while (fitness < fitness_threshold+10 or fitness > fitness_threshold+20) and counter <= 100:
        #     # Mutate the new protein (sample without replacement)
        #     chosen_variants = random.sample(sites.variant, n_variants)
        #     (protein, fitness) = twist_protein(initial_protein, chosen_variants, fitness_table)
        #     counter += 1

        while initial_fitness < fitness_threshold+10 or initial_fitness > fitness_threshold+20:
            # Mutate the new protein (sample without replacement)
            chosen_variants = random.sample(sites.variant, n_variants)
            (protein, fitness) = twist_protein(initial_protein, chosen_variants, fitness_table)
            counter = 0

            if initial_fitness < fitness_threshold+10:  # setting lower bounds of medium fitness
                while fitness < initial_fitness and counter <= 100:
                    # Continue to mutate until it is better than the initial_protein
                    (protein, fitness) = twist_protein(initial_protein, chosen_variants, fitness_table)
                    counter += 1

            elif initial_fitness > fitness_threshold+20:  # set upper bounds of medium fitness
                while fitness > initial_fitness and counter <= 100:
                    # Continue to mutate until it is better than the initial_protein
                    (protein, fitness) = twist_protein(initial_protein, chosen_variants, fitness_table)
                    counter += 1

            initial_protein = protein
            initial_fitness = calculate_fitness(protein, fitness_table)
        protein = initial_protein

    return protein


def mutate_population(current_generation, n_mutations_per_gen, tree,
                      variant_sites, p_location, LG_matrix,
                      fitness_table, fitness_threshold):
    """Mutate a set of sequences based on the LG+I+G model of amino acid
    substitution.
    """
    counter = 0
    successful_mutation = False

    while not successful_mutation:  # Mutate until all branches have fit proteins
        successful_mutation = True

        # NOTE: This could be removed for speed after checking it is not used later
        next_generation = copy.deepcopy(current_generation)
        for q in range(n_mutations_per_gen):  # impliment gamma
            # Pick random key, clone to make a random generation
            pi, protein = random.choice(list(next_generation.items()))
            # Mutate the copy with the randomly chosen residue
            mutant = mutate_protein(protein, p_location, LG_matrix)
            next_generation[pi] = mutant  # update with new sequence

        fitnesses = calculate_population_fitness(next_generation, fitness_table)

        for pi in range(len(fitnesses)):  # if there are, start loop on fitnesses
            if fitnesses[pi] < fitness_threshold:  # if fitness is less than threshold clone a random sequence in its place.

                mutant_index = replace_protein(pi, tree, fitnesses,
                                               fitness_threshold)
                # If no suitable clones are available, re-mutate the generation and start again
                if mutant_index is None:
                    successful_mutation = False
                    break  # out of loop over fitnesses
                else:
                    next_generation[pi] = next_generation[mutant_index]  # swap out unfit clone for fit clone

        counter += 1
        if counter == 100:
            raise Exception("Unable to mutate population: "
                            "maximum tries exceeded!\n"
                            "The mutation rate is too high, mu is too low "
                            "or sigma is too small.")

    return next_generation, fitnesses


def calculate_population_fitness(population, fitness_table):
    """Calculate the fitness of every protein in a population."""
    # Record calculated fitness for each protein in dictionary
    return {pi: calculate_fitness(protein, fitness_table)
            for pi, protein in list(population.items())}
            # for pi, protein in enumerate(population)}
    # TODO: Replace with list once population is a list (or OrderedDict)
    # fitnesslist = [calculate_fitness(protein, fitness_table)
    #                for pi, protein in list(population.items())]


def record_generation_fitness(generation, population, variant_sites,
                              fitness_table, fitness_threshold, record, run_path):
    """Record the fitness of every protein in the generation and store them in
    dictionary. Optionally generate data and figures about fitness.
    """

    # Build distribution of fitness values existing in evolving protein
    fitnesses = build_generation_fitness_table(population, variant_sites, fitness_table, record)

    if record["dot_fitness"]:
        save_dir = os.path.join(run_path, "fitnessdotmatrix")
        plot_threshold_fitness(generation, population, fitnesses, fitness_table, fitness_threshold, save_dir)

    if record["hist_fitness_stats"]:
        # Write a file describing 5 statistical tests on the protein fitness space
        if generation == 0:
            stats_file_name = "normal_distribution_statistics_fitness_space.txt"
            distributions = fitness_table.values
        else:
            stats_file_name = "normal_distribution_statistics_generation{}.txt".format(generation)
            distributions = fitnesses
        stats_full_name = os.path.join(run_path, "fitnessdistribution",
                                       "statistics", stats_file_name)
        write_histogram_statistics(stats_full_name, distributions, record)
        if generation > 0:
            append_ks_statistics(stats_full_name, distributions.ravel(),
                                 fitness_table.values.ravel())

    if record["hist_fitness"]:
        disthistfilename = "generation_{}.png".format(generation)
        disthistfullname = os.path.join(run_path, "fitnessdistribution",
                                        "histograms", disthistfilename)
        plot_histogram_of_fitness(disthistfullname, fitnesses.ravel(),
                                  fitness_table.values.ravel(), fitness_threshold)


def build_generation_fitness_table(population, variant_sites, fitness_table, record):
    """Build a fitness table for given generation's population.

    The array has one row for each protein in the population and the fitness
    value for each amino acid in its position.
    """
    dist_clone_fitness = []
    # Find and plot all fitness values in the current generation
    for pi, protein in list(population.items()):

        if record["invariants"]:
            protein_fitness = [fitness_table.loc[ai, amino_acid]
                               for ai, amino_acid in enumerate(protein)]
        else:
            protein_fitness = [fitness_table.loc[ai, amino_acid]
                               for ai, amino_acid in enumerate(protein)
                               if ai in variant_sites]

        dist_clone_fitness.append(protein_fitness)  # Becomes a new row
    return np.asarray(dist_clone_fitness)


def select_from_pool(protein_index, candidates, fitnesses, fitness_threshold):
    # Filter out original protein and those below the fitness threshold
    pool = [c for c in candidates
            if c != protein_index and fitnesses[c] >= fitness_threshold]
    if len(pool) > 0:
        new_protein_index = random.choice(pool)
    else:
        # raise Exception("No suitable candidates on branch!")
        # warnings.warn("No suitable candidates on branch!")
        # new_protein = np.nan
        new_protein_index = None
        # print({c: fitnesses[c] for c in candidates})
    return new_protein_index


def replace_protein(protein_index, tree, fitnesses, fitness_threshold):

    if protein_index in tree["roots"]:
        new_index = select_from_pool(protein_index, tree["roots"], fitnesses,
                                     fitness_threshold)
    else:  # Protein is in one of the branches
        for branch in tree["branches"]:
            if protein_index in branch:
                new_index = select_from_pool(protein_index, branch, fitnesses,
                                             fitness_threshold)
    return new_index


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
    tree["branches"] = [protein_keys]  # lists of protein keys intialised with non-roots
    return tree


def calc_gens_per_bifurcation(n_generations, n_clones, n_roots):
    pool = n_clones - n_roots
    n_phases = 1  # n_bifurations + 1
    # NOTE: Originally this also claimed to stop with 6 leaves per branch
    while pool >= 6:  # stop when there are 3, 4 or 5 leaves per branch
        pool //= 2  # Floor division
        n_phases += 1
    return int(n_generations/n_phases)  # number of generations per bifurcation


def bifurcate_branches(branches):
    new_bifurcations = []  # temporary store for new bifurcations
    for branch in branches:  # bifuricate each set of leaves
        random.shuffle(branch)
        midpoint = int(len(branch)/2)
        new_bifurcations.append(branch[:midpoint])
        new_bifurcations.append(branch[midpoint:])
    return new_bifurcations[:]


def kill_proteins(population, tree, death_ratio, fitnesses, fitness_threshold):
    n_clones = len(population)
    mortals = random.sample(range(n_clones), int(n_clones*death_ratio))
    for pi in mortals:
        mortal_index = replace_protein(pi, tree,
                                       fitnesses, fitness_threshold)
        if mortal_index is None:  # Should never happen
            warnings.warn("Unable to kill protein {}!".format(pi))
            raise Exception("No suitable candidates on branch!")
        else:
            population[pi] = population[mortal_index]  # Replace dead protein
    return population


def evolve(n_generations, initial_population, fitness_table, fitness_threshold,
           variant_sites, p_location, n_mutations_per_gen,
           n_gens_per_death, death_ratio,
           n_roots, LG_matrix, record, run_path):
    """Generation generator - mutate a protein for a defined number of
    generations according to an LG matrix and gamma distribution.
    """

    n_clones = len(initial_population)
    # Generate list of clone keys for bifurication
    tree = create_tree(n_clones, n_roots)

    rootsfullname = os.path.join(run_path, "start", "Roots.txt")
    write_roots(rootsfullname, tree["roots"])

    # Calculate number of bifurications per generation.
    n_gens_per_bifurcation = calc_gens_per_bifurcation(n_generations, n_clones,
                                                       n_roots)

    population = copy.deepcopy(initial_population)  # current generation
    fitnesses = calculate_population_fitness(population, fitness_table)
    # Record initial population
    record_generation_fitness(0, population, variant_sites,
                              fitness_table, fitness_threshold,
                              record, run_path)
    write_fasta_alignment(population, 0, run_path)

    # Store each generation along with its fitness
    Generation = namedtuple('Generation', ['population', 'fitness', 'final_fitness'])
    # Create a list of generations and add initial population and fitness
    evolution = [Generation(population=population, fitness=fitnesses,
                            final_fitness=fitnesses)]

    for gen in trange(n_generations):  # run evolution for n_generations

        # Bifuricate in even generation numbers so every branch on tree has
        # 3 leaves that have been evolving by the last generation
        if gen > 0 and (gen+1) % n_gens_per_bifurcation == 0 \
                   and len(tree["branches"][0]) > 3:
            tree["branches"] = bifurcate_branches(tree["branches"])

        # TODO: Store population with fitnesses in Generation namedtuple and move checks to within mutate_population
        # Mutate population
        (next_generation, fitnesses) = mutate_population(population, n_mutations_per_gen,
                                                         tree, variant_sites, p_location,
                                                         LG_matrix, fitness_table, fitness_threshold)

        # Allow sequences to die and be replacecd at a predefined rate
        if n_gens_per_death > 0 and (gen+1) % n_gens_per_death == 0:
            post_mutation_fitnesses = calculate_population_fitness(next_generation, fitness_table)
            next_generation = kill_proteins(next_generation, tree, death_ratio,
                                            post_mutation_fitnesses, fitness_threshold)

        final_fitnesses = calculate_population_fitness(next_generation, fitness_table)
        # The population becomes next_generation only if bifurcations (and deaths) were successful
        population = next_generation
        # Record intermediate fitnesses to show brief existence of unfit proteins
        evolution.append(Generation(population=population, fitness=fitnesses,
                                    final_fitness=final_fitnesses))
        # Write fasta every record["fasta_rate"] generations
        if (gen+1) % record["fasta_rate"] == 0:
            write_fasta_alignment(population, gen+1, run_path)
        # Record population details at the end of processing
        if (gen+1) % record["rate"] == 0:
            record_generation_fitness(gen+1, population, variant_sites,
                                      fitness_table, fitness_threshold,
                                      record, run_path)

    write_final_fasta(population, tree, run_path)
    return evolution


def pest(n_generations, fitness_start, fitness_threshold, mu, sigma,
         n_clones=52, n_amino_acids=80, mutation_rate=0.001,
         n_mutations_per_gen=None, n_anchors=None,
         n_gens_per_death=5, death_ratio=0.05, seed=None,
         n_roots=4, gamma=None, record=None):

    # TODO: Add rerun flag to load settings (and seed)
    # settings = json.load(sf)
    if n_mutations_per_gen is None:
        n_mutations_per_gen = int(n_clones*(n_amino_acids)*mutation_rate)
    if n_anchors is None:
        n_anchors = int((n_amino_acids)/10)
    # TODO: switch from random to np.random for proper seeding
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if fitness_start == "low":
        warnings.warn("You have selected low starting fitness. "
                      "If the run fails, please check your fitness threshold"
                      "is low enough: {}".format(fitness_threshold))

    # TODO: Put run_path (and subdirs) in record dict
    # create folder and subfolders
    # PWD = os.path.dirname(__file__)
    run_path = create_output_folders()
    settings_kwargs = {"n_generations": n_generations,
                       "fitness_start": fitness_start,
                       "fitness_threshold": fitness_threshold,
                       "mu": mu,
                       "sigma": sigma,
                       "n_clones": n_clones,
                       "n_amino_acids": n_amino_acids,
                       "mutation_rate": mutation_rate,
                       "n_mutations_per_gen": n_mutations_per_gen,
                       "n_anchors": n_anchors,
                       "n_gens_per_death": n_gens_per_death,
                       "death_ratio": death_ratio,
                       "n_roots": n_roots,
                       "seed": seed,
                       "gamma": gamma,
                       "record": record}
    write_settings_file(run_path, settings_kwargs)  # record run settings
    LG_matrix = load_LG_matrix()  # Load LG matrix
    fitness_table = get_fitness_table(n_amino_acids, mu, sigma, LG_matrix)  # make first fitness dictionary
    write_protein_fitness(run_path, "start", fitness_table)

    sites = get_allowed_sites(n_amino_acids, n_anchors)  # generate variant/invariant sites
    p_location = gamma_ray(n_amino_acids, sites, gamma)  # generate mutation probabilities for every site

    # Generate a superfit protein taking into account the invariant sites created (calling variables in this order stops the evolutionary process being biased by superfit invariant sites.)
    initial_protein = get_fit_protein(fitness_start, n_amino_acids, sites, fitness_table, fitness_threshold)
    # print_protein(initial_protein)
    write_initial_protein(initial_protein, run_path)  # Record initial protein
    initial_population = clone_protein(initial_protein, n_clones)  # make some clones to seed evolution

    history = evolve(n_generations, initial_population, fitness_table,
                     fitness_threshold, sites.variant, p_location,
                     n_mutations_per_gen, n_gens_per_death, death_ratio,
                     n_roots, LG_matrix, record, run_path)
    # TODO: Set lines automatically
    plot_omega, plot_epsilon = True, False
    initial_fitness = calculate_fitness(initial_protein, fitness_table)
    plot_evolution(history, n_clones, n_amino_acids, initial_fitness, fitness_table, fitness_threshold, plot_omega, plot_epsilon, mu, sigma, mutation_rate, run_path)
    return history
