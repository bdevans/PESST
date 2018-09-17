import matplotlib as mpl

from pest.evolution import pest

# NOTE: __path__ is initialized to be a list containing the name of the directory holding the package’s __init__.py
# mpl.use('TkAgg')
mpl.rc('savefig', dpi=300)

# parameters of protein evolution
n_generations = 200  # amount of generations the protein evolves for

# TODO: These could possibly go into their own dictionary too
n_clones = 52  # amount of clones that will be generated in the first generation
# TODO: Place bifurcation parameters into kwargs dict with a flag for bifurcations
n_roots = 4
n_gens_per_death = 5  # TODO: Remove
death_rate = 0.05  # Set to 0 to turn off protein deaths

n_amino_acids = 80  # number of amino acids in the protein including the start methionine
# TODO: Allow user to pass a number but default to None and calculate as follows
n_anchors = int(n_amino_acids/10)  # amount of invariant sites in a generation (not including root)

fitness_threshold = 0  # arbitrary number for fitness threshold
fitness_start = (fitness_threshold + 10, fitness_threshold + 20)  # high, (x, y) or low; must be lower case. If selecting low, fitness threshold needs to be significantly smaller (i.e. 4x) than #positions*mu
# parameters for normal distribution used to select fitness values
mu = -1.2
sigma = 2.5
mutation_rate = 0.001  # Proportion of the total amino acids in mutating in the population each gnereation - should be small!

seed = 42

# TODO: Put into dictionary
fitness = {"start": fitness_start,
           "omega": fitness_threshold,
           "mu": mu,
           "sigma": sigma,
           "delta": mutation_rate}

# parameters for forming discrete gamma distribution used for evolution of protein
gamma = {"shape": 1.9,  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
         "scale": 1/1.9,  # NOTE: 1/gamma_shape. Set as default in func?
         "iterations": 50,
         "samples": 10000}

# Set what to record
record = {"rate": 50,           # write a new fasta file every x generations
          "fasta_rate": 50,     # write a new fasta file every x generations
          "dot_fitness": False,
          "hist_fitness_stats": False,
          "hist_fitness": False,
          "invariants": False}

history = pest(n_generations, fitness_start, fitness_threshold, mu, sigma,
               n_clones, n_roots, n_amino_acids, n_anchors, mutation_rate,
               n_gens_per_death, death_rate, seed, gamma, record)
