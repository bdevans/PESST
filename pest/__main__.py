import matplotlib as mpl

from pest.evolution import pest

# NOTE: __path__ is initialized to be a list containing the name of the directory holding the package’s __init__.py

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# matplotlib.use('TkAgg')
mpl.rc('savefig', dpi=300)

# TODO: Give these default values

# parameters of protein evolution
n_generations = 2000  # amount of generations the protein evolves for
fitness_start = 'medium'  # high, medium or low; must be lower case. If selecting low, fitness threshold needs to be significantly smaller (i.e. 4x) than #positions*mu
# TODO: Create parameters for the numeric literals which define the medium boundaries
fitness_threshold = 0  # arbitrary number for fitness threshold
# parameters for normal distribution used to select fitness values
mu = -1.2
sigma = 2.5

# TODO: These could possibly go into their own dictionary too
n_clones = 52  # amount of clones that will be generated in the first generation
n_amino_acids = 80  # number of amino acids in the protein including the start methionine
mutation_rate = 0.001  # should be small!
# TODO: Allow user to pass a number but default to None and calculate as follows
n_mutations_per_gen = int(n_clones*(n_amino_acids)*mutation_rate)  # number of mutations per generation
n_anchors = int((n_amino_acids)/10)  # amount of invariant sites in a generation (not including root)
deaths_per_generation = 5  # Set to 0 to turn off protein deaths
death_ratio = 0.05
seed = 42

# TODO: Place bifurcation parameters into kwargs dict with a flag for bifurcations
n_roots = 4

# TODO: Put into dictionary
# parameters for forming discrete gamma distribution used for evolution of protein
gamma = {"shape": 1.9,  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
         "scale": 1/1.9,  # NOTE: 1/gamma_shape. Set as default in func?
         "iterations": 50,
         "samples": 10000}

# Set what to record
record = {"rate": 50,           # write a new fasta file every x generations
          "fasta_rate": 50,     # write a new fasta file every x generations
          "dot_fitness": True,
          "hist_fitness_stats": True,
          "hist_fitness": True,
          "invariants": False}

history = pest(n_generations, fitness_start, fitness_threshold, mu, sigma,
               n_clones, n_amino_acids, mutation_rate, n_mutations_per_gen,
               n_anchors, deaths_per_generation, death_ratio, seed,
               n_roots, gamma, record)
