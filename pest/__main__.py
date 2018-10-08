import matplotlib as mpl
from matplotlib import pyplot as plt

from pest.evolution import pest

# NOTE: __path__ is initialized to be a list containing the name of the directory holding the package’s __init__.py
# mpl.use('TkAgg')
mpl.rc('savefig', dpi=300)
plt.style.use('seaborn-ticks')

# parameters of protein evolution
n_generations = 200  # amount of generations the protein evolves for

# TODO: These could possibly go into their own dictionary too
n_clones = 52  # S number of clones in the population, phi
# TODO: Place bifurcation parameters into kwargs dict with a flag for bifurcations
n_roots = 4
death_rate = 0.05  # Set to 0 to turn off protein deaths

clone_size = 100  # number of amino acids in the protein including the start methionine
p_invariant = 0.1  # amount of invariant sites in a generation (not including root)

omega = 0  # arbitrary number for fitness threshold
stability_start = (omega + 10, omega + 20)  # high, (x, y) or low
# parameters for normal distribution used to select fitness values
mu = -1.2
sigma = 2.5
skew = 3  # skewnorm.pdf(x, skew) = 2 * norm.pdf(x) * norm.cdf(skew*x)
mutation_rate = 0.001  # Proportion of the total amino acids in mutating in the population each gnereation - should be small!

seed = None  # Maximum seed: 2**32 - 1

# TODO: Put into dictionary
population = {"n_proteins": n_clones,
              "n_roots": n_roots,
              "death_rate": death_rate,
              "bifurcate": True}

stability = {"start": stability_start,
             "omega": omega,  # Stability threshold
             "mu": mu,
             "sigma": sigma,
             "skew": skew,
             "delta": mutation_rate}

# parameters for forming discrete gamma distribution used for evolution of protein
gamma = {"shape": 1.9,  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
         "scale": 1/1.9,  # theta = 1/beta NOTE: 1/gamma_shape. Set as default in func?
         "iterations": 50,
         "samples": 10000}

# Set what to record
record = {"rate": 50,           # write a new fasta file every x generations
          "fasta_rate": 50,     # write a new fasta file every x generations
          "residues": True,
          "statistics": True,
          "histograms": True,
          "invariants": True,
          "gif": False}

history = pest(n_generations, stability_start, omega, mu, sigma, skew,
               n_clones, n_roots, clone_size, p_invariant, mutation_rate,
               death_rate, seed, gamma, record)
