import os
import pandas as pd

from matplotlib import pyplot as plt

from pesst.evolution import pesst
from pesst.plotting import plot_evolution
from pesst.dataio import load_history, load_settings, load_stability_table
# from pesst.extras.fasta import process_fasta

# TODO: Make this a set of example runs and move back into __main__.py

rerun = True

omega = 5

print(os.getcwd())
output_dir = "Medium"
(history, out_paths) = pesst(n_generations=250, stability_start=(10, 15),
                             omega=0, mu=-1, skew=10, output_dir=output_dir)

settings = load_settings(out_paths)
settings["stability_start"] = 'high'
# (history, out_paths) = pesst(output_dir="High", **settings)
(history, out_paths) = pesst(output_dir="High")
stability_table = load_stability_table(out_paths)

settings["stability_start"] = 'low'
output_dir = "Low"
(history, out_paths) = pesst(output_dir=output_dir, **settings)


output_dir = "Exp1"
if rerun:
    (history, out_paths) = pesst(n_generations=100, stability_start='high',
                                 omega=omega, output_dir=output_dir)
else:  # load data
    (history, generations) = load_history(output_dir)
paper_dir = os.path.join("results", "paper")
os.makedirs(paper_dir)
# stability_table = load_stability_table(out_paths)
stability_table = pd.read_csv(os.path.join(out_paths["initial"], 'stability_table.csv'))
# This is the wrong stability table!
# TODO: Create load_data which calls load_history, load stability_table etc.
fig, ax = plt.subplots(figsize=(12, 8))
plot_evolution(history, stability_table, omega, plot_omega=True, plot_epsilon=True, out_paths=out_paths, ax=ax)
ax.set_ylim(-50, 350)
fig.savefig(os.path.join(paper_dir, "paper_high.png"))


(history, out_paths) = pesst(n_generations=100, stability_start='low', output_dir="Exp2")
paper_dir = os.path.join(out_paths["results"], "paper")
os.makedirs(paper_dir)
stability_table = pd.read_csv(os.path.join(out_paths["initial"], 'stability_table.csv'))
fig, ax = plt.subplots(figsize=(12, 8))
plot_evolution(history, stability_table, omega, plot_omega=False, plot_epsilon=True, out_paths=out_paths, ax=ax)
# ax.set_ylim(-100, 300)
fig.savefig(os.path.join(paper_dir, "paper_low.png"))

# TODO
# settings = load_settings(out_paths)
# settings["sigma"] = 5
# settings["record"]["rate"] = 25
# (history, out_paths) = pesst(**settings)
#
# (history, out_paths) = pesst()
#
# (history, out_paths) = pesst(seed=42)

# (fig, ax_arr) = plot_stability(generation, history, stability_table, omega,
#                    plot_omega, plot_epsilon, n_generations, out_paths)
# [ax_phi, ax_evo, ax_aa_g, ax_aa_0, ax_hist] = ax_arr
# ax_phi.set_ylim()

# process_fasta()
