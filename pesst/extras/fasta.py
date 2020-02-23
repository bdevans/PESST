import os
import shutil
# import csv

import pandas as pd
try:
    from Bio import SeqIO
except ImportError:
    raise ImportError("This module requires Biopython!")

from pesst.evolution import calculate_stability


def process_fasta(stability_matrix, fasta_file, output_path):
    # stabilitymatrix = input("stability matrix location (must be CSV from previous evolution run): ")
    # fasta = input("input fasta file location (must be a sequence or list of sequences in fasta format): ")

    stability_df = pd.read_csv(stability_matrix, index_col="Position")

    # records = list(SeqIO.parse("example.fasta", "fasta"))
    seqlib = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    stabilities = {id: [calculate_stability(list(seq), stability_df), seq]
                   for id, seq in seqlib.items()}

    base_name, _ = os.path.splitext(os.path.basename(fasta_file))
    calcs_path = os.path.join(output_path, f"fitcalc_{base_name}")
    if not os.path.exists(calcs_path):
        os.makedirs(calcs_path)
    else:
        shutil.rmtree(calcs_path)
        os.makedirs(calcs_path)

    with open(os.path.join(calcs_path, base_name+".csv"), "w+") as cf:
        cf.write("clone,stability,sequence\n")
        for id, (stability, sequence) in stabilities.items():
            cf.write(f"{id},{stability},{sequence}\n")

    return stabilities
