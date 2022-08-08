from collections import Counter
import os
import shutil
# import csv

import pandas as pd
try:
    from Bio import SeqIO
except ImportError:
    raise ImportError("This module requires Biopython!")

from pesst.evolution import calculate_stability
from pesst.utilities import compact_protein


def process_fasta(stability_table, fasta_file, out_paths):
    # stabilitymatrix = input("stability matrix location (must be CSV from previous evolution run): ")
    # fasta = input("input fasta file location (must be a sequence or list of sequences in fasta format): ")

    stability_df = pd.read_csv(stability_table, index_col="Position")

    # records = list(SeqIO.parse("example.fasta", "fasta"))
    seqlib = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    stabilities = {seq_id: [calculate_stability(list(seq), stability_df), seq]
                   for seq_id, seq in seqlib.items()}

    base_name, _ = os.path.splitext(os.path.basename(fasta_file))
    calcs_path = os.path.join(out_paths["fastas"], f"fitcalc_{base_name}")
    if not os.path.exists(calcs_path):
        os.makedirs(calcs_path)
    else:
        shutil.rmtree(calcs_path)
        os.makedirs(calcs_path)

    with open(os.path.join(calcs_path, base_name+".csv"), "w+") as cf:
        cf.write("clone,stability,sequence\n")
        for seq_id, (stability, sequence) in stabilities.items():
            cf.write(f"{seq_id},{stability},{sequence}\n")

    return stabilities


def write_consensus(fasta_file, n_top=3):  # , out_paths):
    """Load a fasta file of sequences and output a fasta file with the most
    prevalent amino acids in each position."""
    
    # consensus_path = os.path.join(out_paths["results"], "consensus")
    # if os.path.isdir(consensus_path):
    #     shutil.rmtree(consensus_path)
    # os.makedirs(consensus_path, exist_ok=True)

    base, extension = os.path.splitext(fasta_file)
    consensus_file = f"{base}_consensus.fasta"  # base includes any directories
    
    # Dictionary of id, sequence mappings
    seqlib = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    sequences = list(seqlib.values())
    seqlen = len(sequences[0].seq)  # Length of first sequence - assume all are the same
    
    # Group amino acids by position
    position_groups = [[] for s in range(seqlen)]
    for position in range(seqlen):
        for sequence in sequences:
            position_groups[position].append(str(sequence.seq)[position])

    # Count amino acid counts in each position
    position_counts = [Counter(group) for group in position_groups]

    # Collect the n_top most frequent amino acids in each position (or repeat the most frequent)
    consensus_proteins = [[] for n in range(n_top)]
    for counts in position_counts:
        n_top_amino_acids = counts.most_common(n_top)
        if len(n_top_amino_acids) < n_top:  # Pad list with last value
            for pad in range(n_top - len(n_top_amino_acids)):
                n_top_amino_acids.append(n_top_amino_acids[-1])
        previous_amino_acid, previous_count = n_top_amino_acids[0]
        for c, (amino_acid, count) in enumerate(n_top_amino_acids):
            if count == previous_count:
                consensus_proteins[c].append(amino_acid)
                previous_amino_acid, previous_count = amino_acid, count
            else:
                consensus_proteins[c].append(previous_amino_acid)

    # Write fasta file with the unique consensus proteins from the n_top candidates
    previous_protein = None
    with open(consensus_file, "w") as fh:
        for c_index in range(n_top):
            protein = compact_protein(consensus_proteins[c_index])
            if protein != previous_protein:
                fh.write(f">consensus_{c_index + 1}\n")
                fh.write(protein)
                fh.write("\n")
                previous_protein = protein
            else:
                break
            