import pandas as pd
import numpy as np

def parse_fasta(file):
    """
    Parse a FASTA file into a DataFrame.

    Args:
        file: File object containing FASTA data.

    Returns:
        pd.DataFrame: DataFrame with columns 'id' and 'sequence'.
    """
    sequences = []
    seq_id = None
    seq_lines = []
    for line in file:
        line = line.decode("utf-8").strip()
        if line.startswith(">"):
            if seq_id:
                sequences.append({"id": seq_id, "sequence": "".join(seq_lines)})
            seq_id = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)
    if seq_id:
        sequences.append({"id": seq_id, "sequence": "".join(seq_lines)})
    return pd.DataFrame(sequences)

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

def one_hot_encode(seq, seq_len=127):
    """
    One-hot encode a DNA sequence.

    Args:
        seq (str): DNA sequence.
        seq_len (int): Target sequence length (default: 127).

    Returns:
        np.ndarray: One-hot encoded array of shape (seq_len, 5).
    """
    seq = seq.upper()
    if len(seq) < seq_len:
        seq = seq + "N" * (seq_len - len(seq))
    else:
        seq = seq[:seq_len]
    arr = np.zeros((seq_len, 5), dtype=np.float32)
    for i, base in enumerate(seq):
        arr[i, BASE2IDX.get(base, 4)] = 1.0
    return arr