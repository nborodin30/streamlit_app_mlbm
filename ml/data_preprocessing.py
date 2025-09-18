import os
from Bio import SeqIO
from typing import Dict, List
from hydra.core.hydra_config import HydraConfig

def load_fasta(path: str) -> Dict[str, str]:
    project_root = HydraConfig.get().runtime.cwd
    full_path = os.path.join(project_root, path)
    with open(full_path, "r") as f:
        return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(f, "fasta")}

def preprocess_windows(ref: Dict[str, str], mask: Dict[str, str], window_size: int) -> List[Dict]:
    data = []
    for chrom in sorted(ref.keys()):
        seq = ref[chrom]
        mask_seq = mask[chrom]
        assert len(seq) == len(mask_seq), f"Length mismatch on {chrom}"
        seq_len = len(seq)
        for start in range(0, seq_len - window_size + 1, window_size):
            end = start + window_size
            window_seq = seq[start:end]
            window_mask = mask_seq[start:end]
            n_count = window_seq.count('N')
            if n_count > 0.1 * window_size:
                continue
            center_idx = window_size // 2
            center_nuc = window_seq[center_idx]
            label = int(window_mask[center_idx])
            data.append({
                'chrom': chrom,
                'start': start,
                'sequence': window_seq,
                'center_nucleotide': center_nuc,
                'label': label
            })
    return data

def balance_dataset(data: List[Dict], seed: int) -> List[Dict]:
    import random
    random.seed(seed)
    exons = [d for d in data if d['label'] == 1]
    introns = [d for d in data if d['label'] == 0]
    n_minority = min(len(exons), len(introns))
    exons_balanced = random.choices(exons, k=n_minority)
    introns_balanced = random.choices(introns, k=n_minority)
    return exons_balanced + introns_balanced