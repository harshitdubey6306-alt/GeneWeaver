from src.gpu_utils import get_array_module

# Map DNA bases to integers for fast numerical array operations
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def encode_sequence(seq, xp):
    """Converts a DNA string into a 1D numerical array on the active device."""
    encoded = [BASE_MAP.get(base.upper(), -1) for base in seq]
    return xp.array(encoded, dtype=xp.int8)


def gpu_align(genome_sequence, target_crispr, max_mismatches=1):
    """
    Parallelized CRISPR alignment engine.
    Builds a 2D sliding-window matrix and computes mismatches across all
    positions simultaneously on the GPU (or NumPy fallback).
    """
    xp = get_array_module()
    target_len = len(target_crispr)
    genome_len = len(genome_sequence)

    if genome_len < target_len or target_len == 0:
        return []

    genome_arr = encode_sequence(genome_sequence, xp)
    target_arr = encode_sequence(target_crispr, xp)

    num_windows = genome_len - target_len + 1

    # Construct 2D matrix of all sliding windows using vectorized indexing
    idx = xp.arange(target_len)[None, :] + xp.arange(num_windows)[:, None]
    windows = genome_arr[idx]

    # Compare every window against the target sequence in parallel
    mismatches = xp.sum(windows != target_arr[None, :], axis=1)
    match_indices = xp.where(mismatches <= max_mismatches)[0]

    # Transfer results back to standard Python list if running on GPU
    if hasattr(match_indices, "get"):
        return match_indices.get().tolist()
    return match_indices.tolist()


if __name__ == "__main__":
    sample_genome = "AGCTTAGCTAGCTACGATCGATCGATCGATCGTAGCTAGCTAG"
    crispr_guide = "CGATCGATCG"
    hits = gpu_align(sample_genome, crispr_guide, max_mismatches=1)
    print(f"Parallel alignment matches found at indices: {hits}")