from src.gpu_utils import get_array_module, GPU_AVAILABLE

# Map DNA bases to integers for fast numerical array operations
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def encode_sequence(seq, xp):
    """Converts a DNA string into a 1D numerical array on the active device."""
    encoded = [BASE_MAP.get(base.upper(), -1) for base in seq]
    return xp.array(encoded, dtype=xp.int8)


def release_gpu_memory():
    """Frees unused memory blocks from the CuPy memory pool to prevent OOM."""
    if GPU_AVAILABLE:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def gpu_align(genome_sequence, target_crispr, max_mismatches=1, chunk_size=500_000):
    """
    Memory-optimized parallel CRISPR alignment engine.
    Processes large genomes in overlapping chunks to prevent GPU memory overflow
    while scanning windows simultaneously within each batch.
    """
    xp = get_array_module()
    target_len = len(target_crispr)
    genome_len = len(genome_sequence)

    if genome_len < target_len or target_len == 0:
        return []

    target_arr = encode_sequence(target_crispr, xp)
    all_matches = []
    step = max(1, chunk_size - target_len + 1)

    for start in range(0, genome_len - target_len + 1, step):
        end = min(start + chunk_size, genome_len)
        chunk_arr = encode_sequence(genome_sequence[start:end], xp)
        num_windows = len(chunk_arr) - target_len + 1

        if num_windows <= 0:
            break

        # Construct 2D sliding-window matrix for the current memory-safe chunk
        idx = xp.arange(target_len)[None, :] + xp.arange(num_windows)[:, None]
        windows = chunk_arr[idx]

        # Compare all windows in the chunk against the target in parallel
        mismatches = xp.sum(windows != target_arr[None, :], axis=1)
        local_hits = xp.where(mismatches <= max_mismatches)[0]

        hits_list = local_hits.get().tolist() if hasattr(local_hits, "get") else local_hits.tolist()
        all_matches.extend([start + offset for offset in hits_list])

        # Explicitly release chunk arrays and clear GPU memory pool
        del chunk_arr, idx, windows, mismatches, local_hits
        release_gpu_memory()

    return all_matches


if __name__ == "__main__":
    sample_genome = "AGCTTAGCTAGCTACGATCGATCGATCGATCGTAGCTAGCTAG" * 100
    crispr_guide = "CGATCGATCG"
    hits = gpu_align(sample_genome, crispr_guide, max_mismatches=1, chunk_size=200)
    print(f"Chunked parallel alignment found {len(hits)} total matches.")