def naive_align(genome_sequence, target_crispr, max_mismatches=1):
    """
    Baseline CPU algorithm to find a target CRISPR sequence in a genome.
    Uses a standard sliding window approach and allows for minor mutations.
    """
    match_locations = []
    target_len = len(target_crispr)
    
    # Slide a window across the entire genome string
    for i in range(len(genome_sequence) - target_len + 1):
        window = genome_sequence[i:i+target_len]
        
        # Count how many letters don't match (Hamming distance)
        mismatches = sum(1 for a, b in zip(window, target_crispr) if a != b)
        
        if mismatches <= max_mismatches:
            match_locations.append(i)
            
    return match_locations

if __name__ == "__main__":
    print("CPU baseline alignment module initialized successfully.")