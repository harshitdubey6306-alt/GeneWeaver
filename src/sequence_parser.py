def parse_fasta(file_path):
    """
    Reads a standard FASTA genomic file and returns the pure DNA sequence.
    Ignores metadata header lines starting with '>'.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Combine lines into a single string, skipping the header
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
        return sequence
        
    except FileNotFoundError:
        return "Error: Genomic data file not found. Please check the path."

if __name__ == "__main__":
    print("Sequence parser module initialized successfully.")