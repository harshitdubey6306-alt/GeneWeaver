import re

STRICT_DNA_BASES = {"A", "C", "G", "T"}
IUPAC_AMBIGUOUS_BASES = {"A", "C", "G", "T", "N", "R", "Y", "S", "W", "K", "M"}


class SequenceValidationError(ValueError):
    """Raised when a genomic or CRISPR sequence fails integrity checks."""


def sanitize_sequence(raw_sequence: str) -> str:
    """Strips FASTA header lines if present, removes whitespace, and uppercases."""
    if not isinstance(raw_sequence, str) or not raw_sequence.strip():
        raise SequenceValidationError("Sequence input must be a non-empty string.")

    lines = [
        line.strip()
        for line in raw_sequence.splitlines()
        if line.strip() and not line.strip().startswith(">")
    ]
    cleaned = re.sub(r"\s+", "", "".join(lines)).upper()

    if not cleaned:
        raise SequenceValidationError("No nucleotide bases found after stripping headers/whitespace.")
    return cleaned


def validate_alignment_inputs(
    genome_sequence: str,
    target_crispr: str,
    max_mismatches: int = 1,
    allow_ambiguous: bool = False
) -> tuple[str, str]:
    """
    Validates genome and CRISPR guide sequences for character validity,
    length boundaries, and mismatch thresholds. Returns sanitized sequences.
    """
    genome = sanitize_sequence(genome_sequence)
    target = sanitize_sequence(target_crispr)

    allowed_chars = IUPAC_AMBIGUOUS_BASES if allow_ambiguous else STRICT_DNA_BASES

    invalid_genome = set(genome) - allowed_chars
    if invalid_genome:
        raise SequenceValidationError(
            f"Genome contains invalid characters: {sorted(invalid_genome)}"
        )

    invalid_target = set(target) - allowed_chars
    if invalid_target:
        raise SequenceValidationError(
            f"CRISPR guide contains invalid characters: {sorted(invalid_target)}"
        )

    if len(target) < 5:
        raise SequenceValidationError("CRISPR guide sequence must be at least 5 bases long.")

    if len(target) > len(genome):
        raise SequenceValidationError(
            f"Target length ({len(target)}) cannot exceed genome length ({len(genome)})."
        )

    if max_mismatches < 0 or max_mismatches >= len(target):
        raise SequenceValidationError(
            f"max_mismatches ({max_mismatches}) must be between 0 and {len(target) - 1}."
        )

    return genome, target