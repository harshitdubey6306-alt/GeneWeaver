import pytest
from src.cpu_alignment import naive_align
from src.gpu_alignment import gpu_align
from src.validators import validate_alignment_inputs, SequenceValidationError


def test_cpu_and_gpu_alignment_parity():
    """Ensures vectorized GPU/NumPy alignment matches naive CPU output identically."""
    genome = "AGCTTAGCTAGCTACGATCGATCGATCGATCGTAGCTAGCTAG"
    guide = "CGATCGATCG"

    cpu_hits = naive_align(genome, guide, max_mismatches=1)
    gpu_hits = gpu_align(genome, guide, max_mismatches=1, chunk_size=20)

    assert cpu_hits == gpu_hits
    assert len(gpu_hits) > 0


def test_exact_match_zero_mismatches():
    """Verifies exact string matching when max_mismatches=0."""
    genome = "AAAAATTTTTCCCCCGGGGG"
    guide = "TTTTTCCCCC"
    hits = gpu_align(genome, guide, max_mismatches=0)
    assert hits == [5]


def test_validator_strips_fasta_headers_and_whitespace():
    """Checks that FASTA headers and newlines are cleanly stripped before alignment."""
    raw_fasta = ">chr1_sample_region\nAGCT TAGC\nTACG ATCG"
    clean_genome, clean_target = validate_alignment_inputs(raw_fasta, "tacg atcg")

    assert clean_genome == "AGCTTAGCTACGATCG"
    assert clean_target == "TACGATCG"


def test_validator_rejects_invalid_characters():
    """Ensures non-DNA bases raise SequenceValidationError."""
    with pytest.raises(SequenceValidationError):
        validate_alignment_inputs("AGCT123TAGC", "AGCTT")


def test_validator_rejects_oversized_guide():
    """Ensures target sequences longer than the genome are rejected."""
    with pytest.raises(SequenceValidationError):
        validate_alignment_inputs("AGCT", "AGCTTAGCT")