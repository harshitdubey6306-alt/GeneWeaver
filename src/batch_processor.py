import time
from concurrent.futures import ThreadPoolExecutor
from src.gpu_alignment import gpu_align
from src.gpu_utils import check_hardware_status
from src.database import log_alignment_run
from src.validators import validate_alignment_inputs, SequenceValidationError


def process_single_guide(
    genome: str,
    target_crispr: str,
    max_mismatches: int = 1,
    allow_ambiguous: bool = False,
    log_to_db: bool = True,
) -> dict:
    """Validates and aligns a single CRISPR guide against a sanitized genome."""
    try:
        clean_genome, clean_target = validate_alignment_inputs(
            genome_sequence=genome,
            target_crispr=target_crispr,
            max_mismatches=max_mismatches,
            allow_ambiguous=allow_ambiguous,
        )
    except SequenceValidationError as exc:
        return {
            "target_crispr": target_crispr,
            "status": "failed",
            "error": str(exc),
            "match_count": 0,
            "match_indices": [],
        }

    hardware = check_hardware_status()
    start_time = time.perf_counter()
    matches = gpu_align(clean_genome, clean_target, max_mismatches=max_mismatches)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    run_id = None
    if log_to_db:
        run_id = log_alignment_run(
            target_crispr=clean_target,
            genome_length=len(clean_genome),
            max_mismatches=max_mismatches,
            backend=hardware["backend"],
            execution_ms=elapsed_ms,
            match_indices=matches,
        )

    return {
        "run_id": run_id,
        "target_crispr": clean_target,
        "status": "completed",
        "execution_ms": round(elapsed_ms, 4),
        "match_count": len(matches),
        "match_indices": matches,
    }


def run_batch_alignment(
    genome_sequence: str,
    target_crisprs: list[str],
    max_mismatches: int = 1,
    allow_ambiguous: bool = False,
    max_workers: int = 4,
    log_to_db: bool = True,
) -> dict:
    """
    Executes multiple CRISPR guide alignments concurrently against a single genome.
    Returns aggregated timing metrics and per-guide alignment results.
    """
    if not target_crisprs:
        raise SequenceValidationError("target_crisprs list cannot be empty.")

    batch_start = time.perf_counter()
    hardware = check_hardware_status()

    with ThreadPoolExecutor(max_workers=min(max_workers, len(target_crisprs))) as executor:
        futures = [
            executor.submit(
                process_single_guide,
                genome_sequence,
                guide,
                max_mismatches,
                allow_ambiguous,
                log_to_db,
            )
            for guide in target_crisprs
        ]
        results = [future.result() for future in futures]

    total_ms = (time.perf_counter() - batch_start) * 1000
    completed = sum(1 for r in results if r["status"] == "completed")

    return {
        "batch_status": "completed",
        "backend": hardware["backend"],
        "total_guides": len(target_crisprs),
        "completed_guides": completed,
        "failed_guides": len(target_crisprs) - completed,
        "total_execution_ms": round(total_ms, 4),
        "results": results,
    }