import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.gpu_utils import check_hardware_status
from src.gpu_alignment import gpu_align
from src.database import init_db, log_alignment_run, fetch_recent_runs
from src.validators import validate_alignment_inputs, SequenceValidationError
from src.batch_processor import run_batch_alignment

app = FastAPI(
    title="GeneWeaver API",
    description="REST API for GPU-Accelerated CRISPR Sequence Alignment",
    version="0.4.0"
)


@app.on_event("startup")
def startup_event():
    """Ensures the SQLite database table is initialized when the API starts."""
    init_db()


class AlignmentRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=1, description="Target DNA genome or FASTA text")
    target_crispr: str = Field(..., min_length=5, description="CRISPR guide sequence to search for")
    max_mismatches: int = Field(default=1, ge=0, le=5, description="Maximum allowed base mismatches")
    allow_ambiguous: bool = Field(default=False, description="Allow IUPAC ambiguous bases like N")


class BatchAlignmentRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=1, description="Target DNA genome or FASTA text")
    target_crisprs: list[str] = Field(..., min_length=1, max_length=50, description="List of CRISPR guides")
    max_mismatches: int = Field(default=1, ge=0, le=5, description="Maximum allowed base mismatches")
    allow_ambiguous: bool = Field(default=False, description="Allow IUPAC ambiguous bases like N")


@app.get("/health")
def health_check():
    """Returns API health status and active compute backend (GPU or CPU)."""
    return {
        "status": "online",
        "hardware": check_hardware_status()
    }


@app.post("/api/v1/align")
def submit_alignment(request: AlignmentRequest):
    """Validates genomic inputs, runs parallel alignment, logs metrics, and returns matches."""
    try:
        genome, target = validate_alignment_inputs(
            genome_sequence=request.genome_sequence,
            target_crispr=request.target_crispr,
            max_mismatches=request.max_mismatches,
            allow_ambiguous=request.allow_ambiguous
        )
    except SequenceValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    hardware = check_hardware_status()
    start_time = time.perf_counter()
    matches = gpu_align(genome, target, max_mismatches=request.max_mismatches)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    run_id = log_alignment_run(
        target_crispr=target,
        genome_length=len(genome),
        max_mismatches=request.max_mismatches,
        backend=hardware["backend"],
        execution_ms=elapsed_ms,
        match_indices=matches
    )

    return {
        "run_id": run_id,
        "status": "completed",
        "backend": hardware["backend"],
        "execution_ms": round(elapsed_ms, 4),
        "match_count": len(matches),
        "match_indices": matches
    }


@app.post("/api/v1/align/batch")
def submit_batch_alignment(request: BatchAlignmentRequest):
    """Executes concurrent alignment across multiple CRISPR guide sequences."""
    try:
        return run_batch_alignment(
            genome_sequence=request.genome_sequence,
            target_crisprs=request.target_crisprs,
            max_mismatches=request.max_mismatches,
            allow_ambiguous=request.allow_ambiguous,
        )
    except SequenceValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/v1/history")
def get_alignment_history(limit: int = 10):
    """Fetches recent CRISPR alignment runs and timing metrics from the database."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    return {"runs": fetch_recent_runs(limit=limit)}