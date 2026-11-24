import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.gpu_utils import check_hardware_status
from src.gpu_alignment import gpu_align
from src.database import init_db, log_alignment_run, fetch_recent_runs

app = FastAPI(
    title="GeneWeaver API",
    description="REST API for GPU-Accelerated CRISPR Sequence Alignment",
    version="0.2.0"
)


@app.on_event("startup")
def startup_event():
    """Ensures the SQLite database table is initialized when the API starts."""
    init_db()


class AlignmentRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=1, description="Target DNA genome string (A, C, G, T)")
    target_crispr: str = Field(..., min_length=1, description="Short CRISPR guide sequence to search for")
    max_mismatches: int = Field(default=1, ge=0, le=5, description="Maximum allowed base mismatches")


@app.get("/health")
def health_check():
    """Returns API health status and active compute backend (GPU or CPU)."""
    return {
        "status": "online",
        "hardware": check_hardware_status()
    }


@app.post("/api/v1/align")
def submit_alignment(request: AlignmentRequest):
    """Runs parallel CRISPR alignment, logs metrics to SQLite, and returns matches."""
    valid_bases = {"A", "C", "G", "T"}
    genome = request.genome_sequence.strip().upper()
    target = request.target_crispr.strip().upper()

    if not set(genome).issubset(valid_bases):
        raise HTTPException(status_code=400, detail="Genome sequence must contain only A, C, G, or T.")
    if not set(target).issubset(valid_bases):
        raise HTTPException(status_code=400, detail="CRISPR sequence must contain only A, C, G, or T.")
    if len(target) > len(genome):
        raise HTTPException(status_code=400, detail="Target CRISPR sequence cannot be longer than genome.")

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


@app.get("/api/v1/history")
def get_alignment_history(limit: int = 10):
    """Fetches recent CRISPR alignment runs and timing metrics from the database."""
    return {"runs": fetch_recent_runs(limit=limit)}