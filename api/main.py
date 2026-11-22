from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.gpu_utils import check_hardware_status

app = FastAPI(
    title="GeneWeaver API",
    description="REST API for GPU-Accelerated CRISPR Sequence Alignment",
    version="0.1.0"
)


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
    """Validates and accepts a CRISPR alignment job payload."""
    valid_bases = {"A", "C", "G", "T"}
    if not set(request.genome_sequence.upper()).issubset(valid_bases):
        raise HTTPException(status_code=400, detail="Genome sequence must contain only A, C, G, or T.")
    if not set(request.target_crispr.upper()).issubset(valid_bases):
        raise HTTPException(status_code=400, detail="CRISPR sequence must contain only A, C, G, or T.")

    return {
        "status": "accepted",
        "genome_length": len(request.genome_sequence),
        "target_length": len(request.target_crispr),
        "max_mismatches": request.max_mismatches
    }