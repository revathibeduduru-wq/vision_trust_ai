"""TrustLens Backend Main Application

Entrypoint for FastAPI application with dataset integrity, model provenance,
and risk analytics modules.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints.dataset_analysis import router as dataset_router

app = FastAPI(
    title="TrustLens API",
    description="Offline-First AI/ML Provenance and Dataset Integrity Assurance Platform (SIH26228)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local and hosted frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Member 2: CV / Dataset Integrity Endpoints
app.include_router(dataset_router, prefix="/api/datasets", tags=["Dataset Integrity"])


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "TrustLens Platform",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", tags=["General"])
async def health():
    return {"status": "healthy", "offline_mode": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
