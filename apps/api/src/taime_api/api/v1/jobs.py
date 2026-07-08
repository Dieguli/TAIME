"""Training jobs API endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlmodel import select

from taime_api.api.v1.schemas import JobCreate, JobResponse, JobStatus, SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import Job
from taime_api.services.jobs import JobService

router = APIRouter()


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    sut_type: SUTType | None = None,
    status: JobStatus | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[Job]:
    """List all training jobs, optionally filtered."""
    with get_session() as session:
        query = select(Job)
        if sut_type:
            query = query.where(Job.sut_type == sut_type)
        if status:
            query = query.where(Job.status == status)
        query = query.order_by(Job.created_at.desc()).offset(skip).limit(limit)
        jobs = session.exec(query).all()
        return list(jobs)


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    job_create: JobCreate,
    background_tasks: BackgroundTasks,
) -> Job:
    """Create and start a new training job."""
    service = JobService()
    job = await service.create_job(
        dataset_id=job_create.dataset_id,
        sut_type=job_create.sut_type,
        config=job_create.config,
    )

    # Schedule job execution in background
    background_tasks.add_task(service.execute_job, job.id)

    return job


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int) -> Job:
    """Get a specific job by ID with current status."""
    with get_session() as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: int) -> Job:
    """Cancel a running job."""
    service = JobService()
    job = await service.cancel_job(job_id)
    return job
