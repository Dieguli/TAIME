"""Multiprocessing pool lifecycle management."""

import os
from multiprocessing.pool import Pool
from typing import Any

# Global pool instance
_pool: Pool | None = None

# Get max workers from environment
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))


def start_pool() -> None:
    """Start the multiprocessing pool."""
    global _pool
    if os.getenv("TAIME_DISABLE_POOL") or os.getenv("PYTEST_CURRENT_TEST"):
        return
    if _pool is None:
        _pool = Pool(processes=MAX_WORKERS)


def shutdown_pool() -> None:
    """Shutdown the multiprocessing pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None


def get_pool() -> Pool | None:
    """Get the current pool instance."""
    return _pool


def submit_task(func: Any, args: tuple = ()) -> Any:
    """Submit a task to the pool."""
    if _pool is None:
        raise RuntimeError("Pool not started")
    return _pool.apply_async(func, args)
