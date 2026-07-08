"""Training cancellation helpers."""


class TrainingCancelled(RuntimeError):
    """Raised when a training job is cancelled."""

    def __init__(self, message: str = "Training cancelled") -> None:
        super().__init__(message)
