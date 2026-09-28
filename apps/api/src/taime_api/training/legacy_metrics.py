# Provided by Fundación Valenciaport (THEMIS port use case) for TAIME: the custom
# metric their seed TSMixer models were trained with. Kept verbatim so seeds that
# pickled it from the training script's __main__ can be loaded; do not edit.

import torch
from torchmetrics import Metric


"""Custom metric class used by the seed port model.

Se definió originalmente en el __main__ del script de entrenamiento; se
reproduce aquí tal cual para que TAIME pueda deserializar el modelo semilla.
"""

class PrimerSecuenciaUnosDistance(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        longitud_secuencia: int = 1,
        horizon: int = 48,
        from_logits: bool = True,
        threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.longitud_secuencia = longitud_secuencia
        self.horizon = horizon
        self.from_logits = from_logits
        self.threshold = threshold
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def _binarizar(self, preds: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            preds = torch.sigmoid(preds)
        return (preds >= self.threshold).long()

    def _reshape_if_needed(self, preds, target):
        if preds.ndim == 2 and preds.shape[1] == 1:
            total = preds.shape[0]
            if total % self.horizon != 0:
                raise ValueError(
                    f"No se puede reconstruir secuencias: "
                    f"{total=} no es múltiplo de horizon={self.horizon}"
                )
            batch = total // self.horizon
            preds = preds.view(batch, self.horizon)
            target = target.view(batch, self.horizon)
        return preds, target

    def _encontrar_inicio(self, valores: torch.Tensor) -> int:
        count = 0
        for i in range(valores.numel()):
            if valores[i] == 1:
                count += 1
                if count == self.longitud_secuencia:
                    return i - self.longitud_secuencia + 1
            else:
                count = 0
        # Fallback: primer '1' si no hay secuencia de longitud X
        ones = torch.nonzero(valores, as_tuple=False)
        return int(ones[0]) if ones.numel() > 0 else -1

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = self._reshape_if_needed(preds, target)
        preds = self._binarizar(preds)
        target = (target > 0).long()

        batch_size, seq_len = target.shape
        pred_seq_len = preds.shape[1]  # <-- longitud de preds por separado

        for b in range(batch_size):
            gt_inicio = self._encontrar_inicio(target[b])
            pred_inicio = self._encontrar_inicio(preds[b])

            if gt_inicio == -1 and pred_inicio == -1:
                distance = 0.0
            else:
                if gt_inicio == -1:
                    gt_inicio = seq_len       # penaliza con longitud del ground truth
                if pred_inicio == -1:
                    pred_inicio = pred_seq_len  # penaliza con longitud de la predicción
                distance = abs(gt_inicio - pred_inicio)

            self.total_distance += distance
            self.total_samples += 1

    def compute(self) -> torch.Tensor:
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.total_distance.device)
        return self.total_distance / self.total_samples

