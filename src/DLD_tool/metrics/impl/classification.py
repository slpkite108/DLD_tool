# metrics/impl/classification.py
from __future__ import annotations
from typing import Any, Dict, Optional
from ..base import Metric
from ..utils import to_py_scalar

class ConfusionMatrix(Metric):
    """
    Streaming Confusion Matrix (Tensor optimized)

    지원 입력:
      - torch.Tensor (GPU/CPU)  → 고속 경로
      - numpy / list            → 기존 경로
    """

    def __init__(self, name: str, num_classes: int, ignore_index: Optional[int] = None):
        super().__init__(name)
        self.C = int(num_classes)
        self.ignore_index = ignore_index
        self.reset()

    # -------------------------------------------------
    # 핵심: tensor fast path
    # -------------------------------------------------
    def _update_tensor(self, y_true, y_pred) -> None:
        import torch

        # flatten
        y_true = y_true.view(-1).to(torch.int64)
        y_pred = y_pred.view(-1).to(torch.int64)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        # 유효 범위 마스크
        mask = (y_true >= 0) & (y_true < self.C)
        mask &= (y_pred >= 0) & (y_pred < self.C)

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # GPU bincount 핵심 ⭐
        idx = y_true * self.C + y_pred
        bincount = torch.bincount(idx, minlength=self.C * self.C)

        cm_batch = bincount.reshape(self.C, self.C)

        # 내부 CM은 CPU int로 유지 (메모리 안정)
        self.cm += cm_batch.cpu().to(self.cm.dtype)

    # -------------------------------------------------
    # 기존 python fallback 경로
    # -------------------------------------------------
    def _update_python(self, y_true, y_pred) -> None:
        y_true = to_py_scalar(y_true)
        y_pred = to_py_scalar(y_pred)

        if not isinstance(y_true, list): y_true = [y_true]
        if not isinstance(y_pred, list): y_pred = [y_pred]

        for t, p in zip(y_true, y_pred):
            t = int(t); p = int(p)
            if self.ignore_index is not None and t == self.ignore_index:
                continue
            if 0 <= t < self.C and 0 <= p < self.C:
                self.cm[t][p] += 1

    # -------------------------------------------------
    # public update
    # -------------------------------------------------
    def update(self, **kwargs) -> None:
        y_true = kwargs["y_true"]
        y_pred = kwargs["y_pred"]

        # torch tensor fast path
        try:
            import torch
            if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
                self._update_tensor(y_true, y_pred)
                return
        except Exception:
            pass

        # fallback
        self._update_python(y_true, y_pred)

    def compute(self) -> Dict[str, Any]:
        return {"cm": self.cm.tolist()}

    def reset(self) -> None:
        import torch
        self.cm = torch.zeros((self.C, self.C), dtype=torch.int64)

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "C": self.C, "ignore_index": self.ignore_index, "cm": self.cm}

    def load_state_dict(self, state) -> None:
        import torch
        self.name = state.get("name", self.name)
        self.C = int(state["C"])
        self.ignore_index = state.get("ignore_index", None)
        self.cm = torch.as_tensor(state["cm"], dtype=torch.int64)

class MacroF1FromCM(Metric):
    """
    ConfusionMatrix를 참조해 Macro-F1 계산.
    ConfusionMatrix가 torch.Tensor(cm)을 쓰는 구현과 호환.

    사용:
      cm = ConfusionMatrix("cm", num_classes=C)
      macro = MacroF1FromCM("macro_f1", cm_metric=cm)
    """

    def __init__(self, name: str, cm_metric: ConfusionMatrix):
        super().__init__(name)
        self.cm_metric = cm_metric

    def update(self, **kwargs) -> None:
        # 보통 cm_metric이 update를 받으므로 여기서는 noop
        # 원하면 여기서 self.cm_metric.update(**kwargs)로 위임 가능
        return

    def compute(self) -> Dict[str, Any]:
        import torch

        cm = self.cm_metric.cm
        if isinstance(cm, list):
            cm = torch.tensor(cm, dtype=torch.int64)

        cm = cm.to(torch.float32)
        C = int(self.cm_metric.C)

        f1s = []
        for k in range(C):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp

            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = (2 * prec * rec) / (prec + rec + 1e-12)

            f1s.append(float(f1.item()))

        macro = sum(f1s) / C if C > 0 else 0.0
        return {"macro_f1": macro, "per_class_f1": f1s}

    def reset(self) -> None:
        # cm_metric.reset()는 외부 정책에 따름
        return
