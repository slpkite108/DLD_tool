# metrics_core/impl/running.py
from __future__ import annotations
from typing import Any, Dict
from ..base import Metric
from ..utils import to_py_scalar

class RunningMean(Metric):
    """
    값들을 저장하지 않고(sum, count)만 유지 => 메모리 최소.
    update(value=...)
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.reset()

    def update(self, **kwargs) -> None:
        v = to_py_scalar(kwargs["value"])
        v = float(v)
        self._sum += v
        self._count += 1

    def compute(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "sum": self._sum, "count": self._count}

    def load_state_dict(self, state) -> None:
        self.name = state.get("name", self.name)
        self._sum = float(state["sum"])
        self._count = int(state["count"])


class WelfordMeanVar(Metric):
    """
    분산/표준편차까지 streaming으로 계산(값 저장 없음).
    update(value=...)
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.reset()

    def update(self, **kwargs) -> None:
        x = float(to_py_scalar(kwargs["value"]))
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def compute(self) -> Dict[str, float]:
        if self.n < 2:
            return {"mean": self.mean, "var": 0.0, "std": 0.0, "n": float(self.n)}
        var = self.M2 / (self.n - 1)
        return {"mean": self.mean, "var": var, "std": var ** 0.5, "n": float(self.n)}

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "n": self.n, "mean": self.mean, "M2": self.M2}

    def load_state_dict(self, state) -> None:
        self.name = state.get("name", self.name)
        self.n = int(state["n"])
        self.mean = float(state["mean"])
        self.M2 = float(state["M2"])
