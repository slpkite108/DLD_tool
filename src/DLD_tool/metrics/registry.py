# metrics_core/registry.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Mapping, Optional
from .base import Metric

class MetricRegistry:
    """
    name -> Metric 객체 딕셔너리.
    사용 패턴:
      reg.register(metric)
      reg["loss"].update(value=...)
      reg.compute_all()
    """
    def __init__(self):
        self._m: Dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        if metric.name in self._m:
            raise KeyError(f"Metric already registered: {metric.name}")
        self._m[metric.name] = metric

    def unregister(self, name: str) -> None:
        self._m.pop(name, None)

    def get(self, name: str) -> Metric:
        return self._m[name]

    def __getitem__(self, name: str) -> Metric:
        return self.get(name)

    def names(self) -> Iterable[str]:
        return self._m.keys()

    def items(self):
        return self._m.items()

    def reset_all(self) -> None:
        for _, m in self._m.items():
            m.reset()

    def compute_all(self) -> Dict[str, Any]:
        return {name: m.compute() for name, m in self._m.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {name: m.state_dict() for name, m in self._m.items()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, st in state.items():
            if name in self._m:
                self._m[name].load_state_dict(st)
