# metrics_core/manager.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional
import threading

from .registry import MetricRegistry
from .storage import ResultStorage
from .config import StorageConfig
from .utils import to_py_scalar

class MetricsManager:
    """
    - MetricRegistry를 가지고 있고
    - compute 결과를 ResultStorage 정책에 따라 저장(혹은 저장 안 함)
    - step 기반 snapshot/flush 지원
    - thread-safe 옵션 제공
    """
    def __init__(self, registry: Optional[MetricRegistry] = None, storage_cfg: Optional[StorageConfig] = None):
        self.registry = registry or MetricRegistry()
        self.storage = ResultStorage(storage_cfg or StorageConfig())
        self._lock = threading.Lock()
        self._step = 0

    def update(self, metric_name: str, **kwargs) -> None:
        with self._lock:
            self.registry[metric_name].update(**kwargs)

    def update_many(self, updates: Mapping[str, Dict[str, Any]]) -> None:
        """
        updates = {
          "loss": {"value": loss},
          "cm": {"y_true": y, "y_pred": p},
        }
        """
        with self._lock:
            for name, kw in updates.items():
                self.registry[name].update(**kw)

    def compute(self) -> Dict[str, Any]:
        with self._lock:
            out = self.registry.compute_all()
        return to_py_scalar(out)

    def step(self, store: bool = True) -> Dict[str, Any]:
        """
        보통 train loop에서 n step마다 호출:
          res = mm.step(store=True)
        """
        self._step += 1
        out = self.compute()
        if store:
            self.storage.add(out)
        return out

    def history(self) -> Any:
        return self.storage.get()

    def flush(self) -> None:
        self.storage.flush()

    def reset(self) -> None:
        with self._lock:
            self.registry.reset_all()
        self.storage.reset()
        self._step = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "registry": self.registry.state_dict(),
            "storage": {"cfg": self.storage.cfg.__dict__, "history": self.storage.get()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._step = int(state.get("step", 0))
        if "registry" in state:
            self.registry.load_state_dict(state["registry"])
        # storage history 복원은 모드별로 정책이 달라서 기본은 cfg만 복원 권장
