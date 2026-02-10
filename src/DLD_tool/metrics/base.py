# metrics_core/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

class Metric(ABC):
    """
    독립적으로 작동하는 Metric 객체.
    - update(**kwargs): 누적/상태 업데이트
    - compute(): 현재 값 계산
    - reset(): 상태 초기화
    - state_dict()/load_state_dict(): 체크포인트 지원
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def update(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # 기본 구현: name만
        if "name" in state:
            self.name = str(state["name"])