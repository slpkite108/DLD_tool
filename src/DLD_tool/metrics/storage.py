# metrics_core/storage.py
from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from typing import Any, Deque, List, Optional, Union
from collections import deque
import json
import os

from .config import StorageConfig
from .utils import make_default_metrics_file

class ResultStorage:
    """
    metric 결과(스칼라/딕트 등)를 저장하는 정책 객체.
    메모리/디스크 옵션을 여기서 일괄 처리.
    """

    def __init__(self, cfg: StorageConfig):
        self.cfg = cfg
        self._step = 0

        self._ema: Optional[float] = None
        self._buf: Optional[Deque[Any]] = None

        self._pending_file: List[Any] = []
        if self.cfg.mode in ("last_k", "ring"):
            self._buf = deque(maxlen=self.cfg.maxlen)

        if self.cfg.mode == "file":
            if not self.cfg.file_path:
                if not self.cfg.auto_file:
                    raise ValueError("file_path required when auto_file=False")
                self.cfg = dataclasses.replace(
                    self.cfg,
                    file_path=make_default_metrics_file(self.cfg.file_prefix)
                )

            os.makedirs(os.path.dirname(self.cfg.file_path), exist_ok=True)

    def add(self, value: Any) -> None:
        self._step += 1

        # downsample 적용: 예) 5면 5 step마다 저장
        if self.cfg.downsample > 1 and (self._step % self.cfg.downsample != 0):
            # 저장은 생략하지만 EMA update 같은건 하고싶다면 아래에서 처리 가능
            pass

        if self.cfg.mode == "none":
            return

        if self.cfg.mode == "ema":
            # value가 dict면 사용자가 직접 metric에서 스칼라로 넣는 걸 권장
            v = float(value)
            if self._ema is None:
                self._ema = v
            else:
                a = self.cfg.ema_alpha
                self._ema = a * v + (1 - a) * self._ema
            return

        if self.cfg.mode in ("last_k", "ring"):
            assert self._buf is not None
            if self.cfg.downsample == 1 or (self._step % self.cfg.downsample == 0):
                self._buf.append(value)
            return

        if self.cfg.mode == "file":
            if self.cfg.downsample == 1 or (self._step % self.cfg.downsample == 0):
                self._pending_file.append({"step": self._step, "value": value})
            if len(self._pending_file) >= self.cfg.flush_every:
                self.flush()
            return

        raise ValueError(f"Unknown storage mode: {self.cfg.mode}")

    def flush(self) -> None:
        if self.cfg.mode != "file":
            return
        if not self._pending_file:
            return
        assert self.cfg.file_path is not None
        with open(self.cfg.file_path, "a", encoding="utf-8") as f:
            for row in self._pending_file:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._pending_file.clear()

    def get(self) -> Any:
        if self.cfg.mode == "none":
            return None
        if self.cfg.mode == "ema":
            return self._ema
        if self.cfg.mode in ("last_k", "ring"):
            return list(self._buf) if self._buf is not None else []
        if self.cfg.mode == "file":
            # file 모드는 히스토리를 RAM에서 들고 있지 않음
            return {"file_path": self.cfg.file_path}
        return None

    def reset(self) -> None:
        self._step = 0
        self._ema = None
        if self._buf is not None:
            self._buf.clear()
        self._pending_file.clear()
