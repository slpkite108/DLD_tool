# metrics_core/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

HistoryMode = Literal["none", "last_k", "ring", "ema", "file"]

@dataclass(frozen=True)
class StorageConfig:
    """
    메모리 회피 옵션:
      - none: 히스토리 저장 안 함 (최소 메모리)
      - last_k: 최근 K개만 저장
      - ring: 고정 길이 링 버퍼
      - ema: 지수이동평균만 저장(히스토리 없음)
      - file: 일정 주기로 디스크에 append 저장 (RAM 최소화)
    """
    mode: HistoryMode = "none"
    maxlen: int = 200          # last_k/ring에서 사용
    ema_alpha: float = 0.1     # ema에서 사용
    flush_every: int = 50      # file에서 사용 (n step마다 flush)
    file_path: Optional[str] = None  # file 모드에서 필요
    auto_file: bool = True
    file_prefix: str = "metrics"
    downsample: int = 1        # 히스토리 저장 시 step downsample (1=모두 저장)
    
