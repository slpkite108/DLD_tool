# metrics_core/utils.py
from __future__ import annotations
from typing import Any, Dict
import math
from pathlib import Path
import os
from datetime import datetime

def to_py_scalar(x: Any) -> Any:
    """
    PyTorch/NumPy 스칼라 등을 파이썬 기본 타입으로 변환.
    GPU 텐서/그래프 참조를 끊어서 메모리 누수 방지.
    """
    # torch 텐서 체크 (torch 의존성을 강제하지 않기 위해 try)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            # 그래프 끊고 CPU로 이동 후 스칼라/리스트 변환
            x = x.detach()
            if x.numel() == 1:
                return float(x.cpu().item())
            return x.cpu().tolist()
    except Exception:
        pass

    # numpy 스칼라/배열
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.reshape(-1)[0])
            return x.tolist()
        if isinstance(x, np.generic):
            return float(x)
    except Exception:
        pass

    # 파이썬 기본
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x

    # dict/list 재귀 변환
    if isinstance(x, dict):
        return {k: to_py_scalar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py_scalar(v) for v in x]

    # fallback: 그대로 (단, 참조 큰 객체는 피하는 게 좋음)
    return x

def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)

def is_finite(x: float) -> bool:
    return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))

def get_metrics_cache_dir() -> Path:
    """
    OS별 표준 cache 위치 사용.
    기본: ~/.cache/DLD_tool/metrics
    환경변수 DLD_TOOL_CACHE 로 override 가능.
    """
    base = os.environ.get("DLD_TOOL_CACHE", None)

    if base is None:
        base = Path.home() / ".cache" / "DLD_tool"
    else:
        base = Path(base)

    path = base / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_default_metrics_file(prefix: str = "metrics") -> str:
    """
    자동 run_id 포함 파일명 생성
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(get_metrics_cache_dir() / f"{prefix}_{ts}.jsonl")