"""
examples/iteration_epoch_style.py

실행:
  PYTHONPATH=src python examples/iteration_epoch_style.py

목표:
  - epoch/iteration 구조에서 metrics를 어떻게 업데이트/저장/요약하는지 예시 제공
  - ConfusionMatrix는 torch.Tensor 입력이면 더 빠른 경로(예: bincount)를 타도록 설계되었다는 가정
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch

from DLD_tool.metrics.registry import MetricRegistry
from DLD_tool.metrics.manager import MetricsManager
from DLD_tool.metrics.config import StorageConfig
from DLD_tool.metrics.impl.running import RunningMean
from DLD_tool.metrics.impl.classification import ConfusionMatrix, MacroF1FromCM


# -----------------------------
# Dummy data generator (example)
# -----------------------------
@dataclass
class LoopCfg:
    num_epochs: int = 3
    iters_per_epoch: int = 120
    batch_size: int = 4096          # 프레임 수가 많다는 가정
    num_classes: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    log_every_iter: int = 20        # iter 로그 저장 주기
    do_val: bool = True
    val_iters: int = 30


def make_dummy_batch(cfg: LoopCfg, global_step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    더미 분류 라벨 생성:
      y_true, y_pred: (B,) int64
      loss: scalar float tensor
    """
    # loss: 점진적 감소 + 노이즈
    loss_val = 3.0 * (0.98 ** global_step) + random.random() * 0.1
    loss = torch.tensor(loss_val, device=cfg.device, dtype=torch.float32)

    y_true = torch.randint(0, cfg.num_classes, (cfg.batch_size,), device=cfg.device, dtype=torch.int64)

    # 정답 확률 75% 정도로 예측 생성
    noise_mask = torch.rand((cfg.batch_size,), device=cfg.device) < 0.25
    y_pred = y_true.clone()
    y_pred[noise_mask] = torch.randint(0, cfg.num_classes, (int(noise_mask.sum().item()),), device=cfg.device, dtype=torch.int64)

    return loss, y_true, y_pred


# -----------------------------
# Build metrics manager
# -----------------------------
def build_manager(cfg: LoopCfg, file_path: str | None = None, mode: str = "last_k") -> MetricsManager:
    """
    mode:
      - "last_k": 최근 K개만 메모리에 유지(디버깅/노트북용)
      - "file": jsonl 파일로 저장(장기 학습용)
      - "none": 저장 안함(가장 가벼움)
    """
    reg = MetricRegistry()
    reg.register(RunningMean("loss"))

    cm = ConfusionMatrix("cm", num_classes=cfg.num_classes, ignore_index=None)
    reg.register(cm)
    reg.register(MacroF1FromCM("macro_f1", cm_metric=cm))

    if mode == "file":
        #assert file_path is not None
        #disable on example
        storage_cfg = StorageConfig(
            mode="file",
            file_path=file_path,
            flush_every=10,   # 예시는 짧으니 자주 flush
            downsample=1,
        )
    elif mode == "none":
        storage_cfg = StorageConfig(mode="none")
    else:
        # last_k/ring 등
        storage_cfg = StorageConfig(mode="last_k", maxlen=50, downsample=1)

    return MetricsManager(registry=reg, storage_cfg=storage_cfg)


# -----------------------------
# Train / Val loop example
# -----------------------------
def train_one_epoch(mm: MetricsManager, cfg: LoopCfg, epoch: int, global_step: int) -> int:
    """
    train:
      - epoch 시작에 reset하지 않음(원하면 reset 가능)
      - iter마다 update
      - log_every_iter마다 step(store=True)로 기록 저장
      - epoch 끝에서 compute()로 요약 출력
    """
    mm.reset()  # ✅ epoch 단위 지표를 보고 싶으면 reset() (train도 epoch별로 보는 게 일반적)
    mm.flush()  # file 모드면 이전 버퍼 flush (안 해도 됨)

    for it in range(cfg.iters_per_epoch):
        loss, y_true, y_pred = make_dummy_batch(cfg, global_step)

        # 텐서 입력 그대로 update (ConfusionMatrix가 텐서 fast-path를 타도록)
        mm.update("loss", value=loss)
        mm.update("cm", y_true=y_true, y_pred=y_pred)

        # iter 로그 저장(파일/last_k 등)
        if (it + 1) % cfg.log_every_iter == 0:
            out = mm.step(store=True)  # dict 저장 가능(last_k/file)
            print(
                f"[train] epoch={epoch:02d} it={it+1:04d} "
                f"loss={out['loss']:.4f} macro_f1={out['macro_f1']['macro_f1']:.4f}"
            )

        global_step += 1

    # epoch summary (전체 누적 CM 기반)
    epoch_out = mm.compute()
    print(
        f"[train][epoch_end] epoch={epoch:02d} "
        f"loss={epoch_out['loss']:.4f} macro_f1={epoch_out['macro_f1']['macro_f1']:.4f}"
    )

    # file 모드면 안전하게 flush
    mm.flush()
    return global_step


@torch.no_grad()
def validate(mm: MetricsManager, cfg: LoopCfg, epoch: int) -> None:
    """
    val:
      - 일반적으로 'epoch 시작에 reset'해서 epoch 단위 val 지표를 깔끔하게 측정
      - iter 로그는 선택(여기선 생략하고 epoch 끝 요약만)
    """
    mm.reset()

    for it in range(cfg.val_iters):
        loss, y_true, y_pred = make_dummy_batch(cfg, global_step=0)

        mm.update("loss", value=loss)
        mm.update("cm", y_true=y_true, y_pred=y_pred)

    out = mm.compute()
    print(
        f"[val][epoch_end] epoch={epoch:02d} "
        f"loss={out['loss']:.4f} macro_f1={out['macro_f1']['macro_f1']:.4f}"
    )
    mm.flush()


def main():
    cfg = LoopCfg()

    # 저장 모드 선택:
    # - 장기 학습이면 file 추천
    # - 빠른 테스트면 last_k
    save_mode = "file"  # "last_k" | "file" | "none"

    # file 모드면 저장 위치 지정
    file_path = None
    # if save_mode == "file":
    #     os.makedirs("logs", exist_ok=True)
    #     file_path = "logs/metrics_train.jsonl"

    # train/val 각각 manager를 분리하면 reset 정책이 명확해짐
    mm_train = build_manager(cfg, file_path=file_path, mode=save_mode)
    mm_val = build_manager(cfg, file_path=None, mode="none")  # val은 저장 생략(필요하면 file/last_k로 변경)

    global_step = 0
    print("=== Metrics System Smoke Test Start (iteration/epoch style) ===")

    for epoch in range(cfg.num_epochs):
        global_step = train_one_epoch(mm_train, cfg, epoch=epoch, global_step=global_step)

        if cfg.do_val:
            validate(mm_val, cfg, epoch=epoch)

    # file 모드면 마지막 위치 출력
    if save_mode == "file":
        print("\nSaved metrics file:")
        print(mm_train.history())  # {'file_path': '...'} 형태

    print("=== Finished ===")


if __name__ == "__main__":
    main()
