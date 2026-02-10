if __name__ == "__main__":
    """
    실행:
      PYTHONPATH=src python -m DLD_tool.metrics.base
    """

    import random
    import tempfile
    import os

    from DLD_tool.metrics import (
        MetricRegistry,
        MetricsManager,
        StorageConfig,
        RunningMean,
        ConfusionMatrix,
        MacroF1FromCM,
    )

    print("=== Metrics System Smoke Test Start ===")

    NUM_STEPS = 120
    BATCH = 32
    NUM_CLASSES = 3

    def make_batch(step: int):
        # loss: 감소 추세 + 노이즈
        loss = 3.0 * (0.98 ** step) + random.random() * 0.1
        y_true = [random.randint(0, NUM_CLASSES - 1) for _ in range(BATCH)]
        y_pred = [
            t if random.random() > 0.25 else random.randint(0, NUM_CLASSES - 1)
            for t in y_true
        ]
        return loss, y_true, y_pred

    # ------------------------------------------------------------
    # A) dict 결과 저장: last_k (step(store=True)로 전체 파이프 검증)
    # ------------------------------------------------------------
    print("\n[A] Dict-output + last_k storage test")

    reg = MetricRegistry()
    reg.register(RunningMean("loss"))
    cm = ConfusionMatrix("cm", num_classes=NUM_CLASSES, ignore_index=None)
    reg.register(cm)
    reg.register(MacroF1FromCM("macro_f1", cm_metric=cm))

    mm = MetricsManager(registry=reg, storage_cfg=StorageConfig(mode="last_k", maxlen=10))

    for step in range(NUM_STEPS):
        loss, y_true, y_pred = make_batch(step)
        mm.update("loss", value=loss)
        mm.update("cm", y_true=y_true, y_pred=y_pred)

        if step % 20 == 0:
            out = mm.step(store=True)  # dict 저장
            # 최소 sanity check
            assert "loss" in out and "macro_f1" in out and "cm" in out
            assert isinstance(out["loss"], (int, float))
            assert "macro_f1" in out["macro_f1"]
            print(f"  step {step:03d} | loss={out['loss']:.4f} | macro_f1={out['macro_f1']['macro_f1']:.4f}")

    hist = mm.history()
    assert isinstance(hist, list) and len(hist) <= 10
    print("  history len:", len(hist))

    # state_dict / load_state_dict 점검
    st = mm.state_dict()
    mm2 = MetricsManager(registry=reg, storage_cfg=StorageConfig(mode="last_k", maxlen=10))
    mm2.load_state_dict(st)  # registry state 로드 확인(기본 구현 범위)
    print("  state_dict/load_state_dict: OK (basic)")

    # ------------------------------------------------------------
    # B) file storage: dict 결과를 jsonl로 flush되는지 점검
    # ------------------------------------------------------------
    print("\n[B] Dict-output + file storage test")

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "metrics.jsonl")
        mm_file = MetricsManager(
            registry=reg,
            storage_cfg=StorageConfig(mode="file", file_path=fp, flush_every=5, downsample=1),
        )

        for step in range(25):
            loss, y_true, y_pred = make_batch(step)
            mm_file.update("loss", value=loss)
            mm_file.update("cm", y_true=y_true, y_pred=y_pred)
            mm_file.step(store=True)

        mm_file.flush()
        assert os.path.exists(fp)
        size = os.path.getsize(fp)
        assert size > 0
        print("  file exists and non-empty:", fp, "| bytes:", size)
        print("  history():", mm_file.history())

    # ------------------------------------------------------------
    # C) EMA storage는 스칼라 전용임을 점검
    #   - C1: dict-output + EMA는 "예상대로 실패"하는지
    #   - C2: scalar-only manager를 만들어 EMA가 정상 동작하는지
    # ------------------------------------------------------------
    print("\n[C] EMA storage test (expected failure for dict-output, success for scalar-only)")

    # C1) dict-output + EMA => 현재 storage 구현상 TypeError가 나는 게 정상
    mm_bad = MetricsManager(registry=reg, storage_cfg=StorageConfig(mode="ema", ema_alpha=0.2))
    loss, y_true, y_pred = make_batch(0)
    mm_bad.update("loss", value=loss)
    mm_bad.update("cm", y_true=y_true, y_pred=y_pred)

    try:
        mm_bad.step(store=True)  # dict 저장 시도 -> TypeError 기대
        raise AssertionError("EMA should not accept dict output, but it did not fail.")
    except TypeError:
        print("  C1) OK: dict-output + EMA raises TypeError (as expected)")

    # C2) scalar-only metric set에서 EMA 정상 동작
    reg_scalar = MetricRegistry()
    reg_scalar.register(RunningMean("loss"))  # compute_all => {"loss": float} 가 아니라 dict이긴 함
    # 주의: compute_all은 dict 반환이라서 그대로 EMA에 넣으면 또 실패.
    # 그래서 scalar-only에서는 manager가 저장하는 값이 "float"가 되도록
    # loss metric만 compute해서 storage에 넣는 방식으로 테스트한다(테스트 코드 범위 내).

    mm_ema = MetricsManager(registry=reg_scalar, storage_cfg=StorageConfig(mode="ema", ema_alpha=0.2))

    for step in range(50):
        loss, _, _ = make_batch(step)
        mm_ema.update("loss", value=loss)

        # manager.step(store=True)는 dict를 storage에 넣으므로 피하고,
        # step(store=False)로 compute만 한 뒤 loss 스칼라만 storage에 넣어 EMA 경로 점검
        out = mm_ema.step(store=False)  # {"loss": float}
        mm_ema.storage.add(out["loss"])  # ✅ 스칼라 EMA 정상 동작해야 함

    ema_val = mm_ema.history()
    assert isinstance(ema_val, (int, float)) or ema_val is None
    print("  C2) OK: scalar EMA value:", ema_val)

    print("\n=== Metrics System Smoke Test Finished ===")
