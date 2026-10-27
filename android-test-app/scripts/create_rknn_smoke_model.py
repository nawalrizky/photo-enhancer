#!/usr/bin/env python3
"""Create a deterministic Conv/ReLU model to verify RK3576 runtime inference."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3576)
    weights = rng.normal(0, 0.1, (8, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(0, 0.01, 8).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "weights", "bias"], ["conv"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["conv"], ["output"])],
        "rk3576_runtime_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 16, 16])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    source = args.output_dir / "runtime_smoke.onnx"
    onnx.save(model, source)
    engine = RKNN()
    try:
        for stage, result in [
            ("config", engine.config(target_platform="rk3576")),
            ("load_onnx", engine.load_onnx(model=str(source))),
        ]:
            if result != 0:
                raise RuntimeError(f"Smoke model {stage} failed: {result}")
        if engine.build(do_quantization=False) != 0:
            raise RuntimeError("Smoke model build failed")
        if engine.export_rknn(str(args.output_dir / "runtime_smoke.rknn")) != 0:
            raise RuntimeError("Smoke model export failed")
    finally:
        engine.release()


if __name__ == "__main__":
    main()
