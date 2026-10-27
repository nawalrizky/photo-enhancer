#!/usr/bin/env python3
"""Replace empty Resize ROI constants with valid, equivalent ROI constants.

The FP16 export casts empty ROI constants to float32. ONNX opset 11 requires
this input, while RKNN-Toolkit2 rejects its (0,) shape during constant folding.
For half_pixel mode, Resize ignores ROI; [0,0,0,0,1,1,1,1] is equivalent.
This only rewrites that exact pattern.
"""

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    producers = {output: node for node in model.graph.node for output in node.output}
    repaired = 0

    for resize in model.graph.node:
        if resize.op_type != "Resize" or len(resize.input) < 2:
            continue
        roi_name = resize.input[1]
        cast = producers.get(roi_name)
        if cast is None or cast.op_type != "Cast":
            continue
        constant_name = cast.input[0]
        constant = producers.get(constant_name)
        if constant is None or constant.op_type != "Constant":
            continue
        value_attr = next((attr for attr in constant.attribute if attr.name == "value"), None)
        if value_attr is None:
            continue
        value = numpy_helper.to_array(value_attr.t)
        if value.size != 0:
            continue
        mode = next((attr.s for attr in resize.attribute if attr.name == "coordinate_transformation_mode"), b"")
        if mode != b"half_pixel":
            raise RuntimeError(f"Unsafe Resize mode in {resize.name}: {mode!r}")
        value_attr.t.CopyFrom(numpy_helper.from_array(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=value.dtype)))
        repaired += 1

    if repaired == 0:
        raise RuntimeError("No exclusively consumed empty Resize ROI casts found")

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Repaired {repaired} empty Resize ROI inputs: {args.output}")


if __name__ == "__main__":
    main()
