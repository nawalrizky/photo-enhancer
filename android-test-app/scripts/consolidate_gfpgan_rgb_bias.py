#!/usr/bin/env python3
"""Move GFPGAN's linear RGB skip-branch biases into one final Add.

This is a graph rewrite, not an empirical output color correction. Only RGB
branches whose remaining consumers are Add or constant-preserving Resize are
eligible. Validate the rewritten model against the original before RKNN export.
"""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    model = onnx.load(args.input)
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError('Expected one model output')
    biases = {v.name: numpy_helper.to_array(v) for v in graph.initializer
              if 'to_rgb' in v.name and v.name.endswith('dotbias')}
    if len(biases) != 8 or any(v.shape != (1, 3, 1, 1) for v in biases.values()):
        raise ValueError('Expected eight RGB biases shaped [1,3,1,1]')
    consumers = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    replacements = {}
    removed = []
    for name in biases:
        uses = consumers.get(name, [])
        if len(uses) != 1 or uses[0].op_type != 'Add' or len(uses[0].input) != 2:
            raise ValueError(f'Unexpected bias consumer: {name}')
        node = uses[0]
        replacements[node.output[0]] = next(v for v in node.input if v != name)
        removed.append(node.name)
        pending = list(node.output)
        seen = set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            for following in consumers.get(value, []):
                if following.op_type not in ('Add', 'Resize'):
                    raise ValueError(f'Nonlinear RGB consumer: {following.name}')
                if following.op_type == 'Resize':
                    attrs = {a.name: helper.get_attribute_value(a) for a in following.attribute}
                    if following.input[0] != value or attrs.get('mode', b'nearest') != b'linear':
                        raise ValueError(f'Unsupported RGB Resize: {following.name}')
                pending.extend(following.output)
    kept = []
    for node in graph.node:
        if node.name in removed:
            continue
        for i, value in enumerate(node.input):
            while value in replacements:
                value = replacements[value]
            node.input[i] = value
        kept.append(node)
    output = graph.output[0].name
    internal = output + '_before_rgb_bias'
    for node in kept:
        for i, value in enumerate(node.output):
            if value == output:
                node.output[i] = internal
    dtype = next(iter(biases.values())).dtype
    total = np.sum(np.stack(list(biases.values())).astype(np.float64), axis=0).astype(dtype)
    bias_name = 'gfpgan_consolidated_rgb_bias'
    graph.initializer.append(numpy_helper.from_array(total, bias_name))
    kept.append(helper.make_node('Add', [internal, bias_name], [output], name='ConsolidatedRGBBias'))
    del graph.node[:]
    graph.node.extend(kept)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print({'removed_bias_adds': removed, 'total_rgb_bias': total.reshape(-1).tolist(), 'output': str(args.output)})


if __name__ == '__main__':
    main()
