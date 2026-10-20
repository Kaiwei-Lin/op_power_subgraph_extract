"""Reproduce the supplied MobileNetV3 conversion and compare real activations.

Run: python -m onnx_graph.verify_mobilenet
Requires onnxruntime in addition to requirements.txt.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import onnx
import onnxruntime as ort

from .core import extract_prefix, networkx_to_onnx, onnx_to_networkx, repair_node_names


def session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 2
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def main():
    source = Path(__file__).resolve().parent / "models/mobilenetv3/mobilenetv3-large_224_1.0.onnx"
    output = source.parent / "networkx"
    output.mkdir(exist_ok=True)
    original_bytes = source.read_bytes()
    graph = onnx_to_networkx(source)
    graph_path = output / "mobilenetv3.nx.pkl"
    with graph_path.open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with graph_path.open("rb") as stream:
        restored = networkx_to_onnx(pickle.load(stream))
    assert restored.SerializeToString() == original_bytes
    onnx.save(restored, output / "mobilenetv3.roundtrip.onnx")

    prefix = "MobilenetV3/expanded_conv_1"
    subgraph = extract_prefix(graph, prefix)
    extracted = networkx_to_onnx(subgraph)
    onnx.checker.check_model(extracted, full_check=True)
    onnx.save(extracted, output / "expanded_conv_1.onnx")
    with (output / "expanded_conv_1.nx.pkl").open("wb") as stream:
        pickle.dump(subgraph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(subgraph) == 5
    assert all(n.name.startswith(prefix + "/") for n in extracted.graph.node)
    for i, n in zip(subgraph, extracted.graph.node):
        assert n.SerializeToString() == graph.nodes[i]["proto"].SerializeToString()

    # Expose the real block input/output in the complete model to avoid feeding
    # an unrelated random activation into the extracted block.
    instrumented = deepcopy(graph.graph["model"])
    boundary = {v.name: v for v in [*extracted.graph.input, *extracted.graph.output]}
    original_output_names = [v.name for v in instrumented.graph.output]
    instrumented.graph.output.extend(v for name, v in boundary.items() if name not in original_output_names)
    input_name = instrumented.graph.input[0].name
    x = np.random.default_rng(20261019).uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    full_session = session(instrumented)
    full_values = dict(zip([v.name for v in full_session.get_outputs()], full_session.run(None, {input_name: x})))
    sub_outputs = session(extracted).run(None, {v.name: full_values[v.name] for v in extracted.graph.input})
    errors = {}
    for info, actual in zip(extracted.graph.output, sub_outputs):
        expected = full_values[info.name]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        errors[info.name] = float(np.max(np.abs(actual - expected)))
    roundtrip_outputs = session(restored).run(None, {input_name: x})
    for name, actual in zip(original_output_names, roundtrip_outputs):
        np.testing.assert_allclose(actual, full_values[name], rtol=1e-5, atol=1e-6)

    example = onnx.helper.make_model(onnx.helper.make_graph([
        onnx.helper.make_node("Reshape", ["x", "shape"], ["/mono_model/cfb_d3/LN2/Reshape_output_0"],
                              name="mono_model_cfb_d3_LN2_Reshape"),
    ], "name_repair_example", [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
        [onnx.helper.make_tensor_value_info("/mono_model/cfb_d3/LN2/Reshape_output_0", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.numpy_helper.from_array(np.array([1, 2], np.int64), "shape")]),
        opset_imports=[onnx.helper.make_opsetid("", 15)], ir_version=8)
    onnx.save(example, output / "name_repair_example.onnx")
    fixed, rename_report = repair_node_names(example)
    assert fixed.graph.node[0].name == "/mono_model/cfb_d3/LN2/Reshape"
    onnx.checker.check_model(fixed)
    onnx.save(fixed, output / "name_repair_example.fixed.onnx")
    (output / "name_repair_example.rename.json").write_text(json.dumps(rename_report, indent=2), encoding="utf-8")

    assert source.read_bytes() == original_bytes
    report = dict(source=str(source), source_sha256=hashlib.sha256(original_bytes).hexdigest(),
                  onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                  nodes=len(graph), edges=graph.number_of_edges(), initializers=len(restored.graph.initializer),
                  byte_identical_roundtrip=True, source_unchanged=True,
                  prefix=prefix, selected_nodes=len(subgraph), selected_edges=subgraph.number_of_edges(),
                  selected_initializers=len(extracted.graph.initializer),
                  subgraph_inputs=[v.name for v in extracted.graph.input],
                  subgraph_outputs=[v.name for v in extracted.graph.output],
                  full_and_subgraph_checker="PASS", runtime_comparison="PASS",
                  random_seed=20261019, optimization="disabled", rtol=1e-5, atol=1e-6,
                  subgraph_max_abs_error=errors, example_repair="PASS")
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
