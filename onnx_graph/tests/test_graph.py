import pickle
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper

from onnx_graph import (
    extract_prefix, networkx_to_onnx, nodes_by_prefix, onnx_to_networkx,
    refresh_graph, repair_node_names,
)


def vi(name, shape=(2,), dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


def model(nodes, inputs, outputs, initializers=(), **graph_fields):
    graph = h.make_graph(nodes, "test", inputs, outputs, list(initializers), **graph_fields)
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 15)], ir_version=8,
                        producer_name="test-producer", producer_version="1", doc_string="model docs")


def chain():
    return model([
        h.make_node("Add", ["x", "w"], ["a"], name="root/block_1/Add"),
        h.make_node("Mul", ["a", "a"], ["b"], name="root/block_1/Mul"),
        h.make_node("Relu", ["b"], ["y"], name="root/block_10/Relu"),
    ], [vi("x")], [vi("y")], [numpy_helper.from_array(np.array([1, 2], np.float32), "w")])


class ConversionTests(unittest.TestCase):
    def test_exact_roundtrip_pickle_and_all_fields(self):
        m = chain()
        m.graph.doc_string = "graph docs"
        m.graph.node[0].doc_string = "node docs"
        h.set_model_props(m, {"source": "unit-test"})
        m.graph.input[0].type.tensor_type.shape.dim[0].denotation = "DATA_BATCH"
        raw = m.SerializeToString()
        g = onnx_to_networkx(m)
        self.assertIsInstance(g, nx.MultiDiGraph)
        self.assertEqual(g.number_of_edges(0, 1), 2)
        self.assertEqual({e["input_index"] for e in g[0][1].values()}, {0, 1})
        self.assertEqual(g.nodes[0]["hierarchy"], ("root", "block_1", "Add"))
        restored = networkx_to_onnx(pickle.loads(pickle.dumps(g)))
        self.assertEqual(raw, restored.SerializeToString())
        g.nodes[0]["proto"].name = "changed"
        self.assertEqual(raw, m.SerializeToString())

    def test_missing_graph_name_is_filled_for_full_and_subgraph_export(self):
        for name in (None, "", " \t"):
            with self.subTest(name=name):
                m = chain()
                if name is None:
                    m.graph.ClearField("name")
                else:
                    m.graph.name = name
                m.graph.node[2].ClearField("name")
                raw = m.SerializeToString()
                graph = onnx_to_networkx(m)
                out = networkx_to_onnx(graph)
                self.assertEqual(out.graph.name, "onnx_graph")
                self.assertFalse(out.graph.node[2].HasField("name"))
                self.assertEqual(networkx_to_onnx(onnx_to_networkx(out)).SerializeToString(), out.SerializeToString())
                sub = networkx_to_onnx(extract_prefix(graph, "root/block_1"))
                self.assertEqual(sub.graph.name, "onnx_graph")
                self.assertEqual([v.name for v in sub.graph.output], ["b"])
                self.assertEqual(graph.graph["model"].SerializeToString(), raw)
                self.assertEqual(m.SerializeToString(), raw)
                if name is None:
                    out.graph.ClearField("name")
                else:
                    out.graph.name = name
                self.assertEqual(out.SerializeToString(), raw)

    def test_missing_nested_graph_names_are_filled(self):
        branch = h.make_graph([h.make_node("Identity", ["x"], ["z"])], "", [], [vi("z")])
        m = model([
            h.make_node("If", ["cond"], ["a"], name="block/If", then_branch=branch, else_branch=branch),
            h.make_node("Identity", ["a"], ["y"], name="outside"),
        ], [vi("x"), vi("cond", (), T.BOOL)], [vi("y")])
        raw = m.SerializeToString()
        graph = onnx_to_networkx(m)
        for selected in (graph, extract_prefix(graph, "block")):
            out = networkx_to_onnx(selected)
            self.assertEqual(out.graph.name, "test")
            names = [attr.g.name for attr in out.graph.node[0].attribute]
            self.assertTrue(all(names))
            self.assertEqual(len(set(names)), 2)
            onnx.checker.check_model(out, full_check=True)
        self.assertEqual(m.SerializeToString(), raw)
        self.assertEqual(graph.nodes[0]["proto"].SerializeToString(), m.graph.node[0].SerializeToString())

    def test_example_main_runs_with_missing_graph_name(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from onnx_graph import main as example

        m = chain()
        m.graph.ClearField("name")
        with tempfile.TemporaryDirectory(prefix="onnx-test-", dir=Path.cwd()) as folder:
            directory = Path(folder).resolve()
            self.assertEqual(directory.parent, Path.cwd().resolve())
            source = directory / "source.onnx"
            output = directory / "result"
            onnx.save(m, source)
            with patch.multiple(example, MODEL_PATH=source, PREFIX="root/block_1",
                                REPAIR_NAMES=False, OUTPUT_DIR=output,
                                BATCH_FROM_STATS=False), redirect_stdout(StringIO()):
                example.main()
            exported = onnx.load(output / "subgraph.onnx")
            self.assertEqual(exported.graph.name, "onnx_graph")
            onnx.checker.check_model(exported, full_check=True)
            saved_graph = pickle.loads((output / "full_graph.nx.pkl").read_bytes())
            self.assertFalse(saved_graph.graph["model"].graph.HasField("name"))
            self.assertEqual(source.read_bytes(), m.SerializeToString())

    def test_duplicate_and_unnamed_nodes(self):
        m = chain()
        for n in m.graph.node:
            n.name = ""
        g = onnx_to_networkx(m)
        self.assertEqual(len(g), 3)
        self.assertEqual(networkx_to_onnx(g).SerializeToString(), m.SerializeToString())
        for n in m.graph.node:
            n.name = "duplicate"
        self.assertEqual(nodes_by_prefix(onnx_to_networkx(m), "duplicate"), [0, 1, 2])

    def test_prefix_boundaries_and_independent_copy(self):
        g = onnx_to_networkx(chain())
        self.assertEqual(nodes_by_prefix(g, "/root/block_1/"), [0, 1])
        self.assertEqual(nodes_by_prefix(g, "root/block"), [])
        with self.assertRaisesRegex(ValueError, "No ONNX nodes"):
            extract_prefix(g, "missing")
        sub = extract_prefix(g, "root/block_1")
        sub.nodes[0]["proto"].name = "new/name"
        self.assertEqual(g.nodes[0]["name"], "root/block_1/Add")
        out = networkx_to_onnx(sub)
        self.assertEqual([v.name for v in out.graph.input], ["x"])
        self.assertEqual([v.name for v in out.graph.output], ["b"])
        self.assertEqual([t.name for t in out.graph.initializer], ["w"])
        self.assertEqual(out.graph.node[0].name, "new/name")

    def test_native_networkx_subgraph_and_inferred_boundary(self):
        g = onnx_to_networkx(chain())
        sub = g.subgraph([1]).copy()
        out = networkx_to_onnx(sub)
        self.assertEqual([v.name for v in out.graph.input], ["a"])
        self.assertEqual([v.name for v in out.graph.output], ["b"])
        self.assertEqual(len(out.graph.initializer), 0)
        self.assertEqual(out.graph.input[0].type.tensor_type.elem_type, T.FLOAT)
        self.assertEqual(out.graph.input[0].type.tensor_type.shape.dim[0].dim_value, 2)

    def test_relabelled_graph_and_readonly_subgraph(self):
        m = chain()
        g = nx.relabel_nodes(onnx_to_networkx(m), {0: "first", 1: "second", 2: "third"})
        self.assertEqual(networkx_to_onnx(g).SerializeToString(), m.SerializeToString())
        out = networkx_to_onnx(g.subgraph(["second"]))
        self.assertEqual([v.name for v in out.graph.input], ["a"])
        self.assertEqual([v.name for v in out.graph.output], ["b"])

    def test_valid_zero_operator_model(self):
        m = model([], [vi("x")], [vi("x")])
        self.assertEqual(networkx_to_onnx(onnx_to_networkx(m)).SerializeToString(), m.SerializeToString())

    def test_branch_output_consumed_both_inside_and_outside(self):
        m = chain()
        m.graph.node.append(h.make_node("Identity", ["a"], ["other"], name="outside"))
        m.graph.output.append(vi("other"))
        out = networkx_to_onnx(extract_prefix(onnx_to_networkx(m), "root/block_1"))
        self.assertEqual([v.name for v in out.graph.output], ["a", "b"])

    def test_attribute_and_optional_input_preservation(self):
        m = model([
            h.make_node("Constant", [], ["upper"], name="const", value=numpy_helper.from_array(np.float32(3))),
            h.make_node("Clip", ["x", "", "upper"], ["y"], name="clip"),
        ], [vi("x")], [vi("y")])
        g = onnx_to_networkx(m)
        self.assertEqual(g[0][1][0]["input_index"], 2)
        self.assertEqual(networkx_to_onnx(g).SerializeToString(), m.SerializeToString())

    def test_multiple_outputs_and_parallel_edges(self):
        m = model([
            h.make_node("Split", ["x"], ["a", "b"], name="block/Split", axis=0),
            h.make_node("Add", ["a", "b"], ["y"], name="block/Add"),
        ], [vi("x", (4,))], [vi("y")])
        g = onnx_to_networkx(m)
        self.assertEqual({e["output_index"] for e in g[0][1].values()}, {0, 1})
        out = networkx_to_onnx(g.subgraph([0]).copy())
        self.assertEqual([v.name for v in out.graph.output], ["a", "b"])

    def test_edit_proto_and_rebuild_dependencies(self):
        g = onnx_to_networkx(chain())
        g.nodes[2]["proto"].input[0] = "a"
        g.nodes[2]["proto"].name = "root/new/Relu"
        refresh_graph(g)
        self.assertTrue(g.has_edge(0, 2))
        self.assertFalse(g.has_edge(1, 2))
        self.assertEqual(nodes_by_prefix(g, "root/new"), [2])
        self.assertEqual(networkx_to_onnx(g).graph.node[2].input[0], "a")

    def test_unknown_boundary_is_not_guessed(self):
        g = onnx_to_networkx(chain())
        sub = g.subgraph([1]).copy()
        with self.assertRaisesRegex(ValueError, "Missing boundary type/shape"):
            networkx_to_onnx(sub, infer_shapes=False)
        out = networkx_to_onnx(sub, infer_shapes=False, value_info={"a": vi("a"), "b": vi("b")})
        self.assertEqual(out.graph.input[0].name, "a")
        with self.assertRaisesRegex(ValueError, "does not match"):
            networkx_to_onnx(sub, value_info={"a": vi("wrong")})

    def test_cycles_and_duplicate_producers_fail(self):
        g = onnx_to_networkx(chain())
        g.nodes[0]["proto"].input[0] = "y"
        with self.assertRaisesRegex(ValueError, "cycle"):
            networkx_to_onnx(g)
        m = chain()
        m.graph.node[1].output[0] = "a"
        with self.assertRaisesRegex(ValueError, "Multiple producers"):
            onnx_to_networkx(m)

    def test_control_flow_lexical_captures(self):
        branch = h.make_graph([h.make_node("Identity", ["hidden"], ["z"])], "branch", [], [vi("z")])
        m = model([
            h.make_node("Identity", ["x"], ["hidden"], name="outside"),
            h.make_node("If", ["cond"], ["y"], name="block/If", then_branch=branch, else_branch=branch),
        ], [vi("x"), vi("cond", (), T.BOOL)], [vi("y")])
        g = onnx_to_networkx(m)
        self.assertTrue(g[0][1][0]["implicit"])
        self.assertEqual(g[0][1][0]["tensor"], "hidden")
        out = networkx_to_onnx(extract_prefix(g, "block"))
        self.assertEqual([v.name for v in out.graph.input], ["cond", "hidden"])
        self.assertEqual(out.graph.node[0].SerializeToString(), m.graph.node[1].SerializeToString())

    def test_sparse_initializer_and_overridable_weight(self):
        dense = numpy_helper.from_array(np.array([1, 2], np.float32), "w")
        sparse = h.make_sparse_tensor(numpy_helper.from_array(np.array([3], np.float32), "s"),
                                      numpy_helper.from_array(np.array([0], np.int64)), [2])
        m = model([
            h.make_node("Add", ["x", "w"], ["a"], name="block/Add"),
            h.make_node("Add", ["a", "s"], ["b"], name="block/Add2"),
            h.make_node("Identity", ["b"], ["y"], name="outside"),
        ], [vi("x"), vi("w")], [vi("y")], [dense])
        m.graph.sparse_initializer.append(sparse)
        out = networkx_to_onnx(extract_prefix(onnx_to_networkx(m), "block"))
        self.assertEqual([v.name for v in out.graph.input], ["x", "w"])
        self.assertEqual(out.graph.sparse_initializer[0].SerializeToString(), sparse.SerializeToString())

    def test_local_function_and_quantization_metadata(self):
        m = model([
            h.make_node("Copy", ["x"], ["a"], name="block/Copy", domain="local"),
            h.make_node("Identity", ["a"], ["y"], name="outside"),
        ], [vi("x")], [vi("y")], [numpy_helper.from_array(np.float32(0.1), "scale")],
            value_info=[vi("a")])
        m.opset_import.append(h.make_opsetid("local", 1))
        m.functions.append(h.make_function("local", "Copy", ["x"], ["y"],
            [h.make_node("Identity", ["x"], ["y"])], [h.make_opsetid("", 15)]))
        annotation = m.graph.quantization_annotation.add(tensor_name="a")
        annotation.quant_parameter_tensor_names.add(key="SCALE_TENSOR", value="scale")
        out = networkx_to_onnx(extract_prefix(onnx_to_networkx(m), "block"))
        self.assertEqual(out.functions, m.functions)
        self.assertEqual(out.graph.quantization_annotation, m.graph.quantization_annotation)
        self.assertEqual([t.name for t in out.graph.initializer], ["scale"])

    def test_external_weights_remain_portable(self):
        with tempfile.TemporaryDirectory(prefix="onnx-test-", dir=Path.cwd()) as folder:
            self.assertEqual(Path(folder).resolve().parent, Path.cwd().resolve())
            source = Path(folder) / "source.onnx"
            onnx.save_model(chain(), source, save_as_external_data=True, all_tensors_to_one_file=True,
                            location="weights.bin", size_threshold=0)
            graph = onnx_to_networkx(source)
            out = networkx_to_onnx(graph)
            self.assertEqual(out.graph.initializer[0].data_location, T.DEFAULT)
            self.assertEqual(numpy_helper.to_array(out.graph.initializer[0]).tolist(), [1, 2])


class NameRepairTests(unittest.TestCase):
    def repair(self, name, outputs, inputs=(), op="Reshape"):
        m = model([h.make_node(op, list(inputs), outputs, name=name)], [], [])
        fixed, report = repair_node_names(m)
        self.assertEqual(m.graph.node[0].name, name)
        self.assertEqual(list(fixed.graph.node[0].input), list(inputs))
        self.assertEqual(list(fixed.graph.node[0].output), outputs)
        return fixed.graph.node[0].name, report[0]

    def test_user_example_and_star_separators(self):
        for name in ("mono_model_cfb_d3_LN2_Reshape", "mono_model*cfb_d3*LN2*Reshape", "mono*model*cfb*d3*LN2*Reshape"):
            with self.subTest(name=name):
                fixed, record = self.repair(name, ["/mono_model/cfb_d3/LN2/Reshape_output_0"])
                self.assertEqual(fixed, "/mono_model/cfb_d3/LN2/Reshape")
                self.assertEqual(record["status"], "renamed")

    def test_node_name_repair_also_fills_required_graph_name(self):
        m = model([h.make_node("Identity", ["x"], ["y"])], [vi("x")], [vi("y")])
        m.graph.ClearField("name")
        raw = m.SerializeToString()
        fixed, report = repair_node_names(m)
        self.assertEqual(fixed.graph.name, "onnx_graph")
        self.assertEqual(report[0]["status"], "skipped_unnamed")
        self.assertFalse(fixed.graph.node[0].HasField("name"))
        onnx.checker.check_model(fixed)
        self.assertEqual(m.SerializeToString(), raw)

    def test_unnamed_input_and_output_operators_are_skipped(self):
        m = model([
            h.make_node("Identity", ["input"], ["/net/stem/Identity_output_0"]),
            h.make_node("Identity", ["/net/stem/Identity_output_0"], ["output"]),
        ], [vi("input")], [vi("output")])
        raw = m.SerializeToString()
        self.assertFalse(m.graph.node[0].HasField("name"))
        self.assertEqual(networkx_to_onnx(onnx_to_networkx(m)).SerializeToString(), raw)
        graph = onnx_to_networkx(m, repair_names=True)
        self.assertEqual([graph.nodes[i]["name"] for i in graph], ["", ""])
        self.assertEqual(nodes_by_prefix(graph, "net/stem"), [])
        self.assertEqual(nodes_by_prefix(graph, ""), [0, 1])
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual([graph.nodes[i]["original_name"] for i in graph], ["", ""])
        self.assertEqual([r["status"] for r in graph.graph["rename_report"]], ["skipped_unnamed"] * 2)
        fixed = networkx_to_onnx(graph)
        self.assertEqual(fixed.graph.input, m.graph.input)
        self.assertEqual(fixed.graph.output, m.graph.output)
        self.assertEqual(fixed.SerializeToString(), raw)
        self.assertEqual(m.SerializeToString(), raw)

    def test_unnamed_nodes_ignore_tensor_naming_patterns(self):
        for old_name in ("", " \t"):
            for outputs in (["output"], ["123"], ["Identity:0"], ["/net/Identity_output_0"],
                            ["/net/Split_output_0", "/net/Split_output_1"], ["a", "b"], [""], []):
                with self.subTest(name=old_name, outputs=outputs):
                    fixed, record = self.repair(old_name, outputs, ["/upstream/Identity_output_0"])
                    self.assertEqual(fixed, old_name)
                    self.assertEqual(record["status"], "skipped_unnamed")
                    self.assertEqual(record["candidates"], [])

    def test_named_block_between_unnamed_boundary_operators(self):
        m = model([
            h.make_node("Identity", ["input"], ["/stem/Identity_output_0"]),
            h.make_node("Relu", ["/stem/Identity_output_0"], ["/net/block/Relu_output_0"], name="net_block_Relu"),
            h.make_node("Identity", ["/net/block/Relu_output_0"], ["output"], name=" \t"),
        ], [vi("input")], [vi("output")])
        graph = onnx_to_networkx(m, repair_names=True)
        self.assertEqual(graph.nodes[2]["hierarchy"], ())
        self.assertEqual(nodes_by_prefix(graph, "net/block"), [1])
        self.assertEqual(graph.number_of_edges(), 2)
        submodel = networkx_to_onnx(extract_prefix(graph, "net/block"))
        self.assertEqual([v.name for v in submodel.graph.input], ["/stem/Identity_output_0"])
        self.assertEqual([v.name for v in submodel.graph.output], ["/net/block/Relu_output_0"])
        self.assertEqual(submodel.graph.node[0].name, "/net/block/Relu")
        fixed = networkx_to_onnx(graph)
        fixed.graph.node[1].CopyFrom(m.graph.node[1])
        self.assertEqual(fixed.SerializeToString(), m.SerializeToString())

    def test_input_scope_fallback(self):
        fixed, record = self.repair("mono_model_cfb_d3_LN2_Reshape", ["842"],
                                    ["/mono_model/cfb_d3/LN2/Mul_output_0"])
        self.assertEqual(fixed, "/mono_model/cfb_d3/LN2/Reshape")
        self.assertEqual(record["source"], "io_scope")

    def test_existing_hierarchy_and_unrelated_output_stay(self):
        for name, output, status in [
            ("MobilenetV3/expanded_conv/depthwise/depthwise", "MobilenetV3/expanded_conv/depthwise/BatchNorm/FusedBatchNorm:0", "already_hierarchical"),
            ("unknown_Reshape", "/another/block/Reshape_output_0", "unresolved"),
        ]:
            fixed, record = self.repair(name, [output])
            self.assertEqual(fixed, name)
            self.assertEqual(record["status"], status)

    def test_ambiguous_outputs_remain_unchanged(self):
        name = "a_b_c_Reshape"
        fixed, record = self.repair(name, ["/a_b/c/Reshape_output_0", "/a/b_c/Reshape_output_1"])
        self.assertEqual(fixed, name)
        self.assertEqual(record["status"], "ambiguous")

    def test_multiple_outputs_and_tensorflow_suffix(self):
        fixed, _ = self.repair("a_b_Split", ["/a/b/Split_output_0", "/a/b/Split_output_1"], op="Split")
        self.assertEqual(fixed, "/a/b/Split")
        fixed, _ = self.repair("a_b_Reshape", ["a/b/Reshape:0"])
        self.assertEqual(fixed, "a/b/Reshape")

    def test_collision_report_and_original_name(self):
        m = model([
            h.make_node("Identity", ["x"], ["/a/b/Identity_output_0"], name="a_b_Identity"),
            h.make_node("Identity", ["x"], ["y"], name="/a/b/Identity"),
        ], [vi("x")], [vi("y")])
        fixed, report = repair_node_names(m)
        self.assertEqual(report[0]["status"], "collision")
        self.assertEqual(fixed.graph.node[0].name, "a_b_Identity")
        m.graph.node.pop()
        m.graph.output[0].name = "/a/b/Identity_output_0"
        graph = onnx_to_networkx(m, repair_names=True)
        self.assertEqual(graph.nodes[0]["original_name"], "a_b_Identity")
        self.assertEqual(nodes_by_prefix(graph, "/a/b"), [0])
        self.assertEqual(networkx_to_onnx(graph).graph.node[0].name, "/a/b/Identity")


if __name__ == "__main__":
    unittest.main()
