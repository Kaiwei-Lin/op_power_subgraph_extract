from contextlib import redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T

from onnx_graph import similarity as s


def vi(name, shape=(1, 2, 4, 4), dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


def conv_model(weight=1, channels=2, kernel=1, group=1, dtype=T.FLOAT, name="first"):
    dims = [channels, channels // group, kernel, kernel]
    weight_tensor = h.make_tensor("w", dtype, dims, [weight] * int(np.prod(dims)))
    nodes = [h.make_node("Conv", ["x", "w"], ["a"], group=group,
                         pads=[kernel // 2] * 4, name=name + "/conv"),
             h.make_node("Relu", ["a"], ["y"], name=name + "/relu")]
    return h.make_model(h.make_graph(nodes, name, [vi("x", (1, channels, 4, 4), dtype)],
                                    [vi("y", (1, channels, 4, 4), dtype)], [weight_tensor]),
                        opset_imports=[h.make_opsetid("", 13)], ir_version=8)


def simple_model(nodes, inputs=("x",), outputs=("y",), initializers=()):
    return h.make_model(h.make_graph(nodes, "test", [vi(v) for v in inputs],
                                    [vi(v) for v in outputs], initializers),
                        opset_imports=[h.make_opsetid("", 13)], ir_version=8)


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="onnx-similarity-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)

    def describe(self, model):
        return s.describe_model(model)

    def assertDifferent(self, left, right):
        left, right = self.describe(left), self.describe(right)
        self.assertFalse(s.same_configuration(left, right))
        self.assertLess(s.feature_similarity(left["config"], right["config"]), 1)

    def save(self, model, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, path)
        return path

    def test_names_metadata_and_float_weight_values_do_not_define_configuration(self):
        left = conv_model()
        right = conv_model(weight=9, name="different")
        right.producer_name = "another exporter"
        right.doc_string = "metadata"
        for node in right.graph.node:
            node.name = ""  # 空名称也不影响拓扑比较。
            node.input[:] = ["renamed_" + value for value in node.input]
            node.output[:] = ["renamed_" + value for value in node.output]
        for value in [*right.graph.input, *right.graph.output, *right.graph.initializer]:
            value.name = "renamed_" + value.name
        a, b = self.describe(left), self.describe(right)
        self.assertTrue(s.same_configuration(a, b))
        self.assertEqual(s.feature_similarity(a["config"], b["config"]), 1)

    def test_shapes_dtypes_kernel_group_and_attributes_must_match(self):
        for model in [conv_model(channels=4), conv_model(dtype=T.FLOAT16),
                      conv_model(kernel=3), conv_model(group=2)]:
            with self.subTest(model=model.graph.name):
                self.assertDifferent(conv_model(), model)
        left = simple_model([h.make_node("LeakyRelu", ["x"], ["y"], alpha=0.1)])
        right = simple_model([h.make_node("LeakyRelu", ["x"], ["y"], alpha=0.2)])
        self.assertDifferent(left, right)

    def test_same_operator_counts_different_connections(self):
        left = simple_model([h.make_node("Relu", ["x"], ["a"]),
                             h.make_node("Sigmoid", ["a"], ["y"])])
        right = simple_model([h.make_node("Sigmoid", ["x"], ["a"]),
                              h.make_node("Relu", ["a"], ["y"])])
        a, b = self.describe(left), self.describe(right)
        self.assertEqual(a["operator_counts"], b["operator_counts"])
        self.assertFalse(s.same_configuration(a, b))
        self.assertLess(s.feature_similarity(a["structure"], b["structure"]), 1)
        branch = simple_model([h.make_node("Relu", ["x"], ["a"]),
                               h.make_node("Sigmoid", ["x"], ["b"]),
                               h.make_node("Add", ["a", "b"], ["y"])])
        chain = simple_model([h.make_node("Relu", ["x"], ["a"]),
                              h.make_node("Sigmoid", ["a"], ["b"]),
                              h.make_node("Add", ["a", "b"], ["y"])])
        self.assertDifferent(branch, chain)

    def test_input_ports_and_shared_tensor_identity_are_preserved(self):
        left = simple_model([h.make_node("Relu", ["x"], ["a"]),
                             h.make_node("Sub", ["x", "a"], ["y"])])
        right = deepcopy(left)
        right.graph.node[1].input[:] = ["a", "x"]
        self.assertDifferent(left, right)
        left = simple_model([h.make_node("Add", ["x", "x"], ["y"])], inputs=("x", "z"))
        right = simple_model([h.make_node("Add", ["x", "z"], ["y"])], inputs=("x", "z"))
        self.assertDifferent(left, right)

    def test_equivalent_branches_may_have_different_topological_order(self):
        left = simple_model([h.make_node("Relu", ["x"], ["a"]),
                             h.make_node("Sigmoid", ["x"], ["b"]),
                             h.make_node("Add", ["a", "b"], ["y"])])
        right = deepcopy(left)
        del right.graph.node[:]
        right.graph.node.extend([left.graph.node[1], left.graph.node[0], left.graph.node[2]])
        self.assertTrue(s.same_configuration(self.describe(left), self.describe(right)))

    def test_control_constants_preserved_including_upstream_float_computation(self):
        def reshape(shape):
            node = h.make_node("Reshape", ["x", "shape"], ["y"])
            weight = h.make_tensor("shape", T.INT64, [4], shape)
            model = simple_model([node], initializers=[weight])
            model.graph.output[0].CopyFrom(vi("y", shape))
            return model
        self.assertDifferent(reshape([1, 2, 4, 4]), reshape([1, 2, 2, 8]))

        def clip(value):
            constants = [h.make_tensor("p", T.FLOAT, [], [value]),
                         h.make_tensor("q", T.FLOAT, [], [1])]
            return simple_model([h.make_node("Add", ["p", "q"], ["limit"]),
                                 h.make_node("Clip", ["x", "limit"], ["y"])], initializers=constants)
        self.assertDifferent(clip(0), clip(2))

    def test_constant_weight_and_external_initializer_values_ignored(self):
        left = conv_model()
        right = conv_model(weight=7)
        for model in (left, right):
            weight = deepcopy(model.graph.initializer[0])
            del model.graph.initializer[:]
            model.graph.node.insert(0, h.make_node("Constant", [], ["w"], value=weight))
        self.assertTrue(s.same_configuration(self.describe(left), self.describe(right)))
        model = conv_model()
        model.graph.initializer[0].CopyFrom(
            onnx.numpy_helper.from_array(np.ones([2, 2, 1, 1], np.float32), name="w"))
        path = self.root / "external.onnx"
        onnx.save_model(model, path, save_as_external_data=True,
                        all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
        self.assertTrue(s.same_configuration(self.describe(path), self.describe(conv_model(weight=9))))

    def test_wl_collision_still_requires_isomorphism(self):
        # 有向六环和两个有向三环的 WL 计数相同，但并不同构。
        def item(edges):
            graph = nx.MultiDiGraph()
            graph.add_nodes_from((i, dict(config="same")) for i in range(6))
            graph.add_edges_from((u, v, dict(port=0)) for u, v in edges)
            return dict(graph=graph, complete=True, config=s.wl_features(graph))
        left = item([(i, (i + 1) % 6) for i in range(6)])
        right = item([(i, i // 3 * 3 + (i + 1) % 3) for i in range(6)])
        self.assertEqual(left["config"], right["config"])
        self.assertFalse(s.same_configuration(left, right))
        self.assertLess(s.feature_similarity({(0, "x"): 1}, {(0, "x"): 2}), 1)

    def test_dynamic_shapes_warn_and_unknown_shapes_never_auto_merge(self):
        model = conv_model()
        for value in [*model.graph.input, *model.graph.output]:
            value.type.tensor_type.shape.dim[0].dim_param = "batch"
        item = self.describe(model)
        self.assertTrue(item["dynamic"])
        self.assertTrue(s.same_configuration(item, self.describe(model)))
        unknown = simple_model([h.make_node("Unknown", ["x"], ["a"], domain="custom"),
                                h.make_node("Unknown", ["a"], ["y"], domain="custom")])
        unknown.opset_import.append(h.make_opsetid("custom", 1))
        item = self.describe(unknown)
        self.assertFalse(item["complete"])
        self.assertFalse(s.same_configuration(item, self.describe(unknown)))
        self.assertEqual(s.complete_link_groups(np.ones((2, 2)), 0.9, [False, True]), [[0], [1]])

    def test_complete_link_prevents_transitive_similarity_chain_and_picks_closest(self):
        scores = np.array([[1, .95, .6], [.95, 1, .99], [.6, .99, 1]])
        groups = s.complete_link_groups(scores, .9)
        self.assertEqual(groups, [[0], [1, 2]])
        for group in groups:
            self.assertTrue(all(scores[i, j] >= .9 for i in group for j in group))
        self.assertEqual(s.complete_link_groups(np.ones((4, 4)), 1), [[0, 1, 2, 3]])

    def test_reports_keep_models_separate_levels_and_relocate_manifest(self):
        paths = [self.save(conv_model(), "batch/slash_1/a.onnx"),
                 self.save(conv_model(weight=8), "batch/slash_1/b.onnx"),
                 self.save(conv_model(channels=4), "batch/slash_1/c.onnx"),
                 self.save(conv_model(), "batch/slash_2/a.onnx")]
        before = {path: path.read_bytes() for path in paths}
        root = self.root / "batch"
        manifest = dict(records=[dict(slash_count=1, onnx_file=r"Z:\old\slash_1\a.onnx",
                                     prefix="中文/model", status="exported")])
        (root / "batch_export.json").write_text(json.dumps(manifest), encoding="utf-8")
        with redirect_stdout(StringIO()):
            report = s.group_subgraphs(root)
        self.assertEqual([(l["models"], l["groups"]) for l in report["levels"]], [(3, 2), (1, 1)])
        level = report["levels"][0]
        self.assertEqual([m["name"] for m in level["members"][:2]], ["中文/model", "b"])
        self.assertTrue(all(Path(r["representative_path"]).exists() for r in level["representatives"]))
        for path in paths:
            self.assertEqual(before[path], path.read_bytes())
        folder = root / "grouping_exact" / "slash_1"
        for name, count in [("groups.csv", 3), ("representatives.csv", 2), ("similarity.csv", 3)]:
            self.assertTrue((folder / name).read_bytes().startswith(b"\xef\xbb\xbf"))
            with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), count)
        with redirect_stdout(StringIO()):
            merged = s.group_subgraphs(root, separate_slash_counts=False)
        self.assertEqual([(l["models"], l["groups"]) for l in merged["levels"]], [(4, 2)])

    def test_similar_mode_is_opt_in_and_bad_files_are_reported(self):
        base = conv_model()
        other = conv_model(channels=4)
        a, b = self.describe(base), self.describe(other)
        score = s.feature_similarity(a["config"], b["config"])
        self.assertGreater(score, 0)
        self.save(base, "batch/a.onnx")
        self.save(other, "batch/b.onnx")
        (self.root / "batch/bad.onnx").write_bytes(b"invalid")
        with redirect_stdout(StringIO()):
            report = s.group_subgraphs(self.root / "batch", mode="similar", threshold=score)
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["levels"][0]["groups"], 1)
        self.assertEqual(report["levels"][0]["members"][0]["basis"], "高度相似")
        self.assertAlmostEqual(report["levels"][0]["members"][0]["minimum_similarity"], score)

    def test_unsupported_graphs_and_invalid_configuration_fail_explicitly(self):
        graph = h.make_graph([h.make_node("Identity", ["x"], ["z"])],
                            "branch", [], [vi("z")])
        model = simple_model([h.make_node("If", ["cond"], ["y"], then_branch=graph, else_branch=graph)],
                             inputs=("x",))
        model.graph.input.append(vi("cond", [], T.BOOL))
        with self.assertRaisesRegex(ValueError, "控制流"):
            self.describe(model)
        for kwargs in [dict(mode="bad"), dict(threshold=0), dict(threshold=1.1),
                       dict(threshold=float("nan")), dict(rounds=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                s.group_subgraphs(self.root, **kwargs)
        with self.assertRaises(FileNotFoundError):
            s.group_subgraphs(self.root)
        with self.assertRaises(ValueError):
            s.group_subgraphs(self.root, self.root)

    def test_main_honors_configuration_and_command_line(self):
        self.save(conv_model(), "batch/a.onnx")
        with patch.multiple(s, INPUT_DIR=self.root / "batch", GROUP_MODE="similar",
                            SIMILARITY_THRESHOLD=.8), redirect_stdout(StringIO()):
            report = s.main([])
        self.assertEqual(report["mode"], "similar")
        self.assertEqual(report["threshold"], .8)


if __name__ == "__main__":
    unittest.main()
