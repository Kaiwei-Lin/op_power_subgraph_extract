from contextlib import redirect_stdout
from io import StringIO
import csv
import json
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import TensorProto as T, helper as h

from onnx_graph.stats import analyze_subgraphs, format_report, main, write_csv_reports


def vi(name, dims, dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, dims)


def model(nodes, inputs, outputs, weights=(), opset=17):
    return h.make_model(h.make_graph(nodes, "", inputs, outputs, initializer=weights),
                        opset_imports=[h.make_opsetid("", opset), h.make_opsetid("custom", 1)],
                        ir_version=8)


class StatisticsTests(unittest.TestCase):
    def test_hierarchy_membership_unnamed_and_shallow_nodes(self):
        names = ["root/block_1/Add", "/root/block_1/sub/Relu", "root/block_10/Mul", "", "root", " \t"]
        nodes = [h.make_node(op, args, [out], name=name) for op, args, out, name in zip(
            ["Add", "Relu", "Mul", "Identity", "Relu", "Identity"],
            [["x", "x"], ["a"], ["b", "b"], ["c"], ["d"], ["e"]],
            ["a", "b", "c", "d", "e", "y"], names)]
        m = model(nodes, [vi("x", [2, 3])], [vi("y", [2, 3])])
        raw = m.SerializeToString()
        report = analyze_subgraphs(m, [0, 1, 2])
        self.assertEqual(report["total"]["flops"], 24)
        self.assertEqual(m.SerializeToString(), raw)
        root, blocks, layers = report["levels"]
        self.assertEqual(root["subgraph_count"], 1)
        self.assertEqual(blocks["subgraph_count"], 2)
        self.assertEqual(blocks["subgraphs"][0]["node_ids"], [0, 1])
        self.assertEqual(blocks["subgraphs"][0]["operator_counts"], {"Add": 1, "Relu": 1})
        self.assertEqual(layers["subgraph_count"], 3)
        self.assertEqual(blocks["unnamed_node_ids"], [3, 5])
        self.assertEqual(blocks["shallower_node_ids"], [4])
        for level in report["levels"]:
            grouped = [i for sub in level["subgraphs"] for i in sub["node_ids"]]
            self.assertEqual(len(grouped), len(set(grouped)))
            self.assertEqual(sorted(grouped + level["unnamed_node_ids"] + level["shallower_node_ids"]), list(range(6)))

    def test_name_repair_switch_and_leading_slash(self):
        for name in ("mono_model_cfb_d3_LN2_Relu", "mono_model*cfb_d3*LN2*Relu"):
            output = "/mono_model/cfb_d3/LN2/Relu_output_0"
            m = model([h.make_node("Relu", ["x"], [output], name=name)],
                      [vi("x", [1, 4])], [vi(output, [1, 4])])
            plain = analyze_subgraphs(m, [1], repair_names=False)
            repaired = analyze_subgraphs(m, [1, 2], repair_names=True)
            self.assertEqual(plain["levels"][0]["subgraph_count"], 0)
            self.assertEqual(repaired["levels"][0]["subgraphs"][0]["prefix"], "mono_model/cfb_d3")
            self.assertEqual(repaired["levels"][1]["subgraphs"][0]["prefix"], "mono_model/cfb_d3/LN2")
            self.assertEqual(repaired["nodes"][0]["original_name"], name)

    def test_grouped_and_depthwise_conv_with_bias(self):
        # 72 输出元素，每个使用 2*3*3 个权重；depthwise 每个只用 3*3 个。
        for group, channels, outputs, expected in [(2, 2, 8, (1296, 2664)), (4, 1, 4, (324, 684))]:
            with self.subTest(group=group):
                weights = [h.make_tensor("w", T.FLOAT, [outputs, channels, 3, 3], [0.] * (outputs * channels * 9)),
                           h.make_tensor("bias", T.FLOAT, [outputs], [0.] * outputs)]
                m = model([h.make_node("Conv", ["x", "w", "bias"], ["y"], name="root/block/Conv", group=group)],
                          [vi("x", [1, 4, 5, 5])], [vi("y", None)], weights)
                total = analyze_subgraphs(m)["total"]
                self.assertEqual((total["macs"], total["flops"]), expected)

    def test_matmul_batch_broadcast_and_vectors(self):
        for a, b, expected in [([2, 3, 4], [4, 5], 120), ([4], [4], 4),
                               ([4], [2, 4, 5], 40), ([2, 3, 4], [4], 24)]:
            with self.subTest(a=a, b=b):
                m = model([h.make_node("MatMul", ["a", "b"], ["y"], name="root/block/MatMul")],
                          [vi("a", a), vi("b", b)], [vi("y", None)])
                total = analyze_subgraphs(m)["total"]
                self.assertEqual((total["macs"], total["flops"]), (expected, 2 * expected))

    def test_gemm_transpose_alpha_beta_and_bias(self):
        for beta, expected in [(0., 70), (1., 80), (2., 90)]:
            m = model([h.make_node("Gemm", ["a", "b", "c"], ["y"], name="root/block/Gemm",
                                   transA=1, transB=1, alpha=2., beta=beta)],
                      [vi("a", [3, 2]), vi("b", [5, 3]), vi("c", [5])], [vi("y", None)])
            total = analyze_subgraphs(m)["total"]
            self.assertEqual((total["macs"], total["flops"]), (30, expected))

    def test_broadcast_and_optional_clip_bound(self):
        m = model([h.make_node("Add", ["x", "b"], ["a"], name="root/block/Add"),
                   h.make_node("Clip", ["a", "", "upper"], ["y"], name="root/block/Clip")],
                  [vi("x", [2, 3]), vi("b", [3])], [vi("y", None)],
                  [h.make_tensor("upper", T.FLOAT, [], [6.])])
        total = analyze_subgraphs(m)["total"]
        self.assertEqual(total["flops"], 12)

    def test_pool_softmax_and_normalization(self):
        cases = [
            ("AveragePool", [1, 1, 4, 4], {"kernel_shape": [2, 2], "strides": [2, 2]}, [], 16),
            ("MaxPool", [1, 1, 4, 4], {"kernel_shape": [2, 2], "strides": [2, 2]}, [], 12),
            ("GlobalAveragePool", [1, 1, 4, 4], {}, [], 16),
            ("Softmax", [2, 3], {"axis": -1}, [], 26),
            ("LayerNormalization", [2, 4], {}, [vi("scale", [4]), vi("bias", [4])], 62),
            ("BatchNormalization", [2, 4, 3, 3], {},
             [vi(n, [4]) for n in ("scale", "bias", "mean", "var")], 296),
            ("InstanceNormalization", [2, 4, 3, 3], {}, [vi("scale", [4]), vi("bias", [4])], 528),
        ]
        for op, shape, attrs, extra, expected in cases:
            with self.subTest(op=op):
                m = model([h.make_node(op, ["x"] + [v.name for v in extra], ["y"],
                                       name=f"root/block/{op}", **attrs)],
                          [vi("x", shape), *extra], [vi("y", None)])
                report = analyze_subgraphs(m)
                self.assertEqual(report["total"]["flops"], expected, report["nodes"])

    def test_softmax_opset_axis_semantics(self):
        for opset, expected in [(11, 116), (13, 104)]:
            m = model([h.make_node("Softmax", ["x"], ["y"], name="root/block/Softmax", axis=1)],
                      [vi("x", [2, 3, 4])], [vi("y", None)], opset=opset)
            self.assertEqual(analyze_subgraphs(m)["total"]["flops"], expected)

    def test_dynamic_batch_other_dims_and_stale_shapes(self):
        m = model([h.make_node("Relu", ["x"], ["y"], name="root/block/Relu")],
                  [vi("x", ["batch", 3, "height", 4])], [vi("y", ["batch", 3, "height", 4])])
        raw = m.SerializeToString()
        report = analyze_subgraphs(m, batch_size=1)
        self.assertIsNone(report["total"]["flops"])
        self.assertEqual(report["inputs"][0]["effective_shape"], [1, 3, "height", 4])
        report = analyze_subgraphs(m, input_shapes={"x": [2, 3, 5, 4]})
        self.assertEqual(report["total"]["flops"], 120)
        self.assertEqual(m.SerializeToString(), raw)
        m.graph.input[0].CopyFrom(vi("x", [1, 3]))
        m.graph.output[0].CopyFrom(vi("y", [1, 3]))
        report = analyze_subgraphs(m, input_shapes={"x": [4, 3]})
        self.assertEqual(report["nodes"][0]["outputs"][0]["shape"], [4, 3])
        self.assertEqual(report["total"]["flops"], 12)
        self.assertEqual(analyze_subgraphs(m, batch_size=8)["total"]["flops"], 3)

    def test_unknown_ops_are_not_reported_as_zero(self):
        m = model([h.make_node("Relu", ["x"], ["a"], name="root/block/Relu"),
                   h.make_node("Conv", ["a"], ["b"], domain="custom", name="root/block/Custom"),
                   h.make_node("Identity", ["b"], ["y"], name="root/block/Identity")],
                  [vi("x", [1, 3])], [vi("y", None)])
        report = analyze_subgraphs(m)
        group = report["levels"][0]["subgraphs"][0]
        self.assertEqual(group["known_flops"], 3)
        self.assertIsNone(group["flops"])
        self.assertFalse(group["complete"])
        self.assertEqual(group["unknown_node_ids"], [1])
        self.assertEqual(report["nodes"][2]["flops"], 0)
        self.assertIn("+ ?", format_report(report))
        self.assertIn("custom::Conv", format_report(report))

    def test_integer_arithmetic_is_not_float_flops(self):
        m = model([h.make_node("Add", ["x", "x"], ["y"], name="root/block/Add")],
                  [vi("x", [2], T.INT64)], [vi("y", [2], T.INT64)])
        report = analyze_subgraphs(m)
        self.assertIsNone(report["total"]["flops"])
        self.assertIn("INT64", report["nodes"][0]["reason"])

    def test_configuration_validation(self):
        m = model([], [vi("x", [1])], [vi("x", [1])])
        for kwargs in [{"slash_counts": []}, {"slash_counts": [-1]}, {"batch_size": 0},
                       {"input_shapes": {"missing": [1]}}, {"input_shapes": {"x": [-1]}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_subgraphs(m, **kwargs)

    def test_cli_writes_complete_reports_without_changing_source(self):
        m = model([h.make_node("Relu", ["x"], ["y"], name="root/block/Relu")],
                  [vi("x", ["batch", 3])], [vi("y", None)])
        with tempfile.TemporaryDirectory(prefix="onnx-stats-test-", dir=Path.cwd()) as folder:
            directory = Path(folder).resolve()
            self.assertEqual(directory.parent, Path.cwd().resolve())
            source = directory / "model.onnx"
            onnx.save(m, source)
            raw = source.read_bytes()
            with redirect_stdout(StringIO()):
                main([str(source), "--hierarchy-source", "name", "--slashes", "1", "2", "--no-repair-names",
                      "--no-batch-assumption", "--input-shape", "x=2,3", "--output-dir", str(directory)])
            report = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
            text = (directory / "stats.txt").read_text(encoding="utf-8")
            self.assertEqual(report["total"]["flops"], 6)
            self.assertEqual([level["subgraph_count"] for level in report["levels"]], [1, 1])
            self.assertIn("root/block/Relu", text)
            for depth, prefix in [(1, "root/block"), (2, "root/block/Relu")]:
                path = directory / f"stats_{depth}_slash.csv"
                self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["子图名字"], prefix)
                self.assertEqual(rows[0]["Relu数量"], "1")
                self.assertEqual(rows[0]["FLOPs"], "6")
                self.assertEqual(rows[0]["计算量完整"], "是")
            self.assertEqual(source.read_bytes(), raw)

    def test_csv_operator_columns_unknown_values_and_quoted_names(self):
        prefix = 'root/模块,"一"\n第二行'
        m = model([h.make_node("Relu", ["x"], ["a"], name=prefix + "/Relu"),
                   h.make_node("Custom", ["a"], ["b"], domain="custom", name=prefix + "/Custom"),
                   h.make_node("Identity", ["b"], ["y"], name="root/other/Identity")],
                  [vi("x", [1, 3])], [vi("y", None)])
        report = analyze_subgraphs(m, [1, 2, 5])
        with tempfile.TemporaryDirectory(prefix="onnx-csv-test-", dir=Path.cwd()) as folder:
            directory = Path(folder).resolve()
            self.assertEqual(directory.parent, Path.cwd().resolve())
            paths = write_csv_reports(report, directory)
            self.assertEqual([p.name for p in paths], [f"stats_{k}_slash.csv" for k in (1, 2, 5)])
            all_headers = []
            for level, path in zip(report["levels"], paths):
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    all_headers.append(reader.fieldnames)
                self.assertEqual(len(rows), level["subgraph_count"])
                for group, row in zip(level["subgraphs"], rows):
                    self.assertEqual(row["子图名字"], group["prefix"])
                    self.assertEqual(sum(int(row[f"{op}数量"]) for op in report["total"]["operator_counts"]),
                                     group["node_count"])
                    if not group["complete"]:
                        self.assertEqual([row[key] for key in ("MACs", "FLOPs", "MFLOPs")], ["", "", ""])
                        self.assertEqual(row["计算量完整"], "否")
                        self.assertEqual(row["未知计算量节点数"], "1")
                        self.assertEqual(row["已知FLOPs"], str(group["known_flops"]))
                    else:
                        self.assertEqual(row["FLOPs"], str(group["flops"]))
                if level["slash_count"] == 1:
                    identity = next(row for row in rows if row["子图名字"] == "root/other")
                    self.assertEqual(identity["Relu数量"], "0")
                    self.assertEqual(identity["FLOPs"], "0")
            self.assertEqual(all_headers[0], all_headers[1])
            self.assertEqual(all_headers[0], all_headers[2])


if __name__ == "__main__":
    unittest.main()
