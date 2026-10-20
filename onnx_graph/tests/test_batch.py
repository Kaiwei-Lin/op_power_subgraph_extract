"""CSV 条件筛选、批量导出及逐行错误隔离。"""

from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h
from onnx.reference import ReferenceEvaluator

from onnx_graph import main as batch


def vi(name):
    return h.make_tensor_value_info(name, T.FLOAT, [2])


class BatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="onnx-batch-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)
        self.stats = self.root / "stats"
        self.stats.mkdir()
        self.output = self.root / "exports"
        self.source = self.root / "source.onnx"
        self.model = h.make_model(h.make_graph([
            h.make_node("Add", ["x", "w"], ["a"], name="root/block_1/Add"),
            h.make_node("Relu", ["a"], ["b"], name="root/block_1/Relu"),
            h.make_node("Mul", ["b", "w"], ["y"], name="root/block_10/Mul"),
        ], "", [vi("x")], [vi("y")],
            [h.make_tensor("w", T.FLOAT, [2], [1, 2])]),
            opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        onnx.save(self.model, self.source)
        self.headers = ["子图名字", "节点数", "Add数量", "Relu数量", "Mul数量",
                        "exec_time", "已知exec_time", "计算量完整"]
        self.rows = [
            ["root/block_1", 2, 1, 1, 0, 10, 10, "是"],
            ["root/block_10", 1, 0, 0, 1, "", 7, "否"],
        ]

    def write_csv(self, depth=1, rows=None, headers=None):
        path = self.stats / f"stats_{depth}_slash.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.headers if headers is None else headers)
            writer.writerows(self.rows if rows is None else rows)
        return path

    def export(self, **kwargs):
        self.stdout = StringIO()
        with redirect_stdout(self.stdout):
            return batch.export_from_stats(self.source, self.stats, self.output, **kwargs)

    def test_filter_strict_comparisons_and_unknown_values(self):
        row = {"exec_time": "1e2", "节点数": "5", "计算量完整": "是"}
        for comparison, expected in [(">", False), (">=", True), ("<", False),
                                     ("<=", True), ("==", True), ("!=", False)]:
            with self.subTest(comparison=comparison):
                groups = batch._prepare_filters([[("exec_time", comparison, 100)]])
                self.assertEqual(batch._matches_filters(row, groups), expected)
                self.assertFalse(batch._matches_filters({**row, "exec_time": ""}, groups))
        self.assertTrue(batch._matches_filters(row, []))
        self.assertTrue(batch._matches_filters(row, batch._prepare_filters([
            [("计算量完整", "==", "是"), ("节点数", ">", 4), ("节点数", "<", 6)]])))

    def test_filter_and_or_and_partial_known_time(self):
        row = {"exec_time": "", "已知exec_time": "12.5", "节点数": "8"}
        self.assertFalse(batch._matches_filters(row, batch._prepare_filters([
            [("exec_time", ">", 10), ("节点数", ">", 5)]])))
        self.assertTrue(batch._matches_filters(row, batch._prepare_filters([
            [("exec_time", ">", 10)], [("节点数", ">", 5)]])))
        self.assertTrue(batch._matches_filters(row, batch._prepare_filters([
            [("已知exec_time", ">", 10), ("节点数", ">", 5)]])))

    def test_bad_filter_configuration_and_nonfinite_values_fail(self):
        for filters in [[[]], [[("exec_time", ">", "10")]], [[("节点数", "~", 1)]],
                        [[("节点数", ">", float("nan"))]], [[("节点数",)]],
                        [[("", ">", 1)]]]:
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                batch._prepare_filters(filters)
        for value in ["NaN", "Infinity", "bad"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                batch._matches_filters({"exec_time": value},
                                       batch._prepare_filters([[("exec_time", ">", 0)]]))

    def test_main_exports_all_levels_preserves_boundaries_and_values(self):
        self.write_csv()
        self.write_csv(2, [
            ["root/block_1/Add", 1, 1, 0, 0, 5, 5, "是"],
            ["root/block_1/Relu", 1, 0, 1, 0, 5, 5, "是"],
            ["root/block_10/Mul", 1, 0, 0, 1, "", 7, "否"],
        ])
        # 只读取指定目录，不混入其他统计运行的子目录。
        nested = self.stats / "another_run"
        nested.mkdir()
        (nested / "stats_1_slash.csv").write_text("invalid", encoding="utf-8")
        original = self.source.read_bytes()
        with patch.multiple(batch, MODEL_PATH=self.source, STATS_DIR=self.stats,
                            OUTPUT_DIR=self.output, FILTERS=[], REPAIR_NAMES=False,
                            BATCH_FROM_STATS=True, GENERATE_STATS=False), redirect_stdout(StringIO()):
            report = batch.main()
        self.assertEqual(report["counts"], {"exported": 5})
        for record in report["records"]:
            exported = onnx.load(record["onnx_file"])
            onnx.checker.check_model(exported, full_check=True)
            self.assertTrue(exported.graph.name)
            self.assertEqual(len(exported.graph.node), int(record["statistics"]["节点数"]))
            self.assertEqual(Path(record["onnx_file"]).parent.name, f"slash_{record['slash_count']}")
        chosen = next(r for r in report["records"] if r["prefix"] == "root/block_1")
        exported = onnx.load(chosen["onnx_file"])
        self.assertEqual([n.name for n in exported.graph.node],
                         ["root/block_1/Add", "root/block_1/Relu"])
        self.assertEqual([v.name for v in exported.graph.input], ["x"])
        self.assertEqual([v.name for v in exported.graph.output], ["b"])
        self.assertEqual([t.name for t in exported.graph.initializer], ["w"])
        result = ReferenceEvaluator(exported).run(None, {"x": np.array([2, -4], np.float32)})
        np.testing.assert_array_equal(result[0], np.array([3, 0], np.float32))
        self.assertEqual(self.source.read_bytes(), original)
        manifest = self.output / "batch" / "batch_export.csv"
        self.assertTrue(manifest.read_bytes().startswith(b"\xef\xbb\xbf"))
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 5)

    def test_batch_and_or_filters(self):
        self.write_csv()
        report = self.export(filters=[[("exec_time", ">=", 10), ("节点数", ">", 1)]])
        self.assertEqual(report["counts"], {"exported": 1, "filtered": 1})
        self.output = self.root / "or_exports"
        report = self.export(filters=[[("exec_time", ">", 100)], [("节点数", ">", 0)]])
        self.assertEqual(report["counts"], {"exported": 2})

    def test_empty_selection_still_writes_manifest(self):
        self.write_csv()
        with patch.object(batch, "onnx_to_networkx", side_effect=AssertionError("unneeded load")):
            report = self.export(filters=[[("exec_time", ">", 100)]])
        self.assertEqual(report["counts"], {"filtered": 2})
        self.assertFalse(list(self.output.rglob("*.onnx")))
        self.assertIn("本次未生成新的 ONNX", self.stdout.getvalue())
        self.assertIn("[filtered]", self.stdout.getvalue())
        self.assertIn("exec_time", report["records"][0]["reason"])
        self.assertIn("100", report["records"][0]["reason"])
        self.assertIn("<空>", report["records"][1]["reason"])
        saved = json.loads((self.output / "batch_export.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["counts"], report["counts"])

    def test_empty_csv_reports_no_subgraph_rows(self):
        self.write_csv(rows=[])
        report = self.export()
        self.assertEqual(report["counts"], {})
        self.assertFalse(list(self.output.rglob("*.onnx")))
        self.assertIn("CSV 没有子图数据行", self.stdout.getvalue())

    def test_invalid_rows_and_wrong_model_statistics_are_reported(self):
        self.write_csv(rows=[
            ["", 1, 0, 1, 0, 1, 1, "是"],
            ["root/missing", 1, 0, 1, 0, 1, 1, "是"],
            ["root/block_1", 99, 1, 1, 0, 1, 1, "是"],
            ["root/block_1/Add", 1, 1, 0, 0, 1, 1, "是"],
            ["root/block_10", 1, 1, 0, 0, 1, 1, "是"],
            *self.rows,
        ])
        report = self.export()
        self.assertEqual(report["counts"], {"failed": 5, "exported": 2})
        self.assertTrue(all(r["reason"] for r in report["records"][:5]))
        self.assertIn("[failed] root/missing", self.stdout.getvalue())
        self.assertIn("REPAIR_NAMES", self.stdout.getvalue())

    def test_duplicate_prefix_and_existing_file_are_preserved(self):
        self.write_csv(rows=[self.rows[0], ["/root//block_1/", *self.rows[0][1:]]])
        report = self.export()
        self.assertEqual(report["counts"], {"exported": 1, "duplicate": 1})
        target = Path(report["records"][0]["onnx_file"])
        target.write_bytes(b"user-owned existing file")
        report = self.export()
        self.assertEqual(report["counts"], {"exists": 1, "duplicate": 1})
        self.assertEqual(target.read_bytes(), b"user-owned existing file")

    def test_missing_filter_column_and_malformed_csv_fail_before_export(self):
        self.write_csv(headers=["子图名字", "节点数"], rows=[["root/block_1", 2]])
        with self.assertRaisesRegex(ValueError, "exec_time"):
            self.export(filters=[[("exec_time", ">", 10)]])
        self.assertFalse(self.output.exists())
        self.write_csv(headers=["子图名字", "节点数", "节点数"], rows=[])
        with self.assertRaisesRegex(ValueError, "重复"):
            self.export()
        self.write_csv(headers=["子图名字", "节点数"], rows=[["root/block_1"]])
        with self.assertRaisesRegex(ValueError, "字段数量"):
            self.export()

    def test_no_csv_and_omc_input_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            self.export()
        with self.assertRaisesRegex(ValueError, "原始 .onnx"):
            batch.export_from_stats(self.root / "source.omc", self.stats, self.output)

    def test_repair_names_matches_statistics_prefix(self):
        model = h.make_model(h.make_graph([
            h.make_node("Relu", ["x"], ["/mono_model/cfb_d3/LN2/Relu_output_0"],
                        name="mono_model_cfb_d3_LN2_Relu"),
        ], "", [vi("x")], [vi("/mono_model/cfb_d3/LN2/Relu_output_0")]),
            opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        onnx.save(model, self.source)
        self.write_csv(headers=["子图名字", "节点数"], rows=[["/mono_model/cfb_d3", 1]])
        self.assertEqual(self.export(repair_names=False)["counts"], {"failed": 1})
        report = self.export(repair_names=True)
        self.assertEqual(report["counts"], {"exported": 1})
        exported = onnx.load(report["records"][0]["onnx_file"])
        self.assertEqual(exported.graph.node[0].name, "/mono_model/cfb_d3/LN2/Relu")
        # 使用两脚本的真实默认设置，防止统计时修复、导出时不修复导致零导出。
        from onnx_graph import stats
        stats.write_csv_reports(
            stats.analyze_subgraphs(self.source, [1], repair_names=stats.REPAIR_NAMES), self.stats)
        with patch.multiple(batch, MODEL_PATH=self.source, STATS_DIR=self.stats,
                            OUTPUT_DIR=self.root / "default_main", FILTERS=[],
                            BATCH_FROM_STATS=True, GENERATE_STATS=False), redirect_stdout(StringIO()):
            default_report = batch.main()
        self.assertEqual(default_report["counts"], {"exported": 1})
        onnx.checker.check_model(onnx.load(default_report["records"][0]["onnx_file"]), full_check=True)

    def test_safe_filenames_distinguish_sanitized_names(self):
        prefixes = ["root/a:b", "root/a*b", "../a:b"]
        for node, prefix in zip(self.model.graph.node, prefixes):
            node.name = prefix + "/" + node.op_type
        onnx.save(self.model, self.source)
        self.write_csv(headers=["子图名字", "节点数"], rows=[[p, 1] for p in prefixes])
        report = self.export()
        self.assertEqual(report["counts"], {"exported": 3})
        paths = [Path(r["onnx_file"]) for r in report["records"]]
        self.assertEqual(len(set(paths)), 3)
        for path in paths:
            self.assertEqual(path.parent, self.output / "slash_1")
            self.assertNotIn(":", path.name)
            self.assertNotIn("*", path.name)

    def test_unknown_boundary_failure_does_not_stop_other_subgraphs(self):
        model = h.make_model(h.make_graph([
            h.make_node("Unknown", ["x"], ["a"], name="root/unknown/Op", domain="custom"),
            h.make_node("Identity", ["a"], ["y"], name="outside"),
            h.make_node("Relu", ["x"], ["z"], name="root/good/Relu"),
        ], "test", [vi("x")], [vi("y"), vi("z")]),
            opset_imports=[h.make_opsetid("", 13), h.make_opsetid("custom", 1)], ir_version=8)
        onnx.save(model, self.source)
        self.write_csv(headers=["子图名字", "节点数"],
                       rows=[["root/unknown", 1], ["root/good", 1]])
        report = self.export()
        self.assertEqual(report["counts"], {"failed": 1, "exported": 1})
        self.assertIn("Missing boundary", report["records"][0]["reason"])
        self.assertFalse(Path(report["records"][0]["onnx_file"]).exists())


if __name__ == "__main__":
    unittest.main()
