from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from onnx_graph import compiled_stats as compiled
from onnx_graph.omc import _schema
from onnx_graph.tests import test_omc as fixture


class CompiledStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.Model, _, _, cls.schema_path = _schema()
        except ValueError as error:
            raise unittest.SkipTest(f"OMC 测试需要 GE schema: {error}") from error

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="compiled-stat-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)
        self.models = self.root / "omc"
        self.models.mkdir()
        self.output = self.root / "statistics"

    def save_model(self, relative="slash_1/sg_block_abc_fixed_fuse_20261029_120000.omc", *, unknown=False):
        model = self.Model(name="compiled_subgraph")
        graph = model.graph.add(name="root")
        data = graph.op.add(name="input", type="Data", input=[""])
        fixture.tensor(data.output_desc.add())
        a = graph.op.add(name="compiler_fused_A", type="Add", input=["input:0", "input:0"])
        b = graph.op.add(name="compiler_fused_B", type="CustomOp" if unknown else "Add",
                         input=["compiler_fused_A:0", "compiler_fused_A:0"])
        for op in (a, b):
            dims = (-1,) if unknown and op is b else (1, 4)
            fixture.tensor(op.input_desc.add(), dims)
            fixture.tensor(op.input_desc.add(), dims)
            fixture.tensor(op.output_desc.add(), dims)
        net = graph.op.add(name="output", type="NetOutput", input=["compiler_fused_B:0"])
        fixture.tensor(net.input_desc.add())
        fixture.tensor(net.output_desc.add())
        path = self.models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixture.container(model))
        return path

    def summary(self, rows, fields=None):
        path = self.root / "representatives_summary.csv"
        fields = fields or ["斜杠数", "子图名字", "ONNX文件", "节点数", "FLOPs", "已知exec_time"]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, stem="sg_block_abc", **values):
        return {"斜杠数": "1", "子图名字": "original/block", "ONNX文件": f"Z:\\company\\{stem}.onnx",
                "节点数": "99", "FLOPs": "999", "已知exec_time": "12.5", **values}

    def run_statistics(self, summary=None, output=None):
        return compiled.analyze_compiled_subgraphs(self.models, output or self.output,
                                                   summary_csv=summary, ge_proto_path=self.schema_path)

    def test_whole_omc_statistics_work_without_names_and_keep_onnx_columns(self):
        model = self.save_model()
        summary = self.summary([self.row()])
        before = model.read_bytes(), summary.read_bytes()
        result = self.run_statistics(summary)
        self.assertEqual(result["counts"], {"ok": 1})
        row = result["records"][0]
        self.assertEqual((row["节点数"], row["FLOPs"], row["已知exec_time"]), ("99", "999", "12.5"))
        self.assertEqual((row["OMC_节点数"], row["OMC_FLOPs"], row["OMC_Add数量"]), (2, 8, 2))
        self.assertEqual((row["OMC_输入及权重字节数"], row["OMC_输出字节数"], row["OMC_数据转移字节数"]), (64, 32, 96))
        detail = json.loads(Path(row["OMC统计JSON"]).read_text(encoding="utf-8"))
        self.assertEqual(detail["source_format"], "OMC")
        self.assertEqual(detail["compiled_scope"], "whole_omc")
        self.assertEqual(detail["levels"], [])
        self.assertEqual([n["name"] for n in detail["nodes"]], ["compiler_fused_A", "compiler_fused_B"])
        self.assertEqual(len(detail["omc"]["excluded_node_ids"]), 2)
        self.assertEqual(detail["data_transfer"]["数据转移字节数"], 96)
        self.assertIn("96", Path(row["OMC统计TXT"]).read_text(encoding="utf-8"))
        with (self.output / "omc_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(saved[0]["OMC_Add数量"], "2")
        self.assertEqual((model.read_bytes(), summary.read_bytes()), before)

    def test_unknown_shapes_and_operators_keep_unknown_and_known_parts(self):
        self.save_model(unknown=True)
        row = self.run_statistics()["records"][0]
        self.assertEqual(row["OMC状态"], "ok")
        self.assertIsNone(row["OMC_FLOPs"])
        self.assertEqual(row["OMC_已知FLOPs"], 4)
        self.assertEqual(row["OMC_未知计算量节点数"], 1)
        self.assertIsNone(row["OMC_数据转移字节数"])
        self.assertEqual(row["OMC_已知数据转移字节数"], 48)
        self.assertEqual(row["OMC_未知数据转移节点数"], 1)
        with (self.output / "omc_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))[0]
        self.assertEqual(saved["OMC_FLOPs"], "")
        self.assertEqual(saved["OMC_数据转移字节数"], "")

    def test_missing_ambiguous_failed_and_unused_files_do_not_hide_other_models(self):
        self.save_model()
        self.save_model("slash_1/sg_dup_fuse_20261029_120000.omc")
        self.save_model("slash_1/sg_dup_fuse_20261029_130000.omc")
        orphan = self.save_model("slash_1/sg_orphan.omc")
        bad = self.models / "slash_1/sg_bad.omc"
        bad.write_bytes(b"not an OMC")
        summary = self.summary([self.row(), self.row("sg_missing"), self.row("sg_dup"), self.row("sg_bad")])
        result = self.run_statistics(summary)
        self.assertEqual(result["counts"], {"ok": 1, "missing": 1, "ambiguous": 1, "failed": 1})
        self.assertEqual(result["unused_omc_files"], [str(orphan)])
        for row in result["records"][1:]:
            self.assertTrue(row["OMC原因"])
            self.assertEqual(row["OMC_Add数量"], "")

    def test_slash_level_and_explicit_paths_resolve_compiler_name_ambiguity(self):
        chosen = self.save_model()
        self.save_model("slash_2/sg_block_abc_fixed_fuse_20261029_120000.omc")
        summary = self.summary([self.row()])
        result = self.run_statistics(summary)
        self.assertEqual(result["records"][0]["OMC文件"], str(chosen))
        second = self.save_model("slash_1/sg_block_abc_fixed_fuse_20261029_130000.omc")
        summary = self.summary([self.row(OMC文件=str(second.relative_to(self.models)))],
                               ["斜杠数", "子图名字", "ONNX文件", "节点数", "FLOPs", "已知exec_time", "OMC文件"])
        result = self.run_statistics(summary, self.root / "explicit")
        self.assertEqual(result["counts"], {"ok": 1})
        self.assertEqual(result["records"][0]["OMC文件"], str(second))

    def test_duplicate_mapping_and_output_overwrite_are_rejected(self):
        self.save_model()
        summary = self.summary([self.row(), self.row()])
        self.assertEqual(self.run_statistics(summary)["counts"], {"ambiguous": 2})
        before = (self.output / "omc_summary.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_statistics(summary)
        self.assertEqual((self.output / "omc_summary.csv").read_bytes(), before)

    def test_cli_without_summary_handles_nested_omc_and_missing_inputs(self):
        self.save_model()
        with redirect_stdout(StringIO()) as printed:
            result = compiled.main(["--omc-dir", str(self.models), "--output-dir", str(self.output),
                                    "--ge-proto", self.schema_path])
        self.assertEqual(result["counts"], {"ok": 1})
        self.assertIn("omc_summary.csv", printed.getvalue())
        with self.assertRaises(NotADirectoryError):
            compiled.analyze_compiled_subgraphs(self.root / "missing", self.root / "new")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            compiled.analyze_compiled_subgraphs(empty, self.root / "new")


if __name__ == "__main__":
    unittest.main()
