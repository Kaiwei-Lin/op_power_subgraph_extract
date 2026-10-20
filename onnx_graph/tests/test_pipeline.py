from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import helper as h, TensorProto as T

from onnx_graph import main as batch


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="onnx-pipeline-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)
        self.source = self.root / "source.onnx"
        self.stats = self.root / "statistics"
        self.output = self.root / "result"
        self.profile = self.root / "profile.csv"
        nodes = []
        outputs = []
        for block, op in [("one", "Relu"), ("two", "Relu"), ("three", "Sigmoid")]:
            output = f"/root/{block}/{op}_output_0"
            nodes.append(h.make_node(op, ["x"], [output], name=f"root_{block}_{op}"))
            outputs.append(h.make_tensor_value_info(output, T.FLOAT, ["batch", 2]))
        self.model = h.make_model(h.make_graph(nodes, "", [
            h.make_tensor_value_info("x", T.FLOAT, ["batch", 2])], outputs),
            opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        onnx.save(self.model, self.source)
        with self.profile.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["op name", "total time(us)"])
            for node, values in zip(nodes, [(10, 30), (40, 60), (2, 4)]):
                for value in values:
                    writer.writerow([node.name, value])

    def run_main(self, **overrides):
        settings = dict(MODEL_PATH=self.source, STATS_DIR=self.stats, OUTPUT_DIR=self.output,
                        BATCH_FROM_STATS=True, GENERATE_STATS=True, SLASH_COUNTS=[1, 2],
                        PROFILE_CSV=self.profile, REPAIR_NAMES=True, BATCH_SIZE=1, INPUT_SHAPES={},
                        GE_PROTO_PATH=None, KEEP_REPRESENTATIVES_ONLY=True, GROUP_MODE="exact",
                        SIMILARITY_THRESHOLD=.9, SEPARATE_SLASH_COUNTS=True, FILTERS=[])
        settings.update(overrides)
        with patch.multiple(batch, **settings), redirect_stdout(StringIO()):
            return batch.main()

    def read_stats(self):
        return json.loads((self.stats / "stats.json").read_text(encoding="utf-8"))

    def read_summary(self, report):
        with Path(report["representatives_summary"]).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_statistics_profile_filter_export_and_dedup_in_one_run(self):
        original = self.source.read_bytes()
        self.assertFalse(self.stats.exists())
        report = self.run_main(FILTERS=[[("exec_time", ">", 10), ("节点数", ">", 0)]])
        self.assertEqual(report["counts"], {"exported": 2, "deduplicated": 2, "filtered": 2})
        self.assertEqual(len(report["csv_files"]), 2)
        statistics = self.read_stats()
        self.assertTrue((self.stats / "stats.txt").is_file())
        times = {g["prefix"]: g["exec_time"] for g in statistics["levels"][0]["subgraphs"]}
        self.assertEqual(times, {"root/one": 20, "root/two": 50, "root/three": 3})
        self.assertEqual(statistics["total"]["known_exec_time"], 73)
        self.assertEqual(statistics["levels"][0]["subgraphs"][0]["arithmetic_intensity"], .1)
        for depth in (1, 2):
            with (self.stats / f"stats_{depth}_slash.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("exec_time", rows[0])
            self.assertIn("已知exec_time", rows[0])
        models = list((self.output / "batch").rglob("*.onnx"))
        self.assertEqual(len(models), 2)
        summary = self.read_summary(report)
        self.assertEqual({Path(r["ONNX文件"]) for r in summary}, set(models))
        self.assertEqual({r["斜杠数"] for r in summary}, {"1", "2"})
        self.assertEqual({float(r["已知exec_time"]) for r in summary}, {20})
        self.assertEqual({r["exec_time_unit"] for r in summary}, {"us"})
        self.assertEqual({r["同组子图数"] for r in summary}, {"2"})
        for path in models:
            model = onnx.load(path)
            onnx.checker.check_model(model, full_check=True)
            self.assertIn("/root/", model.graph.node[0].name)
        self.assertEqual(self.source.read_bytes(), original)

    def test_only_fresh_csv_files_are_consumed(self):
        self.stats.mkdir()
        stale = [self.stats / f"stats_{depth}_slash.csv" for depth in ("2", "3", "01")]
        for path in stale:
            path.write_bytes(b"old statistics should not be read")
        report = self.run_main(SLASH_COUNTS=[1])
        self.assertEqual([Path(p).name for p in report["csv_files"]], ["stats_1_slash.csv"])
        self.assertEqual(len(report["records"]), 3)
        self.assertEqual({r["slash_count"] for r in report["records"]}, {1})
        self.assertTrue(all(p.read_bytes() == b"old statistics should not be read" for p in stale))

    def test_existing_statistics_mode_does_not_regenerate_or_require_profile(self):
        self.stats.mkdir()
        path = self.stats / "stats_1_slash.csv"
        path.write_text("子图名字,节点数,exec_time\nroot/one,1,500\n", encoding="utf-8-sig")
        original = path.read_bytes()
        with patch.object(batch, "analyze_subgraphs", side_effect=AssertionError("must not regenerate")):
            report = self.run_main(GENERATE_STATS=False, PROFILE_CSV=self.root / "missing.csv",
                                   SLASH_COUNTS=[99], FILTERS=[[("exec_time", ">", 100)]])
        self.assertEqual(report["counts"], {"exported": 1})
        self.assertEqual(path.read_bytes(), original)

    def test_failed_statistics_never_falls_back_to_stale_csv(self):
        self.stats.mkdir()
        path = self.stats / "stats_1_slash.csv"
        path.write_bytes(b"old statistics")
        with patch.object(batch, "export_from_stats", side_effect=AssertionError("must not export")):
            with self.assertRaises(FileNotFoundError):
                self.run_main(PROFILE_CSV=self.root / "missing.csv")
        self.assertEqual(path.read_bytes(), b"old statistics")
        self.assertFalse(self.output.exists())

    def test_shape_settings_only_affect_estimates_and_profile_is_optional(self):
        report = self.run_main(PROFILE_CSV=None, INPUT_SHAPES={"x": [5, 2]}, BATCH_SIZE=3)
        statistics = self.read_stats()
        self.assertEqual(statistics["inputs"][0]["effective_shape"], [5, 2])
        self.assertEqual(statistics["levels"][0]["subgraphs"][0]["flops"], 10)
        self.assertNotIn("profile", statistics)
        self.assertTrue(all(r["已知exec_time"] == r["exec_time"] == r["exec_time_unit"] == ""
                            for r in self.read_summary(report)))
        for path in report["csv_files"]:
            with Path(path).open(encoding="utf-8-sig") as stream:
                self.assertNotIn("exec_time", csv.DictReader(stream).fieldnames)
        for path in (self.output / "batch").rglob("*.onnx"):
            self.assertEqual(onnx.load(path).graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")

    def test_summary_preserves_zero_known_time_and_incomplete_coverage(self):
        last = "/root/one/Sigmoid_output_0"
        self.model.graph.node.append(h.make_node("Sigmoid", [self.model.graph.node[0].output[0]],
                                                [last], name="root_one_Sigmoid"))
        self.model.graph.output[0].name = last
        onnx.save(self.model, self.source)
        with self.profile.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["op name", "total time(us)"])
            writer.writerows([["root_one_Relu", 0], ["root_one_Relu", 0],
                              ["root_two_Relu", 40], ["root_two_Relu", 60],
                              ["root_three_Sigmoid", 2], ["root_three_Sigmoid", 4]])
        summary = self.read_summary(self.run_main(SLASH_COUNTS=[1]))
        row = next(r for r in summary if r["子图名字"] == "root/one")
        self.assertEqual(float(row["已知exec_time"]), 0)
        self.assertEqual(row["exec_time"], "")
        self.assertEqual(row["时间匹配节点数"], "1")
        self.assertEqual(row["时间未匹配节点数"], "1")

    def test_omc_stops_after_statistics(self):
        statistics = batch.analyze_subgraphs(self.source, [1], batch_size=1)
        statistics["source_format"] = "OMC"
        fake_omc = self.root / "compiled.omc"
        fake_omc.write_bytes(b"not parsed in this pipeline routing test")
        with patch.object(batch, "analyze_subgraphs", return_value=statistics) as analyze, patch.object(
                batch, "export_from_stats", side_effect=AssertionError("OMC must not export ONNX")):
            result = self.run_main(MODEL_PATH=fake_omc, GE_PROTO_PATH=self.root / "ge_ir_pb2.py")
        self.assertEqual(result["source_format"], "OMC")
        self.assertEqual(analyze.call_args.kwargs["ge_proto_path"], self.root / "ge_ir_pb2.py")
        self.assertTrue((self.stats / "stats.json").is_file())
        self.assertFalse(self.output.exists())

    def test_custom_and_standard_operators_with_same_name_count_separately(self):
        self.model.graph.node[0].name = "root/standard/Relu"
        self.model.graph.node[1].name = "root/custom/Relu"
        self.model.graph.node[1].domain = "custom"
        self.model.opset_import.append(h.make_opsetid("custom", 1))
        del self.model.graph.node[2:]
        del self.model.graph.output[2:]
        onnx.save(self.model, self.source)
        report = self.run_main(SLASH_COUNTS=[0], PROFILE_CSV=None)
        self.assertEqual(report["counts"], {"exported": 1})
        self.assertEqual(self.read_stats()["total"]["operator_counts"], {"Relu": 1, "custom::Relu": 1})

    def test_invalid_generated_csv_selection_is_rejected(self):
        self.stats.mkdir()
        outside = self.root / "stats_1_slash.csv"
        outside.write_text("子图名字,节点数\nroot/one,1\n", encoding="utf-8")
        for paths in [[outside], [self.stats / "stats_1_slash.csv"], []]:
            with self.subTest(paths=paths), self.assertRaises((ValueError, FileNotFoundError)):
                batch.export_from_stats(self.source, self.stats, self.output, csv_paths=paths)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
