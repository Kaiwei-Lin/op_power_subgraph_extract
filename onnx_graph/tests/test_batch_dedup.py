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
from onnx_graph import similarity


class BatchDedupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="onnx-dedup-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)
        self.source = self.root / "source.onnx"
        self.stats = self.root / "stats"
        self.stats.mkdir()
        self.output = self.root / "exports"
        self.prefixes = ["root/one", "root/two", "root/three"]
        model = h.make_model(h.make_graph([
            h.make_node(op, ["x"], [name], name=prefix + "/" + op)
            for op, name, prefix in zip(["Relu", "Relu", "Sigmoid"], ["a", "b", "c"], self.prefixes)
        ], "model", [h.make_tensor_value_info("x", T.FLOAT, [2])],
            [h.make_tensor_value_info(n, T.FLOAT, [2]) for n in ["a", "b", "c"]]),
            opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        onnx.save(model, self.source)
        for depth in (1, 2):
            with (self.stats / f"stats_{depth}_slash.csv").open(
                    "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["子图名字", "节点数", "exec_time"])
                for prefix, op in zip(self.prefixes, ["Relu", "Relu", "Sigmoid"]):
                    writer.writerow([prefix if depth == 1 else prefix + "/" + op, 1, 10])

    def export(self, **kwargs):
        with redirect_stdout(StringIO()):
            return batch.export_from_stats(self.source, self.stats, self.output,
                                           deduplicate=True, **kwargs)

    def read_representatives(self):
        result = []
        for path in (self.output / "grouping_exact").glob("slash_*/representatives.csv"):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                result.extend(Path(r["ONNX文件"]) for r in csv.DictReader(stream))
        return result

    def test_main_keeps_only_representatives_and_records_removed_paths(self):
        original = self.source.read_bytes()
        self.output = self.root / "main" / "batch"
        with patch.multiple(batch, MODEL_PATH=self.source, STATS_DIR=self.stats, FILTERS=[],
                            OUTPUT_DIR=self.root / "main", BATCH_FROM_STATS=True, GENERATE_STATS=False,
                            GROUP_MODE="exact", SEPARATE_SLASH_COUNTS=True), redirect_stdout(StringIO()):
            report = batch.main()
        self.assertEqual(report["counts"], {"exported": 4, "deduplicated": 2})
        self.assertEqual(report["deduplication"]["counts"], {"representative": 4, "removed": 2})
        self.assertEqual(set(self.output.rglob("*.onnx")), set(self.read_representatives()))
        for record in report["records"]:
            self.assertTrue(Path(record["representative_onnx"]).is_file())
            self.assertEqual(Path(record["onnx_file"]).exists(), record["status"] != "deduplicated")
        for path in self.read_representatives():
            onnx.checker.check_model(onnx.load(path), full_check=True)
        saved = json.loads((self.output / "batch_export.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["counts"], report["counts"])
        with (self.output / "batch_export.csv").open(encoding="utf-8-sig", newline="") as stream:
            removed = [r for r in csv.DictReader(stream) if r["状态"] == "deduplicated"]
        self.assertEqual(len(removed), 2)
        self.assertTrue(all(r["去重状态"] == "removed" and Path(r["代表ONNX"]).is_file() for r in removed))
        self.assertEqual(self.source.read_bytes(), original)

    def test_disabled_keeps_all_and_rerun_preserves_representatives(self):
        with patch.multiple(batch, MODEL_PATH=self.source, STATS_DIR=self.stats, FILTERS=[],
                            OUTPUT_DIR=self.root, BATCH_FROM_STATS=True, GENERATE_STATS=False,
                            KEEP_REPRESENTATIVES_ONLY=False), redirect_stdout(StringIO()):
            report = batch.main()
        self.assertEqual(report["counts"], {"exported": 6})
        self.assertNotIn("deduplication", report)
        self.output = self.root / "batch"
        self.export()
        before = {path: path.read_bytes() for path in self.read_representatives()}
        report = self.export()
        self.assertEqual(report["counts"], {"exists": 4, "deduplicated": 2})
        self.assertEqual(set(self.output.rglob("*.onnx")), set(before))
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))

    def test_only_selected_models_participate_and_empty_selection_is_valid(self):
        self.output.mkdir()
        old = self.output / "old_unrelated.onnx"
        old.write_bytes(self.source.read_bytes())
        report = self.export(filters=[[("子图名字", "==", "root/one")]])
        grouped = json.loads(Path(report["deduplication"]["grouping_report"]).read_text(encoding="utf-8"))
        self.assertEqual(grouped["files"], 1)
        self.assertTrue(old.exists())
        kept = {p: p.read_bytes() for p in self.output.rglob("*.onnx")}
        report = self.export(filters=[[("exec_time", ">", 999)]])
        self.assertEqual(report["counts"], {"filtered": 6})
        self.assertEqual(report["deduplication"]["counts"], {})
        self.assertTrue(all(p.read_bytes() == data for p, data in kept.items()))

    def test_failed_comparison_keeps_model_and_explains_why(self):
        report = self.export()
        chosen = next(r for r in report["records"]
                      if r["prefix"] == "root/three" and r["status"] == "exported")
        path = Path(chosen["onnx_file"])
        path.write_bytes(b"existing model cannot be parsed")
        report = self.export()
        row = next(r for r in report["records"] if r["onnx_file"] == str(path))
        self.assertEqual(row["dedup_status"], "kept_unchecked")
        self.assertTrue(row["dedup_reason"])
        self.assertEqual(path.read_bytes(), b"existing model cannot be parsed")
        self.assertEqual(report["deduplication"]["counts"]["kept_unchecked"], 1)

    def test_failed_grouping_or_cleanup_does_not_lose_models(self):
        with patch.object(batch, "group_subgraphs", side_effect=ValueError("grouping failed")):
            report = self.export()
        self.assertEqual(report["deduplication"]["status"], "failed")
        self.assertEqual(len(list(self.output.rglob("*.onnx"))), 6)
        with patch.object(Path, "unlink", side_effect=PermissionError("file locked")):
            report = self.export()
        self.assertEqual(report["deduplication"]["counts"], {"representative": 4, "cleanup_failed": 2})
        self.assertEqual(len(list(self.output.rglob("*.onnx"))), 6)

    def test_outside_or_changed_file_aborts_before_any_removal(self):
        real_group = batch.group_subgraphs
        outside = self.root / "unrelated.onnx"
        outside.write_bytes(self.source.read_bytes())

        def invalid_report(*args, **kwargs):
            grouped = real_group(*args, **kwargs)
            grouped["levels"][0]["members"][0]["representative_path"] = str(outside)
            return grouped

        with patch.object(batch, "group_subgraphs", side_effect=invalid_report):
            report = self.export()
        self.assertEqual(report["deduplication"]["status"], "failed")
        self.assertEqual(len(list(self.output.rglob("*.onnx"))), 6)
        self.assertEqual(outside.read_bytes(), self.source.read_bytes())

        def change_file(*args, **kwargs):
            grouped = real_group(*args, **kwargs)
            Path(grouped["levels"][0]["members"][0]["path"]).write_bytes(b"modified during grouping")
            return grouped

        with patch.object(batch, "group_subgraphs", side_effect=change_file):
            report = self.export()
        self.assertEqual(report["deduplication"]["status"], "failed")
        self.assertIn("发生变化", report["deduplication"]["reason"])
        self.assertEqual(len(list(self.output.rglob("*.onnx"))), 6)

    def test_grouping_subset_rejects_outside_paths_and_allows_empty_list(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            similarity.group_subgraphs(self.output, model_paths=[self.source])
        with redirect_stdout(StringIO()):
            grouped = similarity.group_subgraphs(self.output, model_paths=[])
        self.assertEqual(grouped["files"], 0)
        self.assertEqual(grouped["levels"], [])

    def test_main_passes_similarity_mode_and_cross_level_setting(self):
        with patch.multiple(batch, MODEL_PATH=self.source, STATS_DIR=self.stats, FILTERS=[],
                            OUTPUT_DIR=self.root, BATCH_FROM_STATS=True, GENERATE_STATS=False,
                            KEEP_REPRESENTATIVES_ONLY=True,
                            GROUP_MODE="similar", SIMILARITY_THRESHOLD=1, SEPARATE_SLASH_COUNTS=False
                            ), redirect_stdout(StringIO()):
            report = batch.main()
        self.assertEqual(report["counts"], {"exported": 2, "deduplicated": 4})
        self.assertEqual(report["deduplication"]["mode"], "similar")
        self.assertFalse(report["deduplication"]["separate_slash_counts"])


if __name__ == "__main__":
    unittest.main()
