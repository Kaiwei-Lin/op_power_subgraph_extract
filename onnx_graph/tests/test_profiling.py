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

from onnx_graph import stats
from onnx_graph.tests.test_stats import model, vi


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="onnx-profile-test-", dir=Path.cwd())
        self.directory = Path(folder.name).resolve()
        self.assertEqual(self.directory.parent, Path.cwd().resolve())
        self.addCleanup(folder.cleanup)
        self.model = model([
            h.make_node("Add", ["x", "x"], ["a"], name="root/block/Add"),
            h.make_node("Relu", ["a"], ["y"], name="root/block/Relu"),
        ], [vi("x", [1, 4])], [vi("y", [1, 4])])

    def profile(self, rows, fields=(" op name", " total time(us)")):
        path = self.directory / "profile.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_mean_per_operator_then_sum_and_intensity(self):
        path = self.profile([
            [" root/block/Add(Add)", 10], ["root/block/Relu(Activation)", 20],
            ["root/block/Add(Add)", 30], ["root/block/Relu(Activation)", 40],
            ["transdata_for_nd_0(TransData)", 1000],
        ])
        before = path.read_bytes(), self.model.SerializeToString()
        report = stats.analyze_subgraphs(self.model, [1, 2], profile_csv=path)
        self.assertEqual([n["exec_time"] for n in report["nodes"]], [20., 30.])
        self.assertEqual([n["exec_time_samples"] for n in report["nodes"]], [2, 2])
        self.assertEqual(report["total"]["exec_time"], 50.)
        self.assertAlmostEqual(report["total"]["arithmetic_intensity"], 8 / 50)
        self.assertEqual(report["profile"]["matched_node_count"], 2)
        self.assertEqual(report["profile"]["unmatched_operator_count"], 1)
        self.assertEqual(report["profile"]["sample_counts"], [1, 2])
        for level in report["levels"]:
            self.assertEqual(sum(g["exec_time"] for g in level["subgraphs"]), 50.)
        self.assertEqual((path.read_bytes(), self.model.SerializeToString()), before)
        self.assertIn("50.000000 us", stats.format_report(report))

    def test_multiple_runtime_operators_for_one_node(self):
        path = self.profile([["root/block/Add(Add)", 10], ["root/block/Add(Add)", 30],
                             ["root/block/Add(Extra)", 4], ["root/block/Add(Extra)", 8],
                             ["root/block/Relu(Activation)", 30]])
        report = stats.analyze_subgraphs(self.model, profile_csv=path)
        self.assertEqual(report["nodes"][0]["exec_time"], 26.)
        self.assertEqual(report["nodes"][0]["profile_entries"], [0, 1])
        self.assertEqual(report["total"]["exec_time"], 56.)

    def test_partial_match_leaves_full_time_and_intensity_unknown(self):
        path = self.profile([["root/block/Add(Add)", 12]])
        report = stats.analyze_subgraphs(self.model, profile_csv=path)
        total = report["total"]
        self.assertIsNone(total["exec_time"])
        self.assertEqual(total["known_exec_time"], 12.)
        self.assertIsNone(total["arithmetic_intensity"])
        self.assertEqual(total["exec_time_missing_node_ids"], [1])
        exported = stats.write_csv_reports(report, self.directory)[0]
        with exported.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["exec_time"], "")
        self.assertEqual(row["算术强度"], "")
        self.assertEqual(row["已知exec_time"], "12.0")
        self.assertEqual(row["时间匹配节点数"], "1")
        self.assertEqual(row["时间未匹配节点数"], "1")

    def test_zero_duration_zero_flops_and_unknown_flops(self):
        for op, dtype, duration, expected in [("Identity", T.FLOAT, 2, 0.),
                                              ("Relu", T.FLOAT, 0, None),
                                              ("Add", T.INT64, 2, None)]:
            with self.subTest(op=op):
                args = ["x", "x"] if op == "Add" else ["x"]
                m = model([h.make_node(op, args, ["y"], name="root/block/op")],
                          [vi("x", [1, 4], dtype)], [vi("y", [1, 4], dtype)])
                report = stats.analyze_subgraphs(m, profile_csv=self.profile([["root/block/op", duration]]))
                self.assertEqual(report["total"]["exec_time"], duration)
                self.assertEqual(report["total"]["arithmetic_intensity"], expected)

    def test_original_repaired_and_flattened_names(self):
        original = "mono_model_cfb_d3_LN2_Relu"
        repaired = "/mono_model/cfb_d3/LN2/Relu"
        m = model([h.make_node("Relu", ["x"], [repaired + "_output_0"], name=original)],
                  [vi("x", [1, 4])], [vi(repaired + "_output_0", [1, 4])])
        for profile_name in [original, repaired, repaired.lstrip("/"), "mono_model*cfb_d3*LN2*Relu"]:
            with self.subTest(name=profile_name):
                path = self.profile([[profile_name + "(Activation)", 4]])
                report = stats.analyze_subgraphs(m, repair_names=True, profile_csv=path)
                self.assertEqual(report["nodes"][0]["name"], repaired)
                self.assertEqual(report["total"]["exec_time"], 4.)
                self.assertEqual(report["total"]["arithmetic_intensity"], 1.)

    def test_ambiguous_flattened_and_duplicate_names_are_not_distributed(self):
        for names, runtime in [(["root/a_b/Relu", "root/a/b_Relu"], "root*a*b*Relu(Activation)"),
                               (["root/duplicate", "root/duplicate"], "root/duplicate")]:
            with self.subTest(names=names):
                m = model([h.make_node("Relu", ["x"], ["a"], name=names[0]),
                           h.make_node("Relu", ["a"], ["y"], name=names[1])],
                          [vi("x", [1, 4])], [vi("y", [1, 4])])
                report = stats.analyze_subgraphs(m, profile_csv=self.profile([[runtime, 10]]))
                self.assertEqual(report["profile"]["entries"][0]["status"], "ambiguous")
                self.assertEqual(report["profile"]["entries"][0]["candidate_node_ids"], [0, 1])
                self.assertIsNone(report["total"]["exec_time"])
                self.assertEqual(report["total"]["known_exec_time"], 0.)

    def test_exact_names_with_parentheses_take_precedence(self):
        self.model.graph.node[0].name = "root/block/Add(Cpu)"
        path = self.profile([["root/block/Add(Cpu)", 2], ["root/block/Relu(Activation)", 3]])
        report = stats.analyze_subgraphs(self.model, profile_csv=path)
        self.assertEqual(report["total"]["exec_time"], 5.)
        self.assertEqual(report["profile"]["entries"][0]["match_method"], "exact")

    def test_preannotated_csv_and_missing_time(self):
        path = self.profile([["compiled_a", "root/block/Add", 5], ["", "root/block/Relu", ""]],
                            ["op name", "onnx_node_name", "total time(us)"])
        report = stats.analyze_subgraphs(self.model, profile_csv=path)
        self.assertEqual(report["profile"]["entries"][1]["status"], "missing_time")
        self.assertEqual(report["total"]["known_exec_time"], 5.)
        self.assertIsNone(report["total"]["exec_time"])

    def test_invalid_profile_data_is_rejected(self):
        for value in ["-1", "NaN", "inf", "not-a-time"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "第 2 行"):
                stats.analyze_subgraphs(self.model, profile_csv=self.profile([["root/block/Add", value]]))
        for fields, rows in [(["op name", "total time(ms)"], [["root/block/Add", 1]]),
                             (["op name", "op name"], [["a", "b"]]),
                             (["op name", "total time(us)"], [["a"]]),
                             (["op name", "total time(us)"], [])]:
            with self.subTest(fields=fields, rows=rows), self.assertRaises(ValueError):
                stats.analyze_subgraphs(self.model, profile_csv=self.profile(rows, fields))
        path = self.profile([["a", 1, "model_a"], ["b", 2, "model_b"]],
                            ["op name", "total time(us)", "model name"])
        with self.assertRaisesRegex(ValueError, "多个模型"):
            stats.analyze_subgraphs(self.model, profile_csv=path)

    def test_optional_cli_csv_and_disable_configuration(self):
        source = self.directory / "model.onnx"
        onnx.save(self.model, source)
        profile = self.profile([["root/block/Add(Add)", 2], ["root/block/Relu(Activation)", 3]])
        for enabled in (True, False):
            output = self.directory / str(enabled)
            args = [str(source), "--slashes", "1", "2", "--output-dir", str(output)]
            args += ["--profile-csv", str(profile)] if enabled else ["--no-profile"]
            with patch.object(stats, "PROFILE_CSV", profile), redirect_stdout(StringIO()):
                stats.main(args)
            report = json.loads((output / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual("profile" in report, enabled)
            self.assertEqual("exec_time" in report["total"], enabled)
            for depth in (1, 2):
                with (output / f"stats_{depth}_slash.csv").open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                self.assertEqual("exec_time" in reader.fieldnames, enabled)
                self.assertEqual("算术强度" in reader.fieldnames, enabled)
                if enabled:
                    self.assertEqual({row["exec_time_unit"] for row in rows}, {"us"})
                    self.assertEqual({row["算术强度单位"] for row in rows}, {"FLOPs/us"})
                    for row in rows:
                        self.assertAlmostEqual(float(row["算术强度"]), float(row["FLOPs"]) / float(row["exec_time"]))


if __name__ == "__main__":
    unittest.main()
