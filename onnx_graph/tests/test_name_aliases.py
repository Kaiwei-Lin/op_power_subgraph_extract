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

from onnx_graph import check_name_outputs, load_name_aliases, onnx_to_networkx, repair_node_names
from onnx_graph import main as batch, stats
from onnx_graph.__main__ import main as graph_cli


class NameAliasTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="onnx-alias-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)
        self.aliases = {"spNetFC_1": "oSpatialNet_freq_cmps_1"}
        self.alias_file = self.root / "aliases.json"
        self.alias_file.write_text(json.dumps(self.aliases), encoding="utf-8-sig")
        self.old = "spNetFC_1_fconv1_1_conv_Conv"
        self.new = "/oSpatialNet_freq_cmps_1/fconv1.1/conv/Conv"
        self.output = self.new + "_output_0"
        vi = lambda name: h.make_tensor_value_info(name, T.FLOAT, [1, 1, 2, 2])
        self.model = h.make_model(h.make_graph([
            h.make_node("Conv", ["x", "w"], [self.output], name=self.old),
            h.make_node("Relu", [self.output], ["/other/Relu_output_0"], name="unmapped_Relu"),
        ], "aliases", [vi("x")], [vi("/other/Relu_output_0")],
            [h.make_tensor("w", T.FLOAT, [1, 1, 1, 1], [1])]),
            opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        self.source = self.root / "model.onnx"
        onnx.save(self.model, self.source)
        self.profile = self.root / "profile.csv"
        with self.profile.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows([
                ["op name", "total time(us)"],
                ["oSpatialNet_freq_cmps_1*fconv1.1*conv*Conv(Convolution)", 4],
                ["oSpatialNet_freq_cmps_1*fconv1.1*conv*Conv(Convolution)", 8],
                ["unmapped_Relu", 2],
            ])

    def test_json_alias_repairs_user_example_without_changing_tensors_or_source(self):
        original = self.model.SerializeToString()
        plain, _ = repair_node_names(self.model)
        self.assertEqual(plain.graph.node[0].name, self.old)
        fixed, report = repair_node_names(self.model, name_aliases=self.alias_file)
        self.assertEqual(fixed.graph.node[0].name, self.new)
        self.assertEqual(report[0]["source"], "output_alias")
        self.assertEqual(list(fixed.graph.node[0].output), [self.output])
        self.assertEqual(fixed.graph.node[1].input, self.model.graph.node[1].input)
        issues = check_name_outputs(fixed, name_aliases=self.aliases, rename_report=report)
        self.assertEqual([r["node_id"] for r in issues], [1])
        self.assertEqual(issues[0]["name"], "unmapped_Relu")
        self.assertEqual(issues[0]["outputs"], ["/other/Relu_output_0"])
        self.assertEqual(issues[0]["reason"], "no_output_match")
        onnx.checker.check_model(fixed, full_check=True)
        self.assertEqual(self.model.SerializeToString(), original)

    def test_longest_prefix_and_component_boundary_and_no_chained_replacement(self):
        aliases = {"spNetFC": "wrong", **self.aliases, "oSpatialNet_freq_cmps_1": "also_wrong"}
        fixed, _ = repair_node_names(self.model, name_aliases=aliases)
        self.assertEqual(fixed.graph.node[0].name, self.new)
        for name in ("spNetFC_10_fconv1_1_conv_Conv", "outer_" + self.old):
            with self.subTest(name=name):
                self.model.graph.node[0].name = name
                fixed, report = repair_node_names(self.model, name_aliases=self.aliases)
                self.assertEqual(fixed.graph.node[0].name, name)
                self.assertEqual(report[0]["status"], "unresolved")

    def test_alias_still_requires_output_or_input_evidence(self):
        fixed, report = repair_node_names(self.model, name_aliases={"spNetFC_1": "wrong"})
        self.assertEqual(fixed.graph.node[0].name, self.old)
        self.assertEqual(report[0]["status"], "unresolved")
        self.model.graph.node[1].name = "spNetFC_1_fconv1_1_conv_Relu"
        fixed, report = repair_node_names(self.model, name_aliases=self.aliases)
        self.assertEqual(fixed.graph.node[1].name, "/oSpatialNet_freq_cmps_1/fconv1.1/conv/Relu")
        self.assertEqual(report[1]["source"], "io_scope_alias")
        issues = check_name_outputs(fixed, name_aliases=self.aliases, rename_report=report)
        self.assertEqual(issues[0]["original_name"], "spNetFC_1_fconv1_1_conv_Relu")
        self.assertEqual(issues[0]["name"], fixed.graph.node[1].name)
        self.assertEqual(issues[0]["reason"], "no_output_match")

    def test_ambiguity_collision_existing_hierarchy_and_unnamed_diagnostics(self):
        self.model.graph.node[0].output.append("/oSpatialNet/freq_cmps_1/fconv1.1/conv/Conv_output_1")
        fixed, report = repair_node_names(self.model, name_aliases=self.aliases)
        issues = check_name_outputs(fixed, name_aliases=self.aliases, rename_report=report)
        self.assertEqual(report[0]["status"], "ambiguous")
        self.assertEqual(issues[0]["reason"], "ambiguous_outputs")
        self.assertEqual(len(issues[0]["outputs"]), 2)
        self.model.graph.node[0].output.pop()
        self.model.graph.node[1].name = self.new
        fixed, report = repair_node_names(self.model, name_aliases=self.aliases)
        issues = check_name_outputs(fixed, name_aliases=self.aliases, rename_report=report)
        self.assertEqual([r["reason"] for r in issues], ["name_collision", "no_output_match"])
        self.assertEqual(report[1]["status"], "already_hierarchical")
        self.model.graph.node[1].name = ""
        graph = onnx_to_networkx(self.model, repair_names=True, name_aliases=self.alias_file)
        self.assertEqual(graph.graph["name_output_mismatches"], [])

    def test_missing_outputs_are_reported_and_shared_output_stems_are_not_ambiguous(self):
        node = self.model.graph.node[0]
        node.name = self.new
        node.output.append(self.new + "_output_1")
        self.assertEqual([r["node_id"] for r in check_name_outputs(self.model)], [1])
        del node.output[:]
        self.assertEqual(check_name_outputs(self.model)[0]["reason"], "missing_outputs")

    def test_invalid_alias_json_is_rejected(self):
        self.assertEqual(load_name_aliases(None), {})
        self.assertEqual(load_name_aliases(self.alias_file), self.aliases)
        for text in ('[]', 'null', '{"": "x"}', '{"x": "/"}', '{"x": 1}', '{"x": ["y"]}',
                     '{"a": "b", "a": "c"}', '{"a.b": "x", "a_b": "y"}', '{bad'):
            with self.subTest(text=text):
                self.alias_file.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_name_aliases(self.alias_file)
        with self.assertRaises(FileNotFoundError):
            load_name_aliases(self.root / "absent.json")

    def test_profile_uses_same_alias_with_repair_enabled_or_disabled(self):
        for repair in (False, True):
            with self.subTest(repair=repair):
                report = stats.analyze_subgraphs(self.model, repair_names=repair,
                                                 profile_csv=self.profile, name_aliases=self.alias_file)
                self.assertEqual(report["total"]["exec_time"], 8)
                self.assertEqual(report["nodes"][0]["exec_time"], 6)
                self.assertEqual(report["nodes"][0]["name"], self.new if repair else self.old)

    def test_profile_alias_ambiguity_does_not_assign_time(self):
        self.model.graph.node[1].name = "otherLegacy_fconv1_1_conv_Conv"
        aliases = {**self.aliases, "otherLegacy": "oSpatialNet_freq_cmps_1"}
        report = stats.analyze_subgraphs(self.model, repair_names=False,
                                         profile_csv=self.profile, name_aliases=aliases)
        entry = report["profile"]["entries"][0]
        self.assertEqual(entry["status"], "ambiguous")
        self.assertEqual(entry["candidate_node_ids"], [0, 1])
        self.assertEqual(report["total"]["known_exec_time"], 0)

    def test_main_pipeline_prints_and_saves_all_issues_and_preserves_rep_time(self):
        for i in range(4):
            self.model.graph.node.append(h.make_node(
                "Relu", ["x"], [f"/unknown/Relu_{i}_output_0"], name=f"unmapped_{i}"))
        onnx.save(self.model, self.source)
        settings = dict(MODEL_PATH=self.source, OUTPUT_DIR=self.root / "out",
                        STATS_DIR=self.root / "stats", PROFILE_CSV=self.profile,
                        NAME_ALIASES_JSON=self.alias_file, REPAIR_NAMES=True,
                        BATCH_FROM_STATS=True, GENERATE_STATS=True, SLASH_COUNTS=[1, 3],
                        FILTERS=[], KEEP_REPRESENTATIVES_ONLY=True, GROUP_MODE="exact")
        output = StringIO()
        with patch.multiple(batch, **settings), redirect_stdout(output):
            report = batch.main()
        self.assertEqual(report["counts"], {"exported": 2})
        self.assertIn("name: 'unmapped_Relu'", output.getvalue())
        self.assertIn('outputs: ["/other/Relu_output_0"]', output.getvalue())
        self.assertEqual(len(report["name_output_mismatches"]), 5)
        for i in range(4):
            self.assertIn(f"name: 'unmapped_{i}'", output.getvalue())
            self.assertIn(f'outputs: ["/unknown/Relu_{i}_output_0"]', output.getvalue())
        for folder in (settings["STATS_DIR"], settings["OUTPUT_DIR"] / "batch"):
            issues = json.loads((folder / "name_output_mismatches.json").read_text(encoding="utf-8"))
            self.assertEqual(issues, report["name_output_mismatches"])
        with Path(report["representatives_summary"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({float(r["已知exec_time"]) for r in rows}, {6})
        for row in rows:
            self.assertEqual(onnx.load(row["ONNX文件"]).graph.node[0].name, self.new)
        settings.update(GENERATE_STATS=False)
        with patch.multiple(batch, **settings), redirect_stdout(StringIO()) as reused:
            batch.main()
        self.assertIn("unmapped_Relu", reused.getvalue())
        settings.update(BATCH_FROM_STATS=False, PREFIX="oSpatialNet_freq_cmps_1")
        with patch.multiple(batch, **settings), redirect_stdout(StringIO()):
            batch.main()
        self.assertTrue((settings["OUTPUT_DIR"] / "name_output_mismatches.json").is_file())

    def test_cli_alias_flags_and_diagnostic_output(self):
        destination = self.root / "fixed.onnx"
        with patch("sys.argv", ["onnx_graph", "repair-names", str(self.source), str(destination),
                                "--name-aliases", str(self.alias_file)]), redirect_stdout(StringIO()) as output:
            graph_cli()
        self.assertEqual(onnx.load(destination).graph.node[0].name, self.new)
        self.assertEqual(json.loads(output.getvalue())["name_output_mismatches"][0]["name"], "unmapped_Relu")
        with redirect_stdout(StringIO()) as output:
            stats.main([str(self.source), "--name-aliases", str(self.alias_file),
                        "--no-profile", "--output-dir", str(self.root / "cli_stats")])
        self.assertIn("unmapped_Relu", output.getvalue())
        self.assertTrue((self.root / "cli_stats/name_output_mismatches.json").is_file())


if __name__ == "__main__":
    unittest.main()
