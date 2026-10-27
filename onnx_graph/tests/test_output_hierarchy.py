from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator

from onnx_graph import extract_prefix, networkx_to_onnx, nodes_by_prefix, onnx_to_networkx, refresh_graph
from onnx_graph import main as batch, stats
from onnx_graph.__main__ import main as graph_cli


class OutputHierarchyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="onnx-output-test-", dir=Path.cwd())
        self.root = Path(temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(temp.cleanup)
        nodes, outputs, rows = [], [], [["op name", "total time(us)"]]
        for i in (1, 2):
            scope = f"/oSpatialNet_freq_cmps_{i}/fconv1.0"
            first, last = scope + "/Transpose_output_0", scope + "/Relu_output_0"
            name = f"SpNetFC_{i}_fconv1_0_Transpose"
            relu_name = f"wrong/name_{i}/Relu"
            nodes += [h.make_node("Transpose", ["x"], [first], name=name, perm=[1, 0]),
                      h.make_node("Relu", [first], [last], name=relu_name)]
            outputs.append(h.make_tensor_value_info(last, T.FLOAT, [2, 1]))
            rows += [[name + "(Transpose)", i * 10], [name + "(Transpose)", i * 20], [relu_name, 4]]
        # This row must not match the output path instead of the original node name.
        rows.append(["/oSpatialNet_freq_cmps_1/fconv1.0/Transpose", 999])
        self.model = h.make_model(h.make_graph(nodes, "output_hierarchy", [
            h.make_tensor_value_info("x", T.FLOAT, [1, 2])], outputs),
            opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        self.source = self.root / "source.onnx"
        onnx.save(self.model, self.source)
        self.profile = self.root / "profile.csv"
        with self.profile.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        self.prefix = "oSpatialNet_freq_cmps_1/fconv1.0"

    def analyze(self):
        return stats.analyze_subgraphs(self.model, [1, 2], hierarchy_source="outputs",
                                       profile_csv=self.profile, name_aliases=self.root / "unused.json")

    def test_original_names_and_model_survive_output_grouping_pickle_and_export(self):
        original = self.model.SerializeToString()
        graph = onnx_to_networkx(self.model, hierarchy_source="outputs", repair_names=True,
                                 name_aliases=self.root / "unused.json")
        self.assertEqual(graph.graph["rename_report"], [])
        self.assertIsNone(graph.graph["name_output_mismatches"])
        self.assertEqual(graph.graph["output_hierarchy_issues"], [])
        self.assertEqual(graph.nodes[0]["name"], "SpNetFC_1_fconv1_0_Transpose")
        self.assertEqual(graph.nodes[0]["hierarchy_path"], self.prefix + "/Transpose")
        self.assertEqual(nodes_by_prefix(graph, self.prefix), [0, 1])
        self.assertEqual(nodes_by_prefix(graph, "wrong/name_1"), [])
        self.assertEqual(nodes_by_prefix(graph, "oSpatialNet_freq_cmps"), [])
        graph = pickle.loads(pickle.dumps(graph))
        self.assertEqual(networkx_to_onnx(graph).SerializeToString(), original)
        model = networkx_to_onnx(extract_prefix(graph, self.prefix))
        self.assertEqual([n.SerializeToString() for n in model.graph.node],
                         [n.SerializeToString() for n in self.model.graph.node[:2]])
        onnx.checker.check_model(model, full_check=True)
        actual = ReferenceEvaluator(model).run(None, {"x": np.array([[1, -2]], dtype=np.float32)})[0]
        np.testing.assert_array_equal(actual, np.array([[1], [0]], dtype=np.float32))
        self.assertEqual(self.model.SerializeToString(), original)

    def test_output_hierarchy_refreshes_after_edits_without_using_node_name(self):
        graph = onnx_to_networkx(self.model, hierarchy_source="outputs")
        graph.nodes[0]["proto"].name = "/unrelated/name"
        self.assertEqual(nodes_by_prefix(refresh_graph(graph), self.prefix), [0, 1])
        new = "/new/block/Transpose:0"
        graph.nodes[0]["proto"].output[0] = new
        graph.nodes[1]["proto"].input[0] = new
        graph = refresh_graph(graph)
        self.assertEqual(nodes_by_prefix(graph, "new/block"), [0])
        self.assertEqual(nodes_by_prefix(graph, self.prefix), [1])
        self.assertTrue(graph.has_edge(0, 1))

    def test_multi_outputs_common_scope_and_unavailable_paths_are_reported(self):
        pairs = [(["/root/block/Split_output_0", "/root/block/Split_output_1"], ""),
                 (["/root/block/a", "/root/block/b"], "common_output_scope"),
                 (["/one/a", "/two/b"], "conflicting_output_paths"),
                 (["flat_tensor"], "no_output_path"), ([], "missing_outputs"),
                 (["/root/mixed/a", "flat"], "conflicting_output_paths")]
        model = h.make_model(h.make_graph([
            h.make_node("Identity", ["x"], outputs, name=f"name/fallback/{i}")
            for i, (outputs, _) in enumerate(pairs)], "evidence", [], []))
        graph = onnx_to_networkx(model, hierarchy_source="outputs")
        self.assertEqual(nodes_by_prefix(graph, "root/block"), [0, 1])
        self.assertEqual(nodes_by_prefix(graph, "root/block/Split"), [0])
        self.assertEqual(nodes_by_prefix(graph, "root/block/a"), [])
        self.assertEqual(nodes_by_prefix(graph, "name/fallback"), [])
        self.assertEqual(graph.nodes[1]["scope"], "root/block")
        self.assertEqual([r["reason"] for r in graph.graph["output_hierarchy_issues"]],
                         [reason for _, reason in pairs if reason])
        report = stats.analyze_subgraphs(model, [1, 2], hierarchy_source="outputs")
        self.assertEqual(report["levels"][0]["grouped_node_count"], 2)
        self.assertEqual(report["levels"][0]["unscoped_node_ids"], [2, 3, 4, 5])
        self.assertEqual(report["levels"][1]["shallower_node_ids"], [1])
        stats.write_reports(report, self.root / "issues")
        printed = stats.format_report(report)
        for issue in report["output_hierarchy_issues"]:
            self.assertIn(repr(issue["name"]), printed)
            self.assertIn(json.dumps(issue["outputs"]), printed)
        self.assertEqual(json.loads((self.root / "issues/output_hierarchy_issues.json").read_text(encoding="utf-8")),
                         report["output_hierarchy_issues"])

    def test_time_uses_original_names_and_means_not_output_paths_or_aliases(self):
        report = self.analyze()
        self.assertFalse(report["repair_names"])
        self.assertEqual(report["name_aliases"], {})
        self.assertEqual([n["exec_time"] for n in report["nodes"]], [15, 4, 30, 4])
        self.assertEqual(report["profile"]["entries"][-1]["status"], "unmatched")
        self.assertEqual([s["exec_time"] for s in report["levels"][0]["subgraphs"]], [19, 34])
        self.model.graph.node[0].name = ""
        self.model.graph.node[2].name = self.model.graph.node[1].name
        report = self.analyze()
        self.assertEqual(report["levels"][0]["grouped_node_count"], 4)
        self.assertEqual([n["exec_time"] for n in report["nodes"]], [None, None, None, 4])

    def test_main_statistics_filter_dedup_summary_reuse_and_single_prefix(self):
        self.assertEqual(batch.HIERARCHY_SOURCE, "outputs")
        settings = dict(MODEL_PATH=self.source, PROFILE_CSV=self.profile, STATS_DIR=self.root / "stats",
                        OUTPUT_DIR=self.root / "result", HIERARCHY_SOURCE="outputs", REPAIR_NAMES=True,
                        NAME_ALIASES_JSON=self.root / "unused.json", BATCH_FROM_STATS=True, GENERATE_STATS=True,
                        SLASH_COUNTS=[1, 2], FILTERS=[[("exec_time", ">", 18), ("节点数", ">", 1)]],
                        KEEP_REPRESENTATIVES_ONLY=True, GROUP_MODE="exact", SEPARATE_SLASH_COUNTS=True)
        with patch.multiple(batch, **settings), redirect_stdout(StringIO()) as printed:
            report = batch.main()
        self.assertEqual(report["counts"], {"exported": 1, "deduplicated": 1, "filtered": 4})
        self.assertIn("层级来源: outputs", printed.getvalue())
        self.assertIsNone(report["name_output_mismatches"])
        self.assertEqual(report["output_hierarchy_issues"], [])
        with Path(report["representatives_summary"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["Transpose数量"], rows[0]["Relu数量"], float(rows[0]["已知exec_time"])), ("1", "1", 19))
        exported = onnx.load(rows[0]["ONNX文件"])
        self.assertEqual([n.name for n in exported.graph.node], [n.name for n in self.model.graph.node[:2]])
        for key in ("name_output_mismatches", "output_hierarchy_issues"):
            for directory in (settings["STATS_DIR"], settings["OUTPUT_DIR"] / "batch"):
                self.assertEqual(json.loads((directory / f"{key}.json").read_text(encoding="utf-8")), report[key])
        with patch.multiple(batch, **{**settings, "GENERATE_STATS": False}), redirect_stdout(StringIO()):
            reused = batch.main()
        self.assertEqual(reused["counts"], {"exists": 1, "deduplicated": 1, "filtered": 4})
        with patch.multiple(batch, **{**settings, "BATCH_FROM_STATS": False, "PREFIX": self.prefix}), redirect_stdout(StringIO()):
            batch.main()
        self.assertEqual(onnx.load(settings["OUTPUT_DIR"] / "subgraph.onnx").graph.node[0].name,
                         self.model.graph.node[0].name)

    def test_old_csv_and_stale_export_cannot_be_used_for_new_hierarchy(self):
        directory, output = self.root / "stats", self.root / "out"
        stats.write_reports(stats.analyze_subgraphs(self.model, [1]), directory)
        with self.assertRaisesRegex(ValueError, "层级来源"):
            batch.export_from_stats(self.source, directory, output, hierarchy_source="outputs")
        self.assertFalse(output.exists())
        stats.write_reports(self.analyze(), directory)
        with redirect_stdout(StringIO()):
            report = batch.export_from_stats(self.source, directory, output, hierarchy_source="outputs")
        path = Path(report["records"][0]["onnx_file"])
        old = onnx.load(path)
        old.graph.node[0].name = self.prefix + "/Transpose"
        onnx.save(old, path)
        stale = path.read_bytes()
        with redirect_stdout(StringIO()):
            rerun = batch.export_from_stats(self.source, directory, output, hierarchy_source="outputs")
            wrong_mode = batch.export_from_stats(self.source, directory, self.root / "wrong", hierarchy_source="name")
        self.assertEqual(rerun["counts"], {"failed": 1, "exists": 5})
        self.assertIn("OUTPUT_DIR", rerun["records"][0]["reason"])
        self.assertEqual(path.read_bytes(), stale)
        self.assertEqual(wrong_mode["counts"], {"failed": 6})

    def test_cli_and_empty_selection_report_the_hierarchy_source(self):
        destination = self.root / "subgraph.onnx"
        with patch("sys.argv", ["onnx_graph", "extract", str(self.source), str(destination),
                                "--prefix", self.prefix, "--hierarchy-source", "outputs", "--repair-names"]), \
                redirect_stdout(StringIO()) as output:
            graph_cli()
        self.assertEqual(json.loads(output.getvalue())["hierarchy_source"], "outputs")
        self.assertEqual(onnx.load(destination).graph.node[0].name, self.model.graph.node[0].name)
        with redirect_stdout(StringIO()):
            stats.main([str(self.source), "--no-profile", "--output-dir", str(self.root / "stats")])
        data = json.loads((self.root / "stats/stats.json").read_text(encoding="utf-8"))
        self.assertEqual(data["hierarchy_source"], "outputs")
        with redirect_stdout(StringIO()):
            report = batch.export_from_stats(self.source, self.root / "stats", self.root / "empty",
                                             hierarchy_source="outputs", filters=[[("节点数", ">", 100)]])
        self.assertIsNone(report["output_hierarchy_issues"])
        self.assertEqual(report["counts"], {"filtered": 6})

    def test_invalid_hierarchy_source_fails_even_on_an_empty_graph(self):
        model = h.make_model(h.make_graph([], "empty", [], []))
        with self.assertRaisesRegex(ValueError, "hierarchy_source"):
            onnx_to_networkx(model, hierarchy_source="output_typo")


if __name__ == "__main__":
    unittest.main()
