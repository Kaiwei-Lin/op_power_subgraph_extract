import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from onnx_graph import extract_prefix, networkx_to_onnx, nodes_by_prefix, omc_to_networkx
from onnx_graph.omc import _cost, _schema
from onnx_graph.stats import analyze_subgraphs, write_csv_reports
from onnx_graph.__main__ import main as graph_cli


def container(model, width=4):
    payloads = [model.SerializeToString(), b"preserved weights", b"preserved compiled tasks"]
    fmt = "<III" if width == 4 else "<QQQ"
    table = struct.pack("<I", len(payloads)) + bytes(width - 4)
    offset = 0
    for kind, payload in enumerate(payloads):
        table += struct.pack(fmt, kind, offset, len(payload))
        offset += len(payload)
    header = bytearray(256)
    header[:4] = b"IMOD"
    struct.pack_into("<II", header, 4, 256, 0x10000000)
    struct.pack_into("<I", header, 76, len(table) + offset)
    return bytes(header) + table + b"".join(payloads)


def tensor(desc, shape=(1, 4)):
    desc.dtype = 1
    desc.shape.dim.extend(shape)
    desc.layout = "NCHW"


class OmcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.Model, cls.Graph, cls.schema_bytes, cls.schema_path = _schema()
        except ValueError as exc:
            raise unittest.SkipTest(f"OMC 集成测试需要 DDK GE schema: {exc}") from exc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test_omc_", dir=Path.cwd())
        self.directory = Path(self.temp.name).resolve()
        self.assertEqual(self.directory.parent, Path.cwd().resolve())
        self.addCleanup(self.temp.cleanup)
        self.path = self.directory / "test.omc"

    def model(self, name="root/block_1/Add"):
        m = self.Model(name="test")
        root = m.graph.add(name="root_graph")
        root.input.append("input:0:0")
        root.output.append("output:0")
        data = root.op.add(name="input:0", type="Data", input=[""])
        tensor(data.output_desc.add())
        wrapper = root.op.add(name="wrapper", type="GraphOp", input=["input:0:0"])
        tensor(wrapper.input_desc.add()); tensor(wrapper.output_desc.add())
        out = root.op.add(name="output", type="NetOutput", input=["wrapper:0"])
        tensor(out.input_desc.add()); tensor(out.output_desc.add())
        child = self.Graph(name="inner")
        child.input.append("input:0:0")
        child.output.append("innerout:0")
        data = child.op.add(name="input:0", type="Data", input=[""])
        tensor(data.output_desc.add())
        op = child.op.add(name=name, type="Add", input=["input:0:0", "input:0:0", "input:0:-1"])
        tensor(op.input_desc.add()); tensor(op.input_desc.add()); tensor(op.output_desc.add())
        op.output_desc[0].name = "/root/block_1/Add_output_0"
        out = child.op.add(name="innerout", type="NetOutput", input=[f"{name}:0"])
        tensor(out.input_desc.add()); tensor(out.output_desc.add())
        wrapper.attr["subgraph"].bt = child.SerializeToString()
        return m

    def save(self, m=None, width=4):
        raw = container(m if m is not None else self.model(), width)
        self.path.write_bytes(raw)
        return raw

    def test_regions_ports_raw_information_and_pickle(self):
        raw = self.save()
        g = omc_to_networkx(self.path)
        self.assertEqual(g.graph["source_bytes"], raw)
        self.assertEqual(len(g), 6)
        self.assertEqual(g.number_of_edges(), 6)
        self.assertEqual(g.graph["regions"][1]["parent_node_id"], 1)
        self.assertTrue(nx.is_directed_acyclic_graph(g))
        add = nodes_by_prefix(g, "/root/block_1/")[0]
        edges = list(g.in_edges(add, data=True))
        self.assertEqual([e[2]["input_index"] for e in edges], [0, 1, 2])
        self.assertEqual([e[2]["output_index"] for e in edges], [0, 0, -1])
        self.assertTrue(edges[-1][2]["control"])
        self.assertEqual({g.nodes[u]["region_id"] for u, _, _ in edges}, {1})
        self.assertEqual(pickle.loads(pickle.dumps(g)).graph["source_bytes"], raw)
        sub = extract_prefix(g, "root/block_1")
        self.assertEqual(len(sub), 1)
        sub.nodes[add]["attributes"]["changed"] = True
        self.assertNotIn("changed", g.nodes[add]["attributes"])
        self.assertEqual(nodes_by_prefix(sub, ""), [add])
        self.assertEqual(nodes_by_prefix(g, "root/block"), [])
        with self.assertRaisesRegex(ValueError, "OMC"):
            networkx_to_onnx(sub)

    def test_repair_names_uses_tensor_path_and_preserves_proto(self):
        self.save(self.model("root_block_1_Add"))
        plain = omc_to_networkx(self.path)
        repaired = omc_to_networkx(self.path, repair_names=True)
        node_id = nodes_by_prefix(repaired, "root/block_1")[0]
        self.assertEqual(repaired.nodes[node_id]["name"], "/root/block_1/Add")
        self.assertEqual(repaired.nodes[node_id]["original_name"], "root_block_1_Add")
        self.assertEqual(plain.nodes[node_id]["proto"], repaired.nodes[node_id]["proto"])
        self.assertEqual(list(plain.edges), list(repaired.edges))
        self.assertEqual(nodes_by_prefix(plain, "root/block_1"), [])

    def test_alias_mapping_and_diagnostics_keep_global_region_ids(self):
        self.save(self.model("legacy_1_Add"))
        graph = omc_to_networkx(self.path, repair_names=True, name_aliases={"legacy_1": "root/block_1"})
        node_id = nodes_by_prefix(graph, "root/block_1")[0]
        self.assertEqual(graph.nodes[node_id]["name"], "/root/block_1/Add")
        self.assertEqual(graph.graph["rename_report"][node_id]["source"], "output_alias")
        self.assertNotIn(node_id, [r["node_id"] for r in graph.graph["name_output_mismatches"]])
        for issue in graph.graph["name_output_mismatches"]:
            self.assertEqual(issue["original_name"], graph.nodes[issue["node_id"]]["original_name"])
            self.assertEqual(issue["region_id"], graph.nodes[issue["node_id"]]["region_id"])
        report = analyze_subgraphs(self.path, name_aliases={"legacy_1": "root/block_1"}, repair_names=True)
        self.assertEqual(report["levels"][0]["subgraphs"][0]["prefix"], "root/block_1")

    def test_outputs_define_hierarchy_without_changing_ge_names_or_dependencies(self):
        raw = self.save(self.model("unrelated_runtime_Add"))
        graph = omc_to_networkx(self.path, hierarchy_source="outputs", repair_names=True)
        node_id = nodes_by_prefix(graph, "root/block_1")[0]
        self.assertEqual(graph.nodes[node_id]["name"], "unrelated_runtime_Add")
        self.assertEqual(graph.graph["source_bytes"], raw)
        self.assertEqual(graph.graph["rename_report"], [])
        self.assertIsNone(graph.graph["name_output_mismatches"])
        self.assertEqual(list(graph.edges), list(omc_to_networkx(self.path).edges))
        self.assertEqual(graph.nodes[0]["hierarchy"], ())
        self.assertEqual(graph.graph["output_hierarchy_issues"][0]["outputs"], [""])
        report = analyze_subgraphs(self.path, [1], hierarchy_source="outputs")
        self.assertEqual(report["levels"][0]["subgraphs"][0]["node_ids"], [node_id])
        self.assertEqual(report["nodes"][0]["name"], "unrelated_runtime_Add")

    def test_empty_boundary_name_is_preserved_and_ignored(self):
        m = self.model()
        m.graph[0].op[-1].name = ""
        m.graph[0].ClearField("output")
        self.save(m)
        g = omc_to_networkx(self.path, repair_names=True)
        self.assertEqual(g.nodes[2]["name"], "")
        self.assertEqual(g.nodes[2]["hierarchy"], ())
        self.assertEqual(g.graph["rename_report"][2]["status"], "skipped_unnamed")
        self.assertEqual(analyze_subgraphs(self.path)["total"]["node_count"], 1)

    def test_profile_csv_and_per_level_exports(self):
        raw = self.save()
        profile = self.directory / "profile.csv"
        with profile.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows([["op name", "total time(us)"], ["root/block_1/Add(Add)", 4],
                              ["root/block_1/Add(Add)", 8]])
        report = analyze_subgraphs(self.path, [1, 2], profile_csv=profile)
        self.assertEqual(report["source_format"], "OMC")
        self.assertEqual(report["total"]["node_count"], 1)
        self.assertEqual(report["total"]["flops"], 4)
        self.assertEqual(report["total"]["exec_time"], 6)
        self.assertAlmostEqual(report["total"]["arithmetic_intensity"], 4 / 6)
        for p in write_csv_reports(report, self.directory / "csv"):
            with p.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["FLOPs"], "4")
            self.assertEqual(float(rows[0]["exec_time"]), 6)
        self.assertNotIn("profile", analyze_subgraphs(self.path))
        self.assertEqual(self.path.read_bytes(), raw)
        with self.assertRaisesRegex(ValueError, "input-shape"):
            analyze_subgraphs(self.path, input_shapes={"input:0": [2, 4]})

    def test_64_bit_partition_table(self):
        self.save(width=8)
        self.assertEqual(len(omc_to_networkx(self.path)), 6)

    def test_convert_cli_accepts_omc(self):
        self.save()
        destination = self.directory / "model.nx.pkl"
        with patch("sys.argv", ["onnx_graph", "convert", str(self.path), str(destination),
                                "--ge-proto", self.schema_path]), redirect_stdout(StringIO()):
            graph_cli()
        with destination.open("rb") as stream:
            self.assertEqual(len(pickle.load(stream)), 6)

    def test_bad_files_are_rejected(self):
        valid = self.save()
        encrypted = bytearray(valid); encrypted[80] = 1
        overlapping = bytearray(valid)
        struct.pack_into("<I", overlapping, 276, 0)
        bad_version = bytearray(valid); struct.pack_into("<I", bad_version, 8, 7)
        for raw in (b"ONNX", valid[:-1], encrypted, overlapping, bad_version):
            with self.subTest(header=bytes(raw[:16])):
                self.path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    omc_to_networkx(self.path)

    def test_ambiguous_or_missing_dependencies_are_rejected(self):
        for name, ref in (("input:0", "input:0:0"), ("wrapper", "absent:0")):
            m = self.model()
            m.graph[0].op[1].name = name
            m.graph[0].op[1].input[0] = ref
            self.save(m)
            with self.assertRaises(ValueError):
                omc_to_networkx(self.path)

    def test_schema_is_read_without_executing_python(self):
        path = self.directory / "ge_ir_pb2.py"
        path.write_text("raise RuntimeError('must not execute')\n"
                        "DESCRIPTOR = pool.AddSerializedFile(" + repr(self.schema_bytes) + ")\n",
                        encoding="utf-8")
        self.save()
        self.assertEqual(len(omc_to_networkx(self.path, ge_proto_path=path)), 6)


class OmcCostTests(unittest.TestCase):
    def graph(self, w, groups, layout="FRACTAL_Z"):
        g = nx.MultiDiGraph()
        g.add_node(0, constant_tensor={"shape": w, "dtype": "DT_FLOAT16", "layout": layout})
        g.add_node(1, op_type="Convolution", attributes={"groups": groups, "relu_flag": True},
                   inputs=("x:0", "w:0", "bias:0"),
                   input_tensors=[{"shape": [1, 4, 4, 4], "dtype": "DT_FLOAT16", "layout": "NC1HWC0"}],
                   output_tensors=[{"shape": [1, 4, 2, 2], "dtype": "DT_FLOAT16", "layout": "NC1HWC0"}])
        g.add_edge(0, 1, input_index=1, control=False)
        return g

    def test_grouped_and_depthwise_weights_and_fused_relu(self):
        cases = [([4, 2, 3, 3], 2, "FRACTAL_Z", 288),
                 ([4, 1, 3, 3], 4, "FRACTAL_Z", 144),
                 ([1, 4, 3, 3], 4, "FRACTAL_Z_DW_V2", 144),
                 ([1, 4, 3, 3], 4, "FRACTAL_Z_DW_DIAG", 144)]
        for w, groups, layout, macs in cases:
            with self.subTest(layout=layout, groups=groups):
                g = self.graph(w, groups, layout)
                self.assertEqual(_cost(g, 1)[:2], (macs, 2 * macs + 32))
                g.nodes[1]["attributes"]["relu_flag"] = False
                self.assertEqual(_cost(g, 1)[1], 2 * macs + 16)

    def test_unknown_physical_quantized_or_inconsistent_shapes_are_not_zero(self):
        for w, groups, layout in (([4, 2, 3, 3, 16], 2, "FRACTAL_Z"),
                                  ([1, 4, 3, 3], 2, "FRACTAL_Z_DW_V2"),
                                  ([4, 2, 3, 3], 2, "unknown_layout")):
            with self.assertRaises(ValueError):
                _cost(self.graph(w, groups, layout), 1)
        g = self.graph([4, 2, 3, 3], 2)
        g.nodes[0]["constant_tensor"]["dtype"] = "DT_INT8"
        with self.assertRaises(ValueError):
            _cost(g, 1)
        g.nodes[1]["op_type"] = "CustomFusedOp"
        with self.assertRaisesRegex(ValueError, "CustomFusedOp"):
            _cost(g, 1)


if __name__ == "__main__":
    unittest.main()
