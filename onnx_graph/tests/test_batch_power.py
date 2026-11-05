from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from onnx_graph import merge_power as merge
from onnx_graph.compiled_stats import number_models
from onnx_graph.omc import _schema
from onnx_graph.tests import test_compiled_stats as fixtures


class BatchPowerTests(unittest.TestCase):
    save_model = fixtures.CompiledStatisticsTests.save_model

    @classmethod
    def setUpClass(cls):
        try:
            cls.Model, _, _, cls.schema_path = _schema()
        except ValueError as error:
            raise unittest.SkipTest(str(error)) from error

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="batch-power-test-", dir=Path.cwd())
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(self.temp.cleanup)
        self.models = self.root / "batch"
        self.models.mkdir()
        self.powers = self.root / "power"
        self.powers.mkdir()
        self.output = self.root / "result"

    def power_csv(self, number, values=None, db="Work0.db", encoding="utf-8-sig"):
        values = values or {"CPU": "0", "NPU": "3.25 mW"}
        path = self.powers / f"{number}.csv"
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metadata"])
            writer.writerow(["DB", "Inst", "Channel", "Current", "Voltage", "Power"])
            writer.writerows([[db, "inst", c, "unused", "unused", v] for c, v in values.items()])
        return path

    def run_merge(self, **kwargs):
        return merge.merge_compiled_power(self.models, self.powers, self.output,
                                           ge_proto_path=self.schema_path, **kwargs)

    def detailed_xlsx(self, name, headers=None, rows=None):
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("详细 XLSX 功耗测试需要 openpyxl")
        book = Workbook()
        sheet = book.active
        sheet.title = "Work1.db"
        sheet.append(headers or ["Timestamp[us]", " VBUCK0_PERI [W]", "DDRPHY[W]", "CPU[A]", "CPU[V]"])
        for values in (rows if rows is not None else [[1_000_000, 0, .1, 999, 888],
                                                     [1_500_000, .2, .3, 777, 666],
                                                     [3_000_000, .4, .5, 555, 444]]):
            sheet.append(values)
        other = book.create_sheet("Average Values Work0.db")
        other.append(["Timestamp[us]", "VBUCK0_PERI[W]", "DDRPHY[W]"])
        other.append([0, 999, 888])
        path = self.powers / name
        book.save(path)
        book.close()
        return path

    def test_detailed_xlsx_first_sheet_means_and_us_duration(self):
        self.save_model("model.omc")
        path = self.detailed_xlsx("1_db.xlsx")
        before = path.read_bytes()
        row = self.run_merge()["records"][0]
        self.assertEqual(row["功耗状态"], "ok")
        self.assertAlmostEqual(row["VBUCK0_PERI"], .2)
        self.assertAlmostEqual(row["DDRPHY"], .3)
        self.assertNotIn("CPU", row)
        self.assertNotIn("DDR频率_MHz", row)
        self.assertEqual((row["运行时长"], row["数据移动吞吐"]), (2, 48))
        self.assertEqual(path.read_bytes(), before)
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved["VBUCK0_PERI"], "0.2")

    def test_detailed_all_samples_channels_and_partial_empty_cells(self):
        headers = ["Time(us)", *[f"CH_{i}[W]" for i in range(14)], "Other[mW]", "Other[V]"]
        rows = [[1_000_000 + n * 25_000, *[n if i else None for i in range(14)], 999, 999] for n in range(30)]
        rows[0][1] = 0
        path = self.detailed_xlsx("1_db.xlsx", headers, rows)
        values, power_error, seconds, runtime_error = merge.read_detailed_power(path)
        self.assertEqual((power_error, runtime_error), ("", ""))
        self.assertEqual(len(values), 14)
        self.assertEqual(values["CH_0"], 0)
        self.assertEqual(values["CH_13"], 14.5)
        self.assertEqual(seconds, .725)

    def test_detailed_ddr_matches_exact_model_and_frequency_and_mixes_old_format(self):
        for i in range(10):
            self.save_model(f"m_{i:02}.omc")
        self.detailed_xlsx("1-1db.xlsx")
        self.detailed_xlsx("1-9_DB.xlsx", rows=[[0, .4, .2], [1_000_000, .8, .4]])
        self.detailed_xlsx("10_db.xlsx")
        self.power_csv(2, {"VBUCK0_PERI": "0.2", "DDRPHY": "0.3"})
        (self.powers / "1-10db.xlsx").write_bytes(b"invalid frequency")
        (self.powers / "99_db.xlsx").write_bytes(b"unknown model")
        result = self.run_merge()
        records = {(r["编号"], r["DDR频点编号"]): r for r in result["records"]}
        self.assertEqual(records[1, 1]["DDR频率_MHz"], 418)
        self.assertEqual(records[1, 9]["DDR频率_MHz"], 4253)
        self.assertEqual(records[1, 1]["功耗状态"], "ok")
        self.assertEqual(records[1, 9]["VBUCK0_PERI"], .6)
        self.assertEqual(records[10, None]["功耗状态"], "ok")
        self.assertEqual(records[2, None]["VBUCK0_PERI"], "0.2")
        self.assertNotIn((1, None), records)
        self.assertEqual({Path(p).name for p in result["unused_power_files"]}, {"1-10db.xlsx", "99_db.xlsx"})

    def test_detailed_takes_priority_over_old_same_measurement(self):
        self.save_model("model.omc")
        detailed = self.detailed_xlsx("1_db.xlsx")
        old = self.power_csv(1, {"VBUCK0_PERI": "99", "DDRPHY": "88"})
        before = old.read_bytes()
        result = self.run_merge()
        row = result["records"][0]
        self.assertEqual((row["功耗状态"], row["时长状态"]), ("ok", "ok"))
        self.assertEqual(row["功耗文件"], str(detailed))
        self.assertEqual(row["VBUCK0_PERI"], .2)
        self.assertEqual(row["运行时长"], 2)
        self.assertEqual(result["ignored_power_files"], [str(old)])
        self.assertEqual(old.read_bytes(), before)

    def test_detailed_ddr_priority_and_multiple_detailed_ambiguity(self):
        self.save_model("model.omc")
        self.detailed_xlsx("1-1db.xlsx")
        self.power_csv("1-1", {"VBUCK0_PERI": "99", "DDRPHY": "88"})
        self.power_csv("1-2", {"VBUCK0_PERI": "11", "DDRPHY": "22"})
        result = self.run_merge()
        self.assertEqual(result["records"][0]["VBUCK0_PERI"], .2)
        self.assertEqual(result["records"][1]["VBUCK0_PERI"], "11")
        self.assertEqual([Path(p).name for p in result["ignored_power_files"]], ["1-1.csv"])
        self.detailed_xlsx("1-1_db.xlsx")
        row = self.run_merge()["records"][0]
        self.assertEqual((row["功耗状态"], row["时长状态"]), ("ambiguous", "ambiguous"))
        self.assertIn("1-1db.xlsx", row["功耗原因"])
        self.assertIn("1-1_db.xlsx", row["功耗原因"])

    def test_failed_detailed_does_not_fall_back_to_old_power(self):
        self.save_model("model.omc")
        path = self.detailed_xlsx("1_db.xlsx", rows=[[0, "bad", .1], [1_000_000, .2, .3]])
        old = self.power_csv(1)
        result = self.run_merge()
        row = result["records"][0]
        self.assertEqual(row["功耗状态"], "failed")
        self.assertEqual(row["功耗文件"], str(path))
        self.assertNotIn("CPU", row)
        self.assertEqual(result["ignored_power_files"], [str(old)])

    def test_detailed_timestamp_errors_do_not_discard_valid_power(self):
        self.save_model("model.omc")
        for times in ([2, 1], [1, 1], [None, 2], ["NaN", 2], [0]):
            with self.subTest(times=times):
                self.detailed_xlsx("1_db.xlsx", rows=[[t, .2, .3] for t in times])
                row = self.run_merge()["records"][0]
                self.assertEqual(row["功耗状态"], "ok")
                self.assertEqual(row["VBUCK0_PERI"], .2)
                self.assertEqual(row["时长状态"], "failed")
                self.assertIsNone(row["运行时长"])
                self.assertIsNone(row["数据移动吞吐"])

    def test_detailed_power_errors_do_not_discard_valid_duration(self):
        self.save_model("model.omc")
        for value in ("NaN", "inf", "bad", True, "#DIV/0!", None):
            with self.subTest(value=value):
                self.detailed_xlsx("1_db.xlsx", rows=[[0, value, .3], [1_000_000, value, .3]])
                row = self.run_merge()["records"][0]
                self.assertEqual(row["功耗状态"], "failed")
                self.assertEqual(row["时长状态"], "ok")
                self.assertEqual(row["运行时长"], 1)
        for headers in (["Time", "CPU[A]"], ["Time", "CPU[W]", "CPU [W]"], ["Time", "[W]"]):
            with self.subTest(headers=headers):
                path = self.detailed_xlsx("1_db.xlsx", headers=headers, rows=[[0, 1, 2], [1, 3, 4]])
                with self.assertRaises(ValueError):
                    merge.read_detailed_power(path)

    def set_csv_runtime(self, path, value):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        rows[0] = ["metadata", "", "", "", "", "", "not the duration", value]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_numbering_uses_global_case_insensitive_names_then_relative_paths(self):
        names = ["slash_2/sg_a.omc", "slash_10/sg_a.omc", "slash0/sg_Gemm.omc",
                 "slash0/sg_encoder.omc", "slash0/sg_decoder.omc", "slash_1/sg_a.omc"]
        for name in names:
            self.save_model(name)
        rows = number_models(self.models)
        self.assertEqual([r["OMC相对路径"] for r in rows], [names[i] for i in (5, 1, 0, 4, 3, 2)])
        self.assertEqual([r["编号"] for r in rows], list(range(1, 7)))

    def test_global_filename_order_maps_power_across_directories(self):
        self.save_model("a/b.omc")
        self.save_model("b/a.omc")
        self.power_csv(1, {"NPU": "11"})
        self.power_csv(2, {"NPU": "22"})
        result = self.run_merge()
        self.assertEqual([(r["编号"], r["OMC相对路径"], r["NPU"]) for r in result["records"]],
                         [(1, "b/a.omc", "11"), (2, "a/b.omc", "22")])
        self.assertIn("全局", result["numbering_rule"])
        with (self.output / "numbering.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        saved[0]["编号"], saved[1]["编号"] = "2", "1"
        with (self.output / "numbering.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(saved[0]))
            writer.writeheader()
            writer.writerows(saved)
        reused = merge.merge_compiled_power(self.models, self.powers, self.root / "reused",
                                            numbering_csv=self.output / "numbering.csv")
        self.assertEqual([(r["OMC相对路径"], r["NPU"]) for r in reused["records"]],
                         [("a/b.omc", "11"), ("b/a.omc", "22")])
        self.assertEqual(reused["numbering_rule"], "复用指定编号表")

    def test_generic_batch_does_not_use_fixed_legacy_map_and_keeps_power_units(self):
        self.save_model("slash_0/sg_decoder.omc")
        self.save_model("slash_1/sg_encoder.omc")
        first = self.power_csv(1)
        second = self.power_csv(2, {"NPU": "2.5 mW", "CPU": "7"})
        before = first.read_bytes(), second.read_bytes()
        with patch.object(merge, "MODEL_FILES", {}):
            result = self.run_merge()
        self.assertEqual(result["omc_counts"], {"ok": 2})
        self.assertEqual(result["power_counts"], {"ok": 2})
        self.assertEqual(result["records"][0]["CPU"], "0")
        self.assertEqual(result["records"][0]["NPU"], "3.25 mW")
        self.assertEqual(result["records"][1]["NPU"], "2.5 mW")
        for row in result["records"]:
            self.assertEqual((row["OMC_节点数"], row["OMC_Add数量"], row["OMC_FLOPs"], row["OMC_数据转移字节数"]), (2, 2, 8, 96))
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["编号"] for r in rows], ["1", "2"])
        self.assertEqual((first.read_bytes(), second.read_bytes()), before)
        self.assertNotIn("DDR频率_MHz", result["records"][0])
        self.assertNotIn("DDR频率_MHz", rows[0])

    def test_ddr_sweep_maps_all_frequencies_and_reuses_model_statistics(self):
        self.save_model("b/b.omc")
        self.save_model("a/a.omc")
        for model in (1, 2):
            for index in range(1, 10):
                path = self.power_csv(f"{model}-{index}", {"NPU": str(model * 100 + index)})
                self.set_csv_runtime(path, str(index / 2))
        from onnx_graph import compiled_stats
        with patch.object(compiled_stats, "analyze_compiled_subgraphs",
                          wraps=compiled_stats.analyze_compiled_subgraphs) as analyze:
            result = self.run_merge()
        analyze.assert_called_once()
        self.assertEqual(result["omc_counts"], {"ok": 2})
        self.assertEqual(result["power_counts"], {"ok": 18})
        self.assertEqual(result["runtime_counts"], {"ok": 18})
        self.assertEqual(result["unused_power_files"], [])
        for row, (model, index) in zip(result["records"],
                                      [(m, i) for m in (1, 2) for i in range(1, 10)]):
            self.assertEqual((row["编号"], row["DDR频点编号"], row["DDR频率_MHz"]),
                             (model, index, (418, 749, 1066, 1536, 2102, 2746, 3187, 3744, 4253)[index - 1]))
            self.assertEqual(row["OMC名字"], "a.omc" if model == 1 else "b.omc")
            self.assertEqual(row["NPU"], str(model * 100 + index))
            self.assertEqual((row["OMC_Add数量"], row["OMC_FLOPs"], row["OMC_数据转移字节数"]), (2, 8, 96))
            self.assertEqual((row["cube_FLOPs"], row["vector_FLOPs"]), (0, 8))
            self.assertEqual(row["cube_FLOPs"] + row["vector_FLOPs"], row["OMC_FLOPs"])
            self.assertEqual(row["运行时长"], index / 2)
            self.assertAlmostEqual(row["数据移动吞吐"], 192 / index)
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), 18)
        self.assertEqual(saved[-1]["DDR频率_MHz"], "4253")
        with (self.output / "numbering.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_ddr_mixed_legacy_missing_ambiguous_and_invalid_files(self):
        self.save_model("model.omc")
        legacy = self.power_csv(1, {"NPU": "10"})
        self.set_csv_runtime(legacy, "2")
        first = self.power_csv("1-1", {"NPU": "11"})
        self.set_csv_runtime(first, "1")
        (self.powers / "1-1.xlsx").write_bytes(b"ambiguous")
        (self.powers / "1-2.csv").write_text("invalid table", encoding="utf-8")
        (self.powers / "1-10.csv").write_text("invalid frequency", encoding="utf-8")
        self.power_csv("99-1")
        result = self.run_merge()
        self.assertEqual(result["power_counts"], {"ok": 1, "ambiguous": 1, "failed": 1, "missing": 7})
        old, first, invalid, missing = result["records"][:4]
        self.assertIsNone(old["DDR频率_MHz"])
        self.assertEqual((old["NPU"], old["运行时长"]), ("10", 2))
        self.assertEqual(first["时长状态"], "ambiguous")
        self.assertEqual(invalid["功耗状态"], "failed")
        self.assertEqual(missing["功耗状态"], "missing")
        self.assertNotIn("NPU", missing)
        self.assertIsNone(missing["数据移动吞吐"])
        self.assertEqual({Path(p).name for p in result["unused_power_files"]}, {"1-10.csv", "99-1.csv"})

    def test_ddr_cli_can_generate_missing_frequency_rows_without_power(self):
        self.save_model("model.omc")
        args = ["merge_power", "--batch-dir", str(self.models), "--ddr-sweep", "--output-dir", str(self.output)]
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            self.assertEqual(merge.main(), 0)
        result = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(result["records"]), 9)
        self.assertEqual(result["power_counts"], {"missing": 9})

    def test_saved_numbering_validates_hashes_inventory_and_ids(self):
        model = self.save_model("slash_0/sg_a.omc")
        result = merge.merge_compiled_power(self.models, None, self.output, ge_proto_path=self.schema_path)
        self.assertEqual(result["power_counts"], {"missing": 1})
        numbering = self.output / "numbering.csv"
        self.assertEqual(number_models(self.models, numbering)[0]["编号"], 1)
        self.power_csv(1)
        merged = merge.merge_compiled_power(self.models, self.powers, self.root / "with_power",
                                            ge_proto_path=self.schema_path, numbering_csv=numbering)
        self.assertEqual(merged["power_counts"], {"ok": 1})
        raw = model.read_bytes()
        model.write_bytes(raw + b"changed")
        with self.assertRaisesRegex(ValueError, "内容或层级"):
            number_models(self.models, numbering)
        model.write_bytes(raw)
        self.save_model("slash_0/sg_new.omc")
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            number_models(self.models, numbering)

    def test_missing_failed_ambiguous_and_incomplete_power_are_visible(self):
        for i in range(5):
            self.save_model(f"slash_0/sg_{i}.omc")
        self.power_csv(1, {"CPU": "0", "NPU": "1"})
        self.power_csv(2, {"CPU": "2"})
        (self.powers / "3.csv").write_text("not a power table", encoding="utf-8")
        self.power_csv(4)
        (self.powers / "4.xlsx").write_bytes(b"both formats")
        self.power_csv(99)
        result = self.run_merge()
        self.assertEqual(result["power_counts"], {"ok": 1, "incomplete": 1, "failed": 1, "ambiguous": 1, "missing": 1})
        self.assertEqual(result["records"][-1]["OMC状态"], "ok")
        self.assertNotIn("CPU", result["records"][-1])
        self.assertEqual(len(result["unused_power_files"]), 1)

    def test_csv_prefers_work0_and_rejects_duplicate_channels(self):
        path = self.power_csv(1, {"CPU": "2"}, db="Work1.db", encoding="gb18030")
        with path.open("a", encoding="gb18030", newline="") as stream:
            csv.writer(stream).writerows([["DB", "Inst", "Channel", "Current", "Voltage", "Power"],
                                         ["Work0.db", "inst", "CPU", "", "", "0 毫瓦"]])
        self.assertEqual(merge.read_power_csv(path), {"CPU": "0 毫瓦"})
        path.write_text("Channel,Power\nCPU,1\nCPU,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "重复"):
            merge.read_power_csv(path)

    def test_csv_uses_first_twelve_channel_rows_like_the_old_workbook_reader(self):
        path = self.power_csv(1, {f"CH_{i}": str(i) for i in range(13)})
        powers = merge.read_power_csv(path)
        self.assertEqual(len(powers), 12)
        self.assertNotIn("CH_12", powers)

    def test_h1_runtime_and_transfer_throughput_keep_zero_and_missing_distinct(self):
        for i in range(3):
            self.save_model(f"slash_0/sg_{i}.omc")
            path = self.power_csv(i + 1)
            if i < 2:
                self.set_csv_runtime(path, "0.5" if i == 0 else "0")
        result = self.run_merge()
        self.assertEqual(result["runtime_counts"], {"ok": 2, "missing": 1})
        first, zero, missing = result["records"]
        self.assertEqual(first["运行时长"], 0.5)
        self.assertEqual(first["数据移动吞吐"], 192)
        self.assertEqual((first["运行时长单位"], first["数据移动吞吐单位"]), ("s", "B/s"))
        self.assertEqual(zero["运行时长"], 0)
        self.assertIsNone(zero["数据移动吞吐"])
        self.assertIsNone(missing["运行时长"])
        self.assertIsNone(missing["数据移动吞吐"])
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(saved[0]["数据移动吞吐"], "192.0")
        self.assertEqual(saved[1]["数据移动吞吐"], "")
        self.assertEqual(saved[2]["运行时长"], "")

    def test_bad_runtime_does_not_discard_power_and_reads_physical_first_csv_row(self):
        self.save_model("slash_0/sg_a.omc")
        path = self.power_csv(1)
        for value in ("-1", "NaN", "inf", "invalid", "1e999", "1e-999"):
            with self.subTest(value=value):
                self.set_csv_runtime(path, value)
                with self.assertRaises(ValueError):
                    merge.read_runtime_seconds(path)
        result = self.run_merge()
        row = result["records"][0]
        self.assertEqual(row["功耗状态"], "ok")
        self.assertEqual(row["NPU"], "3.25 mW")
        self.assertEqual(row["时长状态"], "failed")
        self.assertIsNone(row["数据移动吞吐"])
        self.set_csv_runtime(path, "1e-3")
        self.assertEqual(merge.read_runtime_seconds(path), .001)
        content = path.read_text(encoding="utf-8-sig")
        path.write_text("\n" + content, encoding="utf-8-sig")
        self.assertIsNone(merge.read_runtime_seconds(path))

    def test_incomplete_transfer_is_not_used_for_throughput(self):
        self.save_model(unknown=True)
        path = self.power_csv(1)
        self.set_csv_runtime(path, "2")
        row = self.run_merge()["records"][0]
        self.assertEqual(row["运行时长"], 2)
        self.assertIsNone(row["OMC_数据转移字节数"])
        self.assertGreater(row["OMC_已知数据转移字节数"], 0)
        self.assertIsNone(row["cube_FLOPs"])
        self.assertIsNone(row["vector_FLOPs"])
        self.assertEqual((row["已知cube_FLOPs"], row["已知vector_FLOPs"]), (0, 4))
        self.assertIsNone(row["数据移动吞吐"])

    def test_cube_vector_summary_columns_keep_nonzero_matrix_and_scalar_work(self):
        model = self.Model(name="cube_vector")
        graph = model.graph.add(name="root")
        for name, dims in (("x", (2, 3)), ("w", (3, 4))):
            data = graph.op.add(name=name, type="Data")
            fixtures.fixture.tensor(data.output_desc.add(), dims)
        matmul = graph.op.add(name="matmul", type="MatMul", input=["x:0", "w:0"])
        fixtures.fixture.tensor(matmul.input_desc.add(), (2, 3))
        fixtures.fixture.tensor(matmul.input_desc.add(), (3, 4))
        fixtures.fixture.tensor(matmul.output_desc.add(), (2, 4))
        add = graph.op.add(name="add", type="Add", input=["matmul:0", "matmul:0"])
        for desc in (add.input_desc.add(), add.input_desc.add(), add.output_desc.add()):
            fixtures.fixture.tensor(desc, (2, 4))
        (self.models / "matrix.omc").write_bytes(fixtures.fixture.container(model))
        self.power_csv(1)
        row = self.run_merge()["records"][0]
        self.assertEqual((row["OMC_FLOPs"], row["cube_FLOPs"], row["vector_FLOPs"]), (56, 48, 8))
        self.assertEqual((row["已知cube_FLOPs"], row["已知vector_FLOPs"]), (48, 8))
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual((saved["cube_FLOPs"], saved["vector_FLOPs"]), ("48", "8"))

    def test_xlsx_runtime_uses_h1_from_the_selected_average_sheet(self):
        try:
            from openpyxl import Workbook, load_workbook
        except ImportError:
            self.skipTest("XLSX 时长测试需要 openpyxl")
        self.save_model("slash_0/sg_a.omc")
        path = self.powers / "1.xlsx"
        book = Workbook()
        book.active.title = "other"
        book.active["H1"] = 999
        sheet = book.create_sheet(merge.SHEET_NAME)
        sheet["H1"] = 4
        sheet.append(["DB", "Inst", "Channel", "Current", "Voltage", "Power"])
        sheet.append(["Work0.db", "inst", "CPU", 1, 1, 2.5])
        book.save(path)
        book.close()
        row = self.run_merge()["records"][0]
        self.assertEqual((row["运行时长"], row["数据移动吞吐"], row["CPU"]), (4, 24, 2.5))
        book = load_workbook(path)
        book[merge.SHEET_NAME]["H1"] = "#VALUE!"
        book.save(path)
        book.close()
        with self.assertRaises(ValueError):
            merge.read_runtime_seconds(path)

    def test_cli_and_existing_output_are_safe(self):
        self.save_model("slash_0/sg_a.omc")
        self.power_csv(1)
        args = ["merge_power", "--batch-dir", str(self.models), "--power-dir", str(self.powers),
                "--output-dir", str(self.output), "--ge-proto", self.schema_path]
        with patch("sys.argv", args), redirect_stdout(StringIO()) as printed:
            self.assertEqual(merge.main(), 0)
        self.assertIn("numbering.csv", printed.getvalue())
        stale = self.output / "stale" / "old.txt"
        stale.parent.mkdir()
        stale.write_text("old", encoding="utf-8")
        self.run_merge(numbering_csv=self.output / "numbering.csv")
        self.assertFalse(stale.exists())
        self.assertTrue((self.output / "summary.csv").is_file())
        for target in (self.models, self.powers, self.root):
            with self.assertRaises(ValueError):
                merge.merge_compiled_power(self.models, self.powers, target)
        self.assertTrue((self.powers / "1.csv").is_file())


if __name__ == "__main__":
    unittest.main()
