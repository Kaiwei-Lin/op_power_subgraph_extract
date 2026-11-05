import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from onnx_graph import plot_ddr_runtime as plot


class DdrRuntimeTests(unittest.TestCase):
    def table(self):
        headers = ["OMC相对路径", "DDR频率_MHz", "运行时长", "运行时长单位"]
        values = [["a/model.omc", 749, .02, "s"], ["a/model.omc", 418, .03, "s"],
                  ["a/model.omc", 1066, None, "s"], ["b/model.omc", 418, 0, "s"],
                  ["legacy.omc", "", 999, "s"]]
        return headers, [dict(zip(headers, row)) for row in values]

    def test_grouping_seconds_to_ms_missing_and_true_zero(self):
        headers, rows = self.table()
        groups = plot.prepare_runtime_series(headers, rows)
        self.assertEqual(set(groups), {"a/model.omc", "b/model.omc"})
        self.assertEqual(groups["a/model.omc"][418], 30)
        self.assertEqual(groups["a/model.omc"][749], 20)
        self.assertTrue(math.isnan(groups["a/model.omc"][1066]))
        self.assertEqual(groups["b/model.omc"][418], 0)

    def test_input_units_and_conflicts(self):
        headers, rows = self.table()
        rows = [rows[0]]
        rows[0]["运行时长单位"] = "us"
        rows[0]["运行时长"] = 1500
        self.assertEqual(plot.prepare_runtime_series(headers, rows)["a/model.omc"][749], 1.5)
        with self.assertRaisesRegex(ValueError, "单位冲突"):
            plot.prepare_runtime_series(headers, rows, "s")
        rows[0]["运行时长单位"] = ""
        with self.assertRaisesRegex(ValueError, "单位不明确"):
            plot.prepare_runtime_series(headers, rows)
        self.assertEqual(plot.prepare_runtime_series(headers, rows, "ms")["a/model.omc"][749], 1500)
        headers = ["OMC名字", "DDR频率 MHz", "执行时间(ms)"]
        rows = [dict(zip(headers, ["model", 418, 2.5]))]
        self.assertEqual(plot.prepare_runtime_series(headers, rows)["model"][418], 2.5)

    def test_invalid_and_duplicate_rows_are_rejected(self):
        headers, rows = self.table()
        with self.assertRaisesRegex(ValueError, "重复"):
            plot.prepare_runtime_series(headers, rows + [rows[0]])
        for value in (-1, "NaN", "inf", True, "invalid"):
            with self.subTest(value=value):
                row = dict(rows[0], 运行时长=value)
                with self.assertRaises(ValueError):
                    plot.prepare_runtime_series(headers, [row])
        with self.assertRaises(ValueError):
            plot.prepare_runtime_series(headers, [dict(rows[0], DDR频率_MHz=123)])
        with self.assertRaisesRegex(ValueError, "没有有效"):
            plot.prepare_runtime_series(headers, [dict(rows[0], 运行时长="")])

    def test_adaptive_individual_axes_and_combined_model_lines(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("需要 matplotlib")
        headers, rows = self.table()
        figures = []
        def capture(fig, *args, **kwargs):
            ax = fig.axes[0]
            self.assertEqual(ax.get_ylabel(), "Execution time (ms)")
            for line in ax.lines:
                self.assertEqual(list(line.get_xdata()), list(plot.FREQUENCIES))
            self.assertEqual(list(ax.get_xticks()), list(plot.FREQUENCIES))
            self.assertEqual(kwargs["dpi"], 300)
            figures.append(([list(line.get_ydata()) for line in ax.lines], ax.get_ylim(),
                            ax.get_legend_handles_labels()[1]))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "summary.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            with patch.object(Figure, "savefig", capture):
                files = plot.plot_summary(source)
            self.assertEqual(len(files), 6)
            self.assertTrue(all(p.parent == source.parent for p in files))
            self.assertEqual(len({p.name for p in files}), 6)
            self.assertEqual(files[-2].name, "all_models_ddr_runtime.png")
            self.assertEqual(files[-1].name, "all_models_ddr_runtime.pdf")
            self.assertEqual(source.read_bytes(), before)
        self.assertEqual(figures[0][0][0][:2], [30, 20])
        self.assertTrue(math.isnan(figures[0][0][0][2]))
        self.assertNotEqual(figures[0][1], figures[2][1])
        self.assertGreater(figures[0][1][0], 0)
        self.assertEqual(len(figures[-1][0]), 2)
        self.assertEqual(figures[-1][2], ["a/model.omc", "b/model.omc"])
        self.assertEqual(figures[-1][0][0][:2], [30, 20])
        self.assertTrue(math.isnan(figures[-1][0][0][2]))
        self.assertEqual(figures[-1][0][1][0], 0)

    def test_cli_output_directory_is_optional(self):
        with patch("sys.argv", ["plot", "--summary", "summary.csv"]), \
                patch.object(plot, "plot_summary", return_value=[Path("figure.png"), Path("figure.pdf"),
                                                               Path("all_models_ddr_runtime.png"),
                                                               Path("all_models_ddr_runtime.pdf")]) as draw, \
                patch("builtins.print"):
            plot.main()
        self.assertIsNone(draw.call_args.args[1])
