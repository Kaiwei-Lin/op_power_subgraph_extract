import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from onnx_graph import plot_channel_throughput as plot


class ChannelThroughputTests(unittest.TestCase):
    def table(self):
        headers = ["OMC名字", "DDR频率_MHz", "OMC_Add数量", "数据移动吞吐", "数据移动吞吐单位", "CPU", "DDRPHY"]
        values = [["a", 418, 2, 2e9, "B/s", .2, "100 mW"],
                  ["b", 749, 3, 1e9, "B/s", .4, None],
                  ["c", "", 1, 0, "B/s", 0, .3],
                  ["d", 418, 1, "", "B/s", .5, .6]]
        return headers, [dict(zip(headers, row)) for row in values]

    def test_auto_channels_exclude_stats_and_keep_zero_and_unlabelled_frequency(self):
        headers, rows = self.table()
        points = plot.channel_points(headers, rows)
        self.assertEqual(list(points), ["CPU", "DDRPHY"])
        self.assertEqual(points["CPU"], [(2, .2), (1, .4), (0, 0)])
        self.assertEqual(points["DDRPHY"], [(2, .1), (0, .3)])

    def test_subset_frequency_and_unit_validation(self):
        headers, rows = self.table()
        points = plot.channel_points(headers, rows, ["DDRPHY"], ddr_frequency=418)
        self.assertEqual(points, {"DDRPHY": [(2, .1)]})
        with self.assertRaisesRegex(ValueError, "单位冲突"):
            plot.channel_points(headers, rows, throughput_unit="GB/s")
        with self.assertRaises(ValueError):
            plot.channel_points(headers, rows, ddr_frequency=math.nan)
        with self.assertRaises(ValueError):
            plot.channel_points(headers, rows, ["missing"])
        rows[0]["数据移动吞吐单位"] = ""
        with self.assertRaisesRegex(ValueError, "单位不明确"):
            plot.channel_points(headers, rows)
        rows[0]["数据移动吞吐"] = "inf"
        with self.assertRaises(ValueError):
            plot.channel_points(headers, rows, throughput_unit="B/s")

    def test_no_frequency_summary_and_annotated_xlsx_headers(self):
        headers = ["模型", "数据移动吞吐率(B/s)", "CPU(rail)", "DDRPHY[W]"]
        rows = [dict(zip(headers, ["a", 1e9, .1, .2]))]
        points = plot.channel_points(headers, rows)
        self.assertEqual(points, {"CPU": [(1, .1)], "DDRPHY": [(1, .2)]})
        with self.assertRaisesRegex(ValueError, "频率列"):
            plot.channel_points(headers, rows, ddr_frequency=418)

    def test_single_figure_scatter_units_and_default_directory(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("需要 matplotlib")
        headers, rows = self.table()
        captured = []
        def capture(fig, *args, **kwargs):
            self.assertEqual(len(fig.axes), 1)
            ax = fig.axes[0]
            self.assertEqual(len(ax.lines), 0)
            self.assertEqual(ax.get_ylabel(), "Average Power (mW)")
            captured.append([collection.get_offsets().tolist() for collection in ax.collections])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "summary.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            original = source.read_bytes()
            with patch.object(Figure, "savefig", capture):
                paths = plot.plot_summary(source)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(all(path.parent == source.parent for path in paths))
            self.assertEqual([p.suffix for p in paths], [".png", ".pdf"])
        self.assertEqual(captured[0][0], [[0, 0], [1, 400], [2, 200]])
        self.assertEqual(captured[0][1], [[0, 300], [2, 100]])
