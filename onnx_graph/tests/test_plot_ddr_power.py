import math
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
from onnx_graph import plot_ddr_power as plot

from onnx_graph.plot_ddr_power import prepare_series, power_watts


class PlotPowerTests(unittest.TestCase):
    def test_power_range_uses_same_frequency_and_channel(self):
        groups = {"a": {418: [.1, .5], 749: [.3, math.nan]},
                  "b": {418: [.4, .5], 749: [.7, .2]},
                  "c": {418: [.2, .5]}}
        ranges = plot.power_ranges(groups, 2)
        self.assertAlmostEqual(ranges[418][0], .3)
        self.assertEqual(ranges[418][1], 0)
        self.assertAlmostEqual(ranges[749][0], .4)
        self.assertTrue(math.isnan(ranges[749][1]))
        self.assertTrue(math.isnan(ranges[1066][0]))

    def test_individual_and_overview_share_y_ticks(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("绘图集成测试需要 matplotlib")
        headers = ["OMC名字", "DDR频率_MHz", "DDRPHY"]
        rows = [dict(zip(headers, ("a", 418, .1))), dict(zip(headers, ("b", 418, .8)))]
        axes = []
        def capture(fig, *args, **kwargs):
            axes.extend((ax.get_ylim(), list(ax.get_yticks())) for ax in fig.axes
                        if ax.get_visible() and ax.get_ylabel() == "Power (W)")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, \
                patch.object(plot, "read_table", return_value=(headers, rows)), \
                patch.object(Figure, "savefig", capture):
            files = plot.plot_summary("unused.csv", directory, ["DDRPHY"], y_min=0, y_max=1, y_step=.2)
        self.assertEqual(len(files), 8)
        self.assertTrue(all(limits == (0, 1) for limits, _ in axes))
        self.assertTrue(all(ticks == axes[0][1] for _, ticks in axes))
        self.assertTrue(any(p.name == "all_models_ddr_power.png" for p in files))

    def test_cli_output_directory_is_optional(self):
        with patch("sys.argv", ["plot", "--summary", "example/summary.csv"]), \
                patch.object(plot, "plot_summary", return_value=[Path("example/chart.png"), Path("example/chart.svg")]) as draw, \
                patch("builtins.print"):
            plot.main()
        self.assertIsNone(draw.call_args.args[1])

    def test_units_and_missing(self):
        self.assertAlmostEqual(power_watts("120 mW"), .12)
        self.assertAlmostEqual(power_watts(120, "mW"), .12)
        self.assertAlmostEqual(power_watts("2e3 uW"), .002)
        self.assertTrue(math.isnan(power_watts("")))
        for value in ("NaN", "abc", True, "inf"):
            with self.assertRaises(ValueError):
                power_watts(value)

    def test_aliases_missing_legacy_and_duplicates(self):
        headers = ["OMC相对路径", "DDR频率_MHz", "DDRPHY(PA)"]
        rows = [dict(zip(headers, ("a/model.omc", 749, "20 mW"))),
                dict(zip(headers, ("b/model.omc", 418, ""))),
                dict(zip(headers, ("legacy.omc", "", "2")))]
        groups = prepare_series(headers, rows, ["DDRPHY"])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups["a/model.omc"][749], [.02])
        self.assertTrue(math.isnan(groups["b/model.omc"][418][0]))
        with self.assertRaisesRegex(ValueError, "重复"):
            prepare_series(headers, rows + [rows[0]], ["DDRPHY"])
        with self.assertRaises(ValueError):
            prepare_series(headers, rows, ["missing"])
