import math
import unittest
from pathlib import Path
from unittest.mock import patch
from onnx_graph import plot_ddr_power as plot

from onnx_graph.plot_ddr_power import prepare_series, power_watts


class PlotPowerTests(unittest.TestCase):
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
