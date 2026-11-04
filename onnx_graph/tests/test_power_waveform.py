import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from onnx_graph import plot_power_waveform as plot


class PowerWaveformTests(unittest.TestCase):
    def setUp(self):
        if plot.load_workbook is None:
            self.skipTest("需要 openpyxl")
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "1-1db.xlsx"

    def save(self, rows=None, headers=None):
        from openpyxl import Workbook
        book = Workbook()
        sheet = book.active
        sheet.title = "First"
        sheet.append(headers or ["Time[us]", "DDRPHY[A]", " DDRPHY [W]", "CPU[W]"])
        for row in (rows if rows is not None else [[1_000_000, "ignored", .1, .2],
                                                   [1_500_000, 999, None, .3],
                                                   [3_000_000, 888, .3, .4]]):
            sheet.append(row)
        other = book.create_sheet("Second")
        other.append(["Time[us]", "DDRPHY[W]"])
        other.append([0, 999])
        book.active = 1
        book.save(self.source)
        book.close()

    def test_first_sheet_matching_channel_elapsed_time_and_gaps(self):
        self.save()
        before = self.source.read_bytes()
        times, powers = plot.read_waveform(self.source, "DDRPHY")
        self.assertEqual(times, [0, 500_000, 2_000_000])
        self.assertEqual((powers[0], powers[2]), (.1, .3))
        self.assertTrue(math.isnan(powers[1]))
        self.assertEqual(plot.read_waveform(self.source, "DDRPHY [W]")[0], times)
        self.assertEqual(self.source.read_bytes(), before)

    def test_invalid_timestamps_and_power_are_reported(self):
        for times in ([1, 1], [2, 1], [None, 2], ["NaN", 2]):
            with self.subTest(times=times):
                self.save([[t, 1, .1] for t in times])
                with self.assertRaises(ValueError):
                    plot.read_waveform(self.source, "DDRPHY")
        for power in (None, "NaN", "bad", True, "#DIV/0!"):
            with self.subTest(power=power):
                self.save([[0, 1, power], [1, 2, power]])
                with self.assertRaises(ValueError):
                    plot.read_waveform(self.source, "DDRPHY")

    def test_missing_and_duplicate_channel_names(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "可选 Channel"):
            plot.read_waveform(self.source, "missing")
        self.save([[0, .1, .2]], ["Time[us]", "DDRPHY[W]", "DDRPHY [W]"])
        with self.assertRaisesRegex(ValueError, "唯一"):
            plot.read_waveform(self.source, "DDRPHY")

    def test_plot_uses_ms_mw_and_original_samples(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("需要 matplotlib")
        self.save()
        before = self.source.read_bytes()
        calls = []
        def capture(fig, *args, **kwargs):
            ax = fig.axes[0]
            calls.append((ax.get_xlabel(), ax.get_ylabel(), list(ax.lines[0].get_xdata()),
                          list(ax.lines[0].get_ydata()), kwargs))
        with patch.object(Figure, "savefig", capture):
            paths = plot.plot_waveform(self.source, "DDRPHY")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:3], ("Elapsed time (ms)", "Power (mW)", [0, 500, 2000]))
        self.assertEqual((calls[0][3][0], calls[0][3][2]), (100, 300))
        self.assertTrue(math.isnan(calls[0][3][1]))
        self.assertEqual(calls[0][4]["dpi"], 300)
        self.assertTrue(all(p.parent == self.source.parent for p in paths))
        self.assertEqual([p.suffix for p in paths], [".png", ".pdf"])
        self.assertEqual(self.source.read_bytes(), before)
