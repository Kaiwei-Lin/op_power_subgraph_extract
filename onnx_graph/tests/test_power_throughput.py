import unittest
from onnx_graph.plot_power_throughput import prepare, correlations


class FixedFrequencyTests(unittest.TestCase):
    def test_units_duplicates_and_unknown_units(self):
        headers = ["OMC名字", "DDR频率_MHz", "数据移动吞吐", "数据移动吞吐单位",
                   "VBUCK0_PERI", "VBUCK1_1V12", "DDRPHY", "VBUCK8_1V8"]
        row = dict(zip(headers, ["a", 418, 2e9, "B/s", .1, .2, .3, .4]))
        data, units = prepare(headers, [row])
        self.assertEqual(data["a", 418]["x"], 2)
        self.assertEqual(units, ["B/s"])
        with self.assertRaisesRegex(ValueError, "重复"):
            prepare(headers, [row, row])
        with self.assertRaisesRegex(ValueError, "单位不明确"):
            prepare(headers, [dict(row, 数据移动吞吐单位="")])

    def test_correlations_and_undefined_cases(self):
        r, rho, status = correlations([(1, 3), (2, 2), (3, 1)])
        self.assertAlmostEqual(r, -1)
        self.assertAlmostEqual(rho, -1)
        self.assertEqual(status, "exploratory")
        self.assertEqual(correlations([(1, 2), (1, 3)])[2], "constant_input")
        self.assertEqual(correlations([(1, 2)])[2], "insufficient_pairs")
