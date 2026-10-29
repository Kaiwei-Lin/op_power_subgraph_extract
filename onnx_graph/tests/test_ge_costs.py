import unittest

import networkx as nx

from onnx_graph.merge_power import transfer_statistics
from onnx_graph.omc import _cost, _resolved_inputs, statistics_nodes


def tensor(shape, dtype="DT_FLOAT16"):
    return dict(name="tensor", shape=shape, dtype=dtype, layout="ND")


def graph(op, inputs, outputs, attrs=None):
    value = nx.MultiDiGraph()
    value.add_node(1, op_type=op, attributes=attrs or {}, inputs=tuple(f"x{i}:0" for i in range(len(inputs))),
                   input_tensors=[tensor(x) for x in inputs], output_tensors=[tensor(x) for x in outputs],
                   name="node", original_name="node", is_container=False, hierarchy=(), hierarchy_path="",
                   hierarchy_reason="", region_id=0)
    return value


class GeCostTests(unittest.TestCase):
    def test_matrix_multiply_batch_transpose_bias_and_shape_validation(self):
        for op, a, b, y, attrs, bias, expected in [
                ("BatchMatMul", [2, 3, 4], [1, 4, 5], [2, 3, 5], {}, None, (120, 240)),
                ("BatchMatMulV2", [2, 4, 3], [5, 4], [2, 3, 5], {"adj_x1": True, "adj_x2": True}, None, (120, 240)),
                ("GemmD", [2, 3], [3, 4], [2, 4], {"alpha": 1., "beta": 1.}, [4], (24, 56)),
                ("GemmV3", [2, 3], [3, 4], [2, 4], {"beta": 0.}, [4], (24, 48))]:
            with self.subTest(op=op):
                g = graph(op, [a, b] + ([bias] if bias is not None else []), [y], attrs)
                self.assertEqual(_cost(g, 1)[:2], expected)
        for inputs, output, attrs in [([[2, 3], [4, 5]], [2, 5], {}),
                                      ([[2, 3, 4], [3, 4, 5]], [2, 3, 5], {}),
                                      ([[2, 3, 4], [4, 5]], [1, 3, 5], {}),
                                      ([[2, 3], [3, 4]], [2, 4], {"adj_x1": True, "transpose_a": False})]:
            with self.assertRaises(ValueError):
                _cost(graph("BatchMatMul", inputs, [output], attrs), 1)
        batched = graph("BatchMatMul", [[2, 3, 4], [1, 4, 5]], [[2, 3, 5]])
        batched.nodes[1]["input_tensors"][1]["layout"] = "FRACTAL_NZ"
        self.assertEqual(_cost(batched, 1)[:2], (120, 240))
        physical = graph("BatchMatMul", [[3, 1, 2, 4], [3, 1, 4, 5]], [[3, 1, 2, 5]])
        physical.nodes[1]["input_tensors"][1]["layout"] = "FRACTAL_NZ"
        with self.assertRaises(ValueError):
            _cost(physical, 1)

    def test_const_descriptor_replaces_placeholder_for_flops_and_transfer(self):
        g = graph("GemmD", [[2, 3], [1], [4]], [[2, 4]], {"beta": 1.})
        g.add_node(0, op_type="Const", constant_tensor=tensor([3, 4]))
        g.add_edge(0, 1, input_index=1, output_index=0, control=False)
        self.assertEqual(_cost(g, 1)[:2], (24, 56))
        self.assertEqual(_resolved_inputs(g, 1)[1]["shape"], [3, 4])
        self.assertEqual(g.nodes[1]["input_tensors"][1]["shape"], [1])
        nodes, _ = statistics_nodes(g)
        _, total = transfer_statistics(dict(source_format="OMC", total={"node_count": 1}, nodes=nodes, levels=[]))
        self.assertEqual(total["数据转移字节数"], 60)
        g.nodes[0]["constant_tensor"] = None
        nodes, _ = statistics_nodes(g)
        _, total = transfer_statistics(dict(source_format="OMC", total={"node_count": 1}, nodes=nodes, levels=[]))
        self.assertIsNone(total["数据转移字节数"])
        self.assertEqual(total["数据转移量完整"], "否")

    def test_norm_prelu_and_movement_formulas(self):
        g = graph("LayerNorm", [[2, 4], [4], [4]], [[2, 4], [2, 1], [2, 1]], {"begin_norm_axis": -1})
        self.assertEqual(_cost(g, 1)[:2], (0, 62))
        g.nodes[1]["attributes"]["norm_type"] = "RMSNorm"
        with self.assertRaises(ValueError):
            _cost(g, 1)
        g = graph("PRelu", [[1, 3, 2, 2], [3, 1, 1]], [[1, 3, 2, 2]], {"data_format": "NCHW"})
        self.assertEqual(_cost(g, 1)[:2], (0, 24))
        for op in ("SplitV", "StridedSliceV2", "ScatterNdUpdate"):
            self.assertEqual(_cost(graph(op, [], []), 1)[:2], (0, 0))

    def test_opaque_fused_kernel_names_are_not_used_to_guess_flops(self):
        g = graph("FusionOpNode", [[2, 4]], [[2, 4]], {"_kernel_name": "fused_sigmoid_mul"})
        with self.assertRaises(ValueError):
            _cost(g, 1)


if __name__ == "__main__":
    unittest.main()
