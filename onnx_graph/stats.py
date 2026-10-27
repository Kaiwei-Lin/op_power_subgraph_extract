"""按名称前缀统计 ONNX/OMC 子图和理论计算量；可直接运行或作为模块导入。"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from math import prod
from pathlib import Path

import onnx

if __package__:
    from .core import _fill_graph_names, format_name_output_mismatches, load_name_aliases, onnx_to_networkx
    from .profiling import attach_profile, time_totals
    from .omc import omc_to_networkx, statistics_nodes
else:
    from core import _fill_graph_names, format_name_output_mismatches, load_name_aliases, onnx_to_networkx
    from profiling import attach_profile, time_totals
    from omc import omc_to_networkx, statistics_nodes


# 直接运行此脚本时，修改这里即可。命令行参数可覆盖这些设置。
PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_DIR / "models/handfree_ainr/handfree_ainr.onnx"
REPAIR_NAMES = True
NAME_ALIASES_JSON = None  # 可选：PACKAGE_DIR / "name_aliases.json"
SLASH_COUNTS = [1, 2]  # 1: root/block；2: root/block/layer；前导 / 不计数
BATCH_SIZE = 1  # 仅将输入的未知第 0 维视为 batch；设 None 可禁用这一假设
INPUT_SHAPES = {}  # 例如 {"input:0": [1, 224, 224, 3]}，用于指定真实输入尺寸
PROFILE_CSV = None  # 可选：Path(r"D:/models/_20261016_110720_op.csv")；None 不增加时间/算术强度列
GE_PROTO_PATH = None  # OMC 可选：DDK 内的 ge_ir_pb2.py；当前仓库会自动查找
OUTPUT_DIR = None  # None: 模型目录/networkx/模型名_stats

CONVENTIONS = [
    "按一次推理、稠密计算估算；1 MAC = 2 FLOPs，偏置及其他逐元素操作另计。",
    "FLOPs 是标量操作近似值：比较、exp/log/sqrt 等各按 1 次；不代表硬件指令数或耗时。",
    "卷积按完整卷积核计数（包括 padding）；池化按完整窗口近似，边界实际操作数可能较少。",
    "形状/索引/数据搬移算子按 0 FLOPs 计；这不代表无内存开销或无耗时。",
    "仅统计主图节点；控制流、未支持的算子及缺失形状记为未知，不计为零。",
    "每个层级单独分组；跨层级的结果包含重复节点，不可相加。浅层名称及无名节点列为未分组。",
]

_MOVEMENT_OPS = {
    "Identity", "Constant", "ConstantOfShape", "Reshape", "Transpose", "Flatten",
    "Squeeze", "Unsqueeze", "Shape", "Size", "Concat", "Split", "Slice", "Gather",
    "GatherElements", "GatherND", "Expand", "Tile", "Pad", "Cast", "CastLike",
}
_ELEMENTWISE_COST = {
    "Add": 1, "Sub": 1, "Mul": 1, "Div": 1, "Pow": 1, "Abs": 1, "Neg": 1,
    "Exp": 1, "Log": 1, "Sqrt": 1, "Reciprocal": 1, "Floor": 1, "Ceil": 1,
    "Round": 1, "Relu": 1, "LeakyRelu": 2, "PRelu": 2, "Sigmoid": 4,
    "Tanh": 6, "HardSigmoid": 4, "HardSwish": 5,
}


def _tensor_info(model):
    info = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.HasField("tensor_type"):
            tensor = value.type.tensor_type
            shape = None
            if tensor.HasField("shape"):
                shape = [d.dim_value if d.HasField("dim_value") else d.dim_param or None
                         for d in tensor.shape.dim]
            info[value.name] = {"shape": shape, "dtype": tensor.elem_type}
    for tensor in model.graph.initializer:
        info[tensor.name] = {"shape": list(tensor.dims), "dtype": tensor.data_type}
    for tensor in model.graph.sparse_initializer:
        info[tensor.values.name] = {"shape": list(tensor.dims), "dtype": tensor.values.data_type}
    return info


def _infer(model, input_shapes, batch_size):
    model = deepcopy(model)
    _fill_graph_names(model.graph)
    before = _tensor_info(model)
    inputs = {v.name: v for v in model.graph.input}
    if set(input_shapes) - inputs.keys():
        raise ValueError(f"未知输入名称: {sorted(set(input_shapes) - inputs.keys())}")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size 必须是正整数或 None")
    assumptions = []
    changed = False
    weights = {t.name for t in model.graph.initializer}
    weights.update(t.values.name for t in model.graph.sparse_initializer)
    for name, value in inputs.items():
        if not value.type.HasField("tensor_type"):
            if name in input_shapes:
                raise ValueError(f"输入 {name!r} 不是普通张量，不能指定张量形状")
            continue
        shape = value.type.tensor_type.shape
        if name in input_shapes:
            dims = list(input_shapes[name])
            if any(type(d) is not int or d < 0 for d in dims):
                raise ValueError(f"输入 {name!r} 的维度必须是非负整数")
            shape.ClearField("dim")
            shape.SetInParent()
            for size in dims:
                shape.dim.add().dim_value = size
            assumptions.append(f"指定输入 {name}: {dims}")
            changed = True
        elif batch_size is not None and name not in weights and shape.dim:
            if not shape.dim[0].HasField("dim_value"):
                shape.dim[0].dim_value = batch_size
                assumptions.append(f"假设输入 {name} 的未知第 0 维为 batch={batch_size}")
                changed = True
    if changed:
        # 输入尺寸改变后，缓存的中间/输出尺寸可能过期，必须重新推断。
        for value in [*model.graph.value_info, *model.graph.output]:
            if value.type.HasField("tensor_type"):
                value.type.tensor_type.ClearField("shape")
    warnings = []
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except (onnx.shape_inference.InferenceError, RuntimeError, ValueError) as exc:
        warnings.append(f"形状推断未完成，将使用可用信息: {exc}")
    info = _tensor_info(model)
    input_report = [{"name": name, "original_shape": before.get(name, {}).get("shape"),
                     "effective_shape": info.get(name, {}).get("shape")}
                    for name in inputs if name not in weights]
    return info, input_report, assumptions, warnings


def _shape(info, name):
    shape = info.get(name, {}).get("shape")
    if shape is None or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError(f"张量 {name!r} 的形状不确定: {shape}")
    return shape


def _cost(node, info, opset):
    """返回 MACs、FLOPs 和公式；无法可靠估算时抛 ValueError。"""
    op = node.op_type
    if node.domain not in ("", "ai.onnx"):
        raise ValueError(f"暂不支持自定义 domain: {node.domain}")
    if op in _MOVEMENT_OPS:
        return 0, 0, "形状/索引/数据搬移，不计浮点计算"
    supported = set(_ELEMENTWISE_COST) | {
        "Conv", "MatMul", "Gemm", "Clip", "Sum", "Min", "Max", "Softmax",
        "AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool",
        "ReduceSum", "ReduceProd", "ReduceMean", "ReduceMax", "ReduceMin",
        "BatchNormalization", "LayerNormalization", "InstanceNormalization",
    }
    if op not in supported:
        raise ValueError(f"暂未实现 {op} 的计算量公式（含控制流/量化/融合算子）")
    dtype = info.get(node.input[0], {}).get("dtype", 0) if node.input else 0
    dtype_name = onnx.TensorProto.DataType.Name(dtype)
    if not (dtype_name.startswith("FLOAT") or dtype_name in {"DOUBLE", "BFLOAT16"}):
        raise ValueError(f"输入类型 {dtype_name} 不是已知浮点类型")
    n = prod(_shape(info, node.output[0]))
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if op == "Conv":
        w = _shape(info, node.input[1])
        if len(w) < 3:
            raise ValueError("卷积权重维数应至少为 3")
        macs = n * prod(w[1:])  # W[1] 已经是 C/group，不能再除以 group。
        bias = n if len(node.input) > 2 and node.input[2] else 0
        return macs, 2 * macs + bias, "MACs=输出元素数×(C/group)×卷积核体积；FLOPs=2×MACs+偏置"
    if op in {"MatMul", "Gemm"}:
        a, b = (_shape(info, name) for name in node.input[:2])
        if not a or not b or (op == "Gemm" and (len(a) != 2 or len(b) != 2)):
            raise ValueError("矩阵输入维数不合法")
        k = a[0] if op == "Gemm" and attrs.get("transA", 0) else a[-1]
        bk = b[-1] if op == "Gemm" and attrs.get("transB", 0) else b[-2] if len(b) > 1 else b[0]
        if k != bk:
            raise ValueError(f"矩阵乘法的 K 不一致: {k} != {bk}")
        macs = n * k
        flops = 2 * macs
        if op == "Gemm":
            flops += n * (attrs.get("alpha", 1.0) != 1.0)
            beta = attrs.get("beta", 1.0)
            if len(node.input) > 2 and node.input[2] and beta != 0:
                flops += n * (1 + (beta != 1.0))
        return macs, flops, "MACs=输出元素数×K；Gemm 另计 alpha/beta 缩放及偏置相加"
    if op in _ELEMENTWISE_COST:
        factor = _ELEMENTWISE_COST[op]
        return 0, n * factor, f"每个输出元素约 {factor} 次标量操作"
    if op == "Clip":
        bounds = 2 if opset < 11 else sum(bool(x) for x in node.input[1:3])
        return 0, n * bounds, "输出元素数×有效上下界数量"
    if op in {"Sum", "Min", "Max"}:
        return 0, n * (len(node.input) - 1), "输出元素数×(输入数量-1)"
    if op.endswith("Pool"):
        kernel = (prod(_shape(info, node.input[0])[2:]) if op.startswith("Global")
                  else prod(attrs["kernel_shape"]))
        factor = kernel if "Average" in op else kernel - 1
        return 0, n * factor, "输出元素数×窗口体积（平均池化）或×(窗口体积-1)（最大池化）"
    if op == "Softmax":
        dims = _shape(info, node.input[0])
        axis = attrs.get("axis", -1 if opset >= 13 else 1)
        if not -len(dims) <= axis < len(dims):
            raise ValueError("Softmax axis 超出维数")
        axis %= len(dims)
        width = dims[axis] if opset >= 13 else prod(dims[axis:])
        groups = n // width if width else 0
        return 0, 5 * n - 2 * groups, "稳定 Softmax：max、减法、exp、求和、除法，约 5N-2组数"
    if op.startswith("Reduce"):
        size = prod(_shape(info, node.input[0]))
        if size == 0 or size < n:
            raise ValueError("不估算空集合或尺寸异常的归约")
        return 0, size if op == "ReduceMean" else size - n, "归约约 N-M 次；均值另计 M 次除法"
    dims = _shape(info, node.input[0])
    if op == "BatchNormalization":
        if attrs.get("training_mode", 0) or len(node.output) > 1:
            raise ValueError("仅支持推理模式 BatchNormalization")
        return 0, 4 * n + 2 * dims[1], "未融合 BN：4N+2C（减、除、乘、加及每通道 epsilon/sqrt）"
    axis = attrs.get("axis", -1) if op == "LayerNormalization" else 2
    if not -len(dims) <= axis < len(dims):
        raise ValueError("归一化 axis 超出维数")
    groups = prod(dims[:axis % len(dims)])
    bias = bool(len(node.input) > 2 and node.input[2])
    return 0, (6 + bias) * n + 3 * groups, "归一化近似：6N+3组数，有偏置另加 N"


def _totals(nodes, include_time=False):
    unknown = [n["node_id"] for n in nodes if n["flops"] is None]
    macs = sum(n["macs"] for n in nodes if n["macs"] is not None)
    flops = sum(n["flops"] for n in nodes if n["flops"] is not None)
    result = {
        "node_count": len(nodes),
        "operator_counts": dict(sorted(Counter(n["operator"] for n in nodes).items())),
        "known_macs": macs, "known_flops": flops,
        "macs": None if unknown else macs, "flops": None if unknown else flops,
        "complete": not unknown, "unknown_node_ids": unknown,
    }
    if include_time:
        result.update(time_totals(nodes, result["flops"]))
    return result


def analyze_subgraphs(source, slash_counts=(1, 2), *, repair_names=True,
                      batch_size=None, input_shapes=None, profile_csv=None, ge_proto_path=None,
                      name_aliases=None):
    """统计每个层级的互不重叠前缀分组；按扩展名识别 OMC/OM。"""
    aliases = load_name_aliases(name_aliases)
    counts = list(dict.fromkeys(slash_counts))
    if not counts or any(type(k) is not int or k < 0 for k in counts):
        raise ValueError("slash_counts 必须包含至少一个非负整数")
    is_omc = isinstance(source, (str, Path)) and Path(source).suffix.lower() in {".omc", ".om"}
    extra = {}
    conventions = CONVENTIONS
    if is_omc:
        if input_shapes:
            raise ValueError("OMC 使用编译时的张量形状，不支持 --input-shape 覆盖；请重新编译模型")
        graph = omc_to_networkx(source, repair_names=repair_names, ge_proto_path=ge_proto_path,
                               name_aliases=aliases)
        nodes, excluded = statistics_nodes(graph)
        graph_name = graph.graph["name"]
        root_regions = {r["region_id"] for r in graph.graph["regions"] if r["parent_node_id"] is None}
        inputs = [{"name": d["name"], "original_shape": t["shape"], "effective_shape": t["shape"]}
                  for _, d in graph.nodes(data=True) if d["op_type"] == "Data" and d["region_id"] in root_regions
                  for t in d["output_tensors"]]
        assumptions = ["OMC 使用文件中的编译形状；batch_size 假设不应用于 OMC。"]
        warnings = list(graph.graph["warnings"])
        conventions = [c for i, c in enumerate(CONVENTIONS) if i != 4] + [
            "OMC 展开 GraphOp.subgraph；Const、Data、NetOutput、已展开 GraphOp 不计入运算节点。",
            "OMC 保留 GE 编译后算子类型；融合 ReLU 计入卷积 FLOPs，不额外恢复原 ONNX 节点。",
            "FLOPs 按逻辑形状估算，不包含物理通道补齐或专用内核开销；缺少公式/逻辑形状记为未知。",
        ]
        extra["omc"] = {"schema_path": graph.graph["schema_path"], "all_node_count": len(graph),
                        "edge_count": graph.number_of_edges(), "excluded_node_ids": excluded,
                        "excluded_operator_counts": dict(Counter(graph.nodes[i]["op_type"] for i in excluded)),
                        "regions": [{k: v for k, v in r.items() if k != "proto"} for r in graph.graph["regions"]]}
    else:
        graph = onnx_to_networkx(source, repair_names=repair_names, name_aliases=aliases)
        model = graph.graph["model"]
        graph_name = model.graph.name
        info, inputs, assumptions, warnings = _infer(model, input_shapes or {}, batch_size)
        opset = next((x.version for x in model.opset_import if x.domain in ("", "ai.onnx")), 1)
        nodes = []
        for node_id, data in graph.nodes(data=True):
            node = data["proto"]
            record = {"node_id": node_id, "name": node.name, "original_name": data["original_name"],
                      "operator": f"{node.domain}::{node.op_type}" if node.domain else node.op_type,
                      "hierarchy": list(data["hierarchy"]),
                      "inputs": [{"name": x, **info.get(x, {})} for x in node.input if x],
                      "outputs": [{"name": x, **info.get(x, {})} for x in node.output if x]}
            try:
                record["macs"], record["flops"], record["formula"] = _cost(node, info, opset)
            except (ValueError, KeyError, IndexError) as exc:
                record.update(macs=None, flops=None, reason=str(exc))
            nodes.append(record)
    include_time = profile_csv is not None
    profile = attach_profile(nodes, profile_csv, name_aliases=aliases) if include_time else None
    levels = []
    for count in counts:
        groups = defaultdict(list)
        unnamed, shallow = [], []
        for node in nodes:
            parts = node["hierarchy"]
            if not parts:
                unnamed.append(node["node_id"])
            elif len(parts) <= count:
                shallow.append(node["node_id"])
            else:
                groups["/".join(parts[:count + 1])].append(node)
        subgraphs = [{"prefix": prefix, "node_ids": [n["node_id"] for n in members],
                      **_totals(members, include_time)} for prefix, members in sorted(groups.items())]
        levels.append({"slash_count": count, "subgraph_count": len(subgraphs),
                       "grouped_node_count": sum(g["node_count"] for g in subgraphs),
                       "unnamed_node_ids": unnamed, "shallower_node_ids": shallow,
                       "subgraphs": subgraphs})
    report = {"source": str(Path(source).resolve()) if isinstance(source, (str, Path)) else "<ModelProto>",
            "source_format": "OMC" if is_omc else "ONNX",
            "graph_name": graph_name, "repair_names": repair_names,
            "name_aliases": aliases, "name_output_mismatches": graph.graph["name_output_mismatches"],
            "rename_report": graph.graph.get("rename_report", []), "inputs": inputs,
            "assumptions": assumptions, "warnings": warnings, "conventions": conventions,
            "total": _totals(nodes, include_time), "levels": levels, "nodes": nodes, **extra}
    if include_time:
        report["profile"] = profile
    return report


def format_report(report):
    def amount(total):
        suffix = " + ?（含未知算子）" if not total["complete"] else ""
        return (f"{total['known_macs']:,} MACs; {total['known_flops']:,} FLOPs"
                f" ({total['known_flops'] / 1e6:.6f} MFLOPs){suffix}")

    def ops(total):
        return ", ".join(f"{op}×{n}" for op, n in total["operator_counts"].items())

    source_format = report.get("source_format", "ONNX")
    scope = "展开后运算图" if source_format == "OMC" else "主图"
    lines = [f"{source_format}: {report['source']}", f"REPAIR_NAMES: {report['repair_names']}",
             f"{scope}: {report['total']['node_count']} 个节点；{amount(report['total'])}",
             f"{scope}算子: {ops(report['total'])}", "", "输入:"]
    lines += [f"  {x['name']}: {x['original_shape']} -> {x['effective_shape']}" for x in report["inputs"]]
    lines += ["", "估算口径与假设:"]
    lines += [f"  - {x}" for x in report["conventions"] + report["assumptions"] + report["warnings"]]
    if report["rename_report"]:
        statuses = Counter(x["status"] for x in report["rename_report"])
        lines += [f"名称处理: {dict(statuses)}（详细记录见 JSON）"]
    lines += ["", format_name_output_mismatches(report.get("name_output_mismatches", []))]
    if "profile" in report:
        profile = report["profile"]
        lines += ["", f"运行时间 CSV: {profile['source']}",
                  f"有效时间匹配: {profile['matched_node_count']}/{report['total']['node_count']} 个 {source_format} 节点；"
                  f"运行算子 {profile['matched_operator_count']}/{profile['operator_count']} 个；"
                  f"各算子采样次数: {profile['sample_counts']}"]
        lines += [f"  - {x}" for x in profile["conventions"]]
        lines += [f"{scope}已匹配时间合计: {report['total']['known_exec_time']:.6f} us"]
    for level in report["levels"]:
        lines += ["", f"前缀含 {level['slash_count']} 个 /：{level['subgraph_count']} 个子图，"
                  f"覆盖 {level['grouped_node_count']} 个节点；"
                  f"未分组：无名 {len(level['unnamed_node_ids'])}，名称较浅 {len(level['shallower_node_ids'])}"]
        for i, group in enumerate(level["subgraphs"], 1):
            lines += [f"{i}. {group['prefix']}",
                      f"   节点数: {group['node_count']}；算子: {ops(group)}",
                      f"   计算量: {amount(group)}"]
            if group["unknown_node_ids"]:
                lines += [f"   未知节点 ID: {group['unknown_node_ids']}"]
            if "profile" in report:
                duration = f"{group['exec_time']:.6f}" if group["exec_time"] is not None else "未知"
                ratio = (f"{group['arithmetic_intensity']:.6f}"
                         if group["arithmetic_intensity"] is not None else "未知")
                lines += [f"   exec_time: {duration} us；已知部分: {group['known_exec_time']:.6f} us；"
                          f"时间匹配节点: {group['exec_time_matched_nodes']}/{group['node_count']}",
                          f"   算术强度（FLOPs/exec_time）: {ratio} FLOPs/us"]
        for label, key in [("无名", "unnamed_node_ids"), ("名称较浅", "shallower_node_ids")]:
            if level[key]:
                lines += [f"  未分组（{label}）节点 ID: {level[key]}"]
    unknown = [n for n in report["nodes"] if n["flops"] is None]
    if unknown:
        lines += ["", "无法估算的节点（完整总量未知，以上仅给出已知部分）:"]
        lines += [f"  ID={n['node_id']} {n['name'] or '<无名>'} ({n['operator']}): {n['reason']}"
                  for n in unknown]
    return "\n".join(lines) + "\n"


def write_csv_reports(report, output_dir):
    """每个斜杠层级一个 CSV，算子各占一列，未知计算量留空。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    operators = sorted(report["total"]["operator_counts"])
    headers = ["子图名字", "节点数", *[f"{op}数量" for op in operators],
               "MACs", "FLOPs", "MFLOPs", "计算量完整", "未知计算量节点数",
               "已知MACs", "已知FLOPs"]
    include_time = "profile" in report
    if include_time:
        headers += ["exec_time", "exec_time_unit", "算术强度", "算术强度单位", "已知exec_time",
                    "时间匹配节点数", "时间未匹配节点数"]
    paths = []
    for level in report["levels"]:
        path = output / f"stats_{level['slash_count']}_slash.csv"
        # UTF-8 BOM 方便 Excel 识别中文；newline 避免 Windows 多出空行。
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for group in level["subgraphs"]:
                row = [
                    group["prefix"], group["node_count"],
                    *[group["operator_counts"].get(op, 0) for op in operators],
                    group["macs"], group["flops"],
                    group["flops"] / 1e6 if group["flops"] is not None else None,
                    "是" if group["complete"] else "否", len(group["unknown_node_ids"]),
                    group["known_macs"], group["known_flops"],
                ]
                if include_time:
                    row += [group["exec_time"], "us", group["arithmetic_intensity"], "FLOPs/us",
                            group["known_exec_time"], group["exec_time_matched_nodes"],
                            len(group["exec_time_missing_node_ids"])]
                writer.writerow(row)
        paths.append(path)
    return paths


def write_reports(report, output_dir):
    """保存同一份统计的 JSON、TXT 和分层 CSV，返回本次生成的 CSV 路径。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "stats.txt").write_text(format_report(report), encoding="utf-8")
    (output / "name_output_mismatches.json").write_text(
        json.dumps(report.get("name_output_mismatches", []), ensure_ascii=False, indent=2), encoding="utf-8")
    return write_csv_reports(report, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=MODEL_PATH)
    parser.add_argument("--slashes", type=int, nargs="+", default=SLASH_COUNTS,
                        help="前缀中 / 的数量，不包括前导 /；例如 --slashes 1 2")
    parser.add_argument("--repair-names", action=argparse.BooleanOptionalAction, default=REPAIR_NAMES)
    parser.add_argument("--name-aliases", type=Path, default=NAME_ALIASES_JSON,
                        help="可选 JSON 前缀别名：{旧前缀: 新前缀}")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-batch-assumption", dest="batch_size", action="store_const", const=None)
    parser.add_argument("--input-shape", action="append", default=[], metavar="NAME=DIM,DIM,...")
    profile_options = parser.add_mutually_exclusive_group()
    profile_options.add_argument("--profile-csv", type=Path, default=PROFILE_CSV,
                                 help="可选的算子运行时间 CSV，时间列为 total time(us)")
    profile_options.add_argument("--no-profile", dest="profile_csv", action="store_const", const=None,
                                 help="禁用配置中的运行时间 CSV")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ge-proto", type=Path, default=GE_PROTO_PATH,
                        help="OMC 解析使用的 DDK ge_ir_pb2.py 路径；ONNX 输入不需要")
    args = parser.parse_args(argv)
    shapes = dict(INPUT_SHAPES)
    for specification in args.input_shape:
        try:
            name, dims = specification.split("=", 1)
            shapes[name] = [int(x) for x in dims.split(",")] if dims else []
        except ValueError:
            parser.error(f"输入形状格式错误: {specification!r}，应为 NAME=1,3,224,224")
    try:
        report = analyze_subgraphs(args.model, args.slashes, repair_names=args.repair_names,
                                   batch_size=args.batch_size, input_shapes=shapes, profile_csv=args.profile_csv,
                                   ge_proto_path=args.ge_proto, name_aliases=args.name_aliases)
    except ValueError as exc:
        parser.error(str(exc))
    suffix = "_omc_stats" if report["source_format"] == "OMC" else "_stats"
    output = Path(args.output_dir) if args.output_dir is not None else args.model.parent / "networkx" / f"{args.model.stem}{suffix}"
    csv_paths = write_reports(report, output)
    print(format_report(report))
    print(f"报告已保存: {output.resolve() / 'stats.txt'}")
    print(f"详细数据: {output.resolve() / 'stats.json'}")
    for path in csv_paths:
        print(f"分层 CSV: {path.resolve()}")


if __name__ == "__main__":
    main()
