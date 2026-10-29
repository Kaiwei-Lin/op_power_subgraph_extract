"""Read unencrypted IMOD OMC files using the GE schema supplied with Huawei DDK.

The NetworkX graph contains separate graph regions, linked by parent_node_id
metadata. Edges stay inside their original region; GraphOp remains a container.
Raw protobufs and the complete source file are retained, including weights and
compiled tasks. This is an analysis representation, not an OMC/ONNX exporter.
"""

import ast
from collections import defaultdict
from math import prod
import os
from pathlib import Path
import struct

import networkx as nx
from google.protobuf import descriptor_pool, json_format, message_factory
from google.protobuf.message import DecodeError
import onnx

if __package__:
    from .core import check_name_outputs, repair_node_names
    from .core import _hierarchy_options, _output_hierarchy_issue, _set_node_hierarchy
else:
    from core import check_name_outputs, repair_node_names
    from core import _hierarchy_options, _output_hierarchy_issue, _set_node_hierarchy


_SCHEMA_RELATIVE = Path("tools/tools_ascendc/package/python/opc_tool/ge_ir_pb2.py")
_STRUCTURAL = {"Const", "Data", "NetOutput"}
_MOVEMENT = {"Permute", "TransData", "CastT", "Reshape", "Squeeze", "Unsqueeze",
             "Flatten", "GatherV2D", "ConcatD", "SplitD", "SplitV", "StridedSliceV2",
             "ScatterNdUpdate", "Identity"}


def _schema(path=None):
    if path is None:
        path = os.environ.get("ONNX_GRAPH_GE_PROTO")
    if path is None and os.environ.get("DDK_PATH"):
        path = Path(os.environ["DDK_PATH"]) / _SCHEMA_RELATIVE
    if path is None:
        local = Path(__file__).resolve().parents[2] / "spatialnet_field_kit/microbenchmark/.local"
        candidates = sorted(p / _SCHEMA_RELATIVE for p in local.glob("DDK-tools-*")
                            if (p / _SCHEMA_RELATIVE).is_file())
        if len(candidates) != 1:
            raise ValueError("请用 --ge-proto 指定 DDK 的 ge_ir_pb2.py（或设置 ONNX_GRAPH_GE_PROTO）")
        path = candidates[0]
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"GE schema 文件不存在: {path}")
    # Read the descriptor literal only. Do not execute the SDK Python module;
    # older generated modules are also incompatible with newer protobuf APIs.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    descriptors = []
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.keyword) and node.arg == "serialized_pb":
            value = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "AddSerializedFile" and len(node.args) == 1):
            value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, bytes):
            descriptors.append(value.value)
    if len(set(descriptors)) != 1:
        raise ValueError(f"无法从 {path} 读取唯一的 GE protobuf descriptor")
    pool = descriptor_pool.DescriptorPool()
    try:
        pool.AddSerializedFile(descriptors[0])
        model = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))
        graph = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.GraphDef"))
    except (TypeError, KeyError) as exc:
        raise ValueError("GE schema 缺少 ModelDef/GraphDef 或依赖其他 schema") from exc
    return model, graph, descriptors[0], str(path.resolve())


def _partitions(raw):
    if len(raw) < 260 or raw[:4] != b"IMOD":
        raise ValueError("仅支持带 IMOD 文件头的未加密 OMC/OM 文件")
    head_size, version = struct.unpack_from("<II", raw, 4)
    if head_size != 256 or version != 0x10000000:
        raise ValueError(f"未验证的 IMOD 版本/文件头: version={version:#x}, header={head_size}")
    if raw[80]:
        raise ValueError("不支持加密 OMC")
    declared = struct.unpack_from("<I", raw, 76)[0]
    if declared != len(raw) - head_size:
        raise ValueError("IMOD 长度字段与文件大小不一致，文件可能截断或格式不受支持")
    count = struct.unpack_from("<I", raw, head_size)[0]
    for width, fmt in ((4, "<III"), (8, "<QQQ")):
        start = head_size + width + count * 3 * width
        if count < 1 or count > 32 or start > len(raw):
            continue
        records = []
        for i in range(count):
            kind, offset, size = struct.unpack_from(fmt, raw, head_size + width + i * 3 * width)
            records.append({"type": kind, "offset": start + offset, "size": size})
        ordered = sorted(records, key=lambda p: p["offset"])
        if (len({p["type"] for p in records}) != count or not any(p["type"] == 0 for p in records)
                or any(p["type"] >= 32 or p["offset"] + p["size"] > len(raw) for p in records)
                or any(a["offset"] + a["size"] > b["offset"] for a, b in zip(ordered, ordered[1:]))):
            continue
        return records
    raise ValueError("无效或不支持的 IMOD 分区表")


def _dictionary(message):
    return json_format.MessageToDict(message, preserving_proto_field_name=True,
                                    use_integers_for_enums=True)


def _text(value):
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value


def _attribute(value):
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    field = getattr(value, kind)
    if kind == "s":
        return _text(field)
    if kind in {"i", "f", "b", "bt", "dt"}:
        return field
    if kind == "list":
        fields = [(f, v) for f, v in field.ListFields() if f.name != "val_type"]
        if not fields:
            return []
        if len(fields) == 1:
            f, values = fields[0]
            return [_dictionary(v) if f.message_type else _text(v) if f.name == "s" else v
                    for v in values]
    return _dictionary(field)


def _tensor(desc):
    dtype = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type.values_by_number.get(desc.dtype)
    return {"name": desc.name, "shape": list(desc.shape.dim) if desc.HasField("shape") else None,
            "dtype": dtype.name if dtype else f"UNKNOWN({desc.dtype})",
            "layout": desc.layout}


def omc_to_networkx(source, *, ge_proto_path=None, repair_names=False, name_aliases=None,
                   hierarchy_source="name"):
    """Keep every GE operator, including Const, Data, NetOutput and GraphOp.

    Integer IDs are unique across regions. node['proto'] is the full serialized
    GE OpDef, not an ONNX NodeProto. Node names are display labels; dependencies
    always use the original names and ports. Empty names are preserved.
    """
    repair_names, aliases = _hierarchy_options(hierarchy_source, repair_names, name_aliases)
    path = Path(source)
    raw = path.read_bytes()
    partitions = _partitions(raw)
    Model, Graph, schema, schema_path = _schema(ge_proto_path)
    part = next(p for p in partitions if p["type"] == 0)
    try:
        model = Model.FromString(raw[part["offset"]:part["offset"] + part["size"]])
    except DecodeError as exc:
        raise ValueError("OMC 模型定义不能按此 GE schema 解析") from exc
    if not model.graph:
        raise ValueError("OMC 中没有可解析的 GraphDef")
    result = nx.MultiDiGraph()
    result.graph.update(source_format="omc", source=str(path.resolve()), name=model.name,
                        source_bytes=raw, schema_bytes=schema, schema_path=schema_path,
                        partitions=partitions, model_proto=model.SerializeToString(),
                        regions=[], rename_report=[], warnings=[], name_aliases=aliases,
                        name_output_mismatches=[] if hierarchy_source == "name" else None,
                        hierarchy_source=hierarchy_source, repair_names=repair_names, output_hierarchy_issues=[])

    def visit(region, parent=None, depth=0):
        if depth > 64:
            raise ValueError("OMC 子图嵌套超过 64 层")
        region_id = len(result.graph["regions"])
        ids = list(range(len(result), len(result) + len(region.op)))
        result.graph["regions"].append({"region_id": region_id, "name": region.name,
                                       "parent_node_id": parent, "node_ids": ids,
                                       "inputs": list(region.input), "outputs": list(region.output),
                                       "proto": region.SerializeToString()})
        names = {}
        for node_id, op in zip(ids, region.op):
            if op.name.strip():
                if op.name in names:
                    raise ValueError(f"OMC 图 {region.name!r} 中有重名节点 {op.name!r}")
                names[op.name] = node_id
            attrs = {k: _attribute(v) for k, v in op.attr.items()}
            inputs = list(op.input)
            tensors_in = [dict(_tensor(d), name=inputs[i] if i < len(inputs) else d.name)
                          for i, d in enumerate(op.input_desc)]
            tensors_out = [dict(_tensor(d), name=d.name or f"{op.name}:{i}")
                           for i, d in enumerate(op.output_desc)]
            constant = (op.attr["value"].t if op.type == "Const" and "value" in op.attr
                        and op.attr["value"].WhichOneof("value") == "t" else None)
            result.add_node(node_id, proto=op.SerializeToString(), name=op.name, original_name=op.name,
                            original_index=node_id, op_type=op.type, domain="ge", region_id=region_id,
                            inputs=tuple(inputs), outputs=tuple(f"{op.name}:{i}" for i in range(len(op.output_desc))),
                            attributes=attrs, input_tensors=tensors_in, output_tensors=tensors_out,
                            hierarchy_outputs=tuple(d.name for d in op.output_desc),
                            constant_tensor=_tensor(constant.desc) if constant is not None else None,
                            is_container=False)
        if hierarchy_source == "name":
            evidence = onnx.helper.make_model(onnx.helper.make_graph([
                onnx.helper.make_node(op.type, [d.name for d in op.input_desc if d.name],
                                      [d.name for d in op.output_desc if d.name], name=op.name)
                for op in region.op], "name_evidence", [], []))
            report = []
            if repair_names:
                evidence, report = repair_node_names(evidence, name_aliases=aliases)
            issues = check_name_outputs(evidence, name_aliases=aliases, rename_report=report)
            for issue in issues:
                issue.update(node_id=ids[issue["node_id"]], region_id=region_id)
            result.graph["name_output_mismatches"].extend(issues)
            if repair_names:
                for node_id, record in zip(ids, report):
                    record["node_id"] = node_id
                    result.nodes[node_id]["name"] = record["new_name"]
                result.graph["rename_report"].extend(report)
        for node_id, op in zip(ids, region.op):
            for port, ref in enumerate(op.input):
                if not ref:
                    continue
                producer, colon, index = ref.rpartition(":")
                if not colon or producer not in names:
                    raise ValueError(f"OMC 图 {region.name!r} 中无法解析依赖 {ref!r}")
                try:
                    index = int(index)
                except ValueError as exc:
                    raise ValueError(f"OMC 依赖端口无效: {ref!r}") from exc
                if index < -1:
                    raise ValueError(f"OMC 依赖端口无效: {ref!r}")
                result.add_edge(names[producer], node_id, tensor=ref, input_index=port,
                                output_index=index, control=index == -1, region_id=region_id)
            if op.type == "GraphOp" and "subgraph" in op.attr:
                value = op.attr["subgraph"]
                try:
                    child = Graph.FromString(value.bt) if value.WhichOneof("value") == "bt" else value.g
                    if not child.op:
                        raise ValueError("GraphOp 子图为空或没有可识别的子图字段")
                    result.nodes[node_id]["is_container"] = True
                    visit(child, node_id, depth + 1)
                except DecodeError as exc:
                    raise ValueError(f"无法解析 GraphOp {op.name!r} 的子图") from exc
        return region_id

    for region in model.graph:
        visit(region)
    hierarchy = defaultdict(list)
    for node_id, data in result.nodes(data=True):
        parts = _set_node_hierarchy(data, hierarchy_source, data["hierarchy_outputs"])
        if data["hierarchy_reason"]:
            issue = _output_hierarchy_issue(node_id, data, data["hierarchy_outputs"])
            issue["region_id"] = data["region_id"]
            result.graph["output_hierarchy_issues"].append(issue)
        hierarchy[""].append(node_id)
        for end in range(1, len(parts) + 1):
            hierarchy["/".join(parts[:end])].append(node_id)
    result.graph["hierarchy"] = dict(hierarchy)
    result.graph["original_node_ids"] = tuple(result)
    return result


def _shape(tensor):
    shape = tensor.get("shape")
    if shape is None or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError(f"OMC 张量的逻辑形状不确定: {shape}")
    if (tensor.get("layout", "").startswith("FRACTAL") or tensor.get("layout") == "NC1HWC0") and len(shape) > 4:
        raise ValueError("只有物理分块形状，无法可靠恢复逻辑 FLOPs")
    return shape


def _resolved_inputs(graph, node_id):
    tensors = [dict(t) for t in graph.nodes[node_id]["input_tensors"]]
    for upstream, _, edge in graph.in_edges(node_id, data=True):
        port = edge["input_index"]
        if edge["control"] or port >= len(tensors) or graph.nodes[upstream]["op_type"] != "Const":
            continue
        # Const 的算子输出/消费者 input_desc 常是 [1] 占位，实际权重在 value.t.desc。
        value = graph.nodes[upstream].get("constant_tensor") if edge["output_index"] == 0 else None
        value = value or dict(shape=None, dtype="UNKNOWN_CONST", layout="")
        tensors[port] = {**value, "name": tensors[port]["name"], "metadata_source": "Const.value.t.desc"}
    return tensors


def _cost(graph, node_id):
    data = graph.nodes[node_id]
    op, attrs = data["op_type"], data["attributes"]
    if op in _MOVEMENT:
        return 0, 0, "形状变换、类型转换或数据搬移，按 0 FLOPs 计"
    inputs, outputs = _resolved_inputs(graph, node_id), data["output_tensors"]
    if not inputs or inputs[0]["dtype"] not in {"DT_FLOAT", "DT_FLOAT16", "DT_DOUBLE", "DT_BF16"}:
        raise ValueError("输入不是已知浮点类型；不把量化整数运算当作 FLOPs")
    n = prod(_shape(outputs[0]))
    if op in {"MatMul", "BatchMatMul", "BatchMatMulV2", "GemmD", "GemmV3"}:
        if len(inputs) < 2:
            raise ValueError("矩阵乘法缺少两个输入")
        if inputs[1]["dtype"] not in {"DT_FLOAT", "DT_FLOAT16", "DT_DOUBLE", "DT_BF16"}:
            raise ValueError("矩阵权重不是已知浮点类型")
        a, b, y = _shape(inputs[0]), _shape(inputs[1]), _shape(outputs[0])
        if len(a) < 2 or len(b) < 2 or len(y) < 2:
            raise ValueError("矩阵乘法缺少至少二维的逻辑形状")
        if any(t.get("layout", "").startswith("FRACTAL") and len(shape) > 3
               for t, shape in ((inputs[0], a), (inputs[1], b), (outputs[0], y))):
            raise ValueError("无法确认分块矩阵的逻辑 M、K、N，保持计算量未知")
        transpose = []
        for keys in (("adj_x1", "transpose_a", "transpose_x1", "transA"),
                     ("adj_x2", "transpose_b", "transpose_x2", "transB")):
            values = {attrs[k] for k in keys if k in attrs}
            if len(values) > 1 or any(v not in (0, 1) for v in values):
                raise ValueError("矩阵转置属性不一致或无效")
            transpose.append(bool(next(iter(values), False)))
        m, k = a[-2:][::-1] if transpose[0] else a[-2:]
        kb, width = b[-2:][::-1] if transpose[1] else b[-2:]
        if k != kb or y[-2:] != [m, width]:
            raise ValueError("矩阵乘法输入/输出的 M、K、N 不一致")
        length = max(len(a), len(b)) - 2
        aa, bb = [1] * (length - len(a) + 2) + a[:-2], [1] * (length - len(b) + 2) + b[:-2]
        if any(x != z and x != 1 and z != 1 for x, z in zip(aa, bb)):
            raise ValueError("矩阵乘法 batch 无法广播")
        if y[:-2] != [z if x == 1 else x for x, z in zip(aa, bb)]:
            raise ValueError("矩阵乘法输出 batch 与输入广播不一致")
        if attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) not in (0, 1):
            raise ValueError("暂未估算此矩阵乘法 alpha/beta")
        bias = op in {"GemmD", "GemmV3"} and len(inputs) > 2 and attrs.get("beta", 1) != 0
        if bias:
            c = _shape(inputs[2])
            if len(c) > len(y) or any(x not in (1, z) for x, z in zip([1] * (len(y) - len(c)) + c, y)):
                raise ValueError("Gemm 偏置无法广播到输出")
        macs = n * k
        return macs, 2 * macs + n * (bias + bool(attrs.get("relu_flag", False))), \
            "矩阵乘法 MACs=输出元素数×K；FLOPs=2MACs+有效偏置+融合ReLU"
    if attrs.get("alpha", 1) != 1 or attrs.get("beta", 0) != 0:
        raise ValueError("暂未估算非默认 alpha/beta 的融合操作")
    if op == "Convolution":
        weights = [graph.nodes[u].get("constant_tensor") for u, _, e in graph.in_edges(node_id, data=True)
                   if e["input_index"] == 1 and not e["control"]]
        if len(weights) != 1 or weights[0] is None:
            raise ValueError("卷积缺少 Const.value.t.desc 中的逻辑权重形状")
        weight = weights[0]
        w, x, y = _shape(weight), _shape(inputs[0]), _shape(outputs[0])
        groups = attrs.get("groups", 1)
        if weight["layout"] in {"FRACTAL_Z_DW_V2", "FRACTAL_Z_DW_DIAG"}:
            # These DDK depthwise descriptors store logical [1, C, KH, KW].
            # Verify multiplier=1 explicitly before treating them as [C,1,KH,KW].
            if (len(w) != 4 or len(x) != 4 or len(y) != 4
                    or w[:2] != [1, groups] or groups != x[1] or groups != y[1]):
                raise ValueError("深度卷积描述不满足已验证的 [1,C,KH,KW]、multiplier=1 格式")
            w = [w[1], w[0], *w[2:]]
        elif weight["layout"] not in {"NCHW", "ND", "FRACTAL_Z"}:
            raise ValueError(f"未验证的卷积权重布局: {weight['layout']}")
        if (len(w) != 4 or len(x) != 4 or len(y) != 4 or groups < 1
                or attrs.get("data_format", "NCHW") != "NCHW"
                or w[0] != y[1] or w[1] * groups != x[1] or y[1] % groups
                or weight["dtype"] not in {"DT_FLOAT", "DT_FLOAT16", "DT_DOUBLE", "DT_BF16"}):
            raise ValueError("卷积逻辑形状/分组/类型不满足已验证的 NCHW、OIHW 公式")
        macs = n * prod(w[1:])
        bias = len(data["inputs"]) > 2 and bool(data["inputs"][2])
        return macs, 2 * macs + n * (bias + bool(attrs.get("relu_flag", False))), \
            "逻辑输出元素数×权重(C/group×KH×KW)；FLOPs=2MACs+偏置+融合ReLU"
    if op in {"Add", "Sub", "Mul", "RealDiv", "Div"}:
        return 0, n, "每个逻辑输出元素 1 次运算"
    if op == "LayerNorm":
        x = _shape(inputs[0])
        axis = attrs.get("begin_norm_axis")
        if (type(axis) is not int or not -len(x) <= axis < len(x)
                or attrs.get("end_norm_axis", -1) not in (-1, len(x) - 1)
                or attrs.get("norm_type", "LayerNorm") != "LayerNorm"
                or _shape(outputs[0]) != x):
            raise ValueError("LayerNorm 缺少明确的归一化轴或形状")
        groups = prod(x[:axis % len(x)])
        width = prod(x[axis % len(x):])
        if not width:
            raise ValueError("不估算空集合的 LayerNorm")
        if any(prod(_shape(t)) not in (1, width) for t in inputs[1:3]):
            raise ValueError("LayerNorm scale/bias 与归一化宽度不一致")
        if any(prod(_shape(t)) != groups for t in outputs[1:3]):
            raise ValueError("LayerNorm mean/variance 与分组数量不一致")
        return 0, (6 + (len(inputs) > 2)) * n + 3 * groups, "LayerNorm 近似 6N+3组数，有偏置另加 N"
    if op == "PRelu":
        x = _shape(inputs[0])
        fmt = attrs.get("data_format", "NCHW")
        if len(inputs) < 2 or len(x) < 2 or fmt not in {"NCHW", "NHWC"} or _shape(outputs[0]) != x:
            raise ValueError("PRelu 缺少有效输入形状或布局")
        if prod(_shape(inputs[1])) not in {1, x[1] if fmt == "NCHW" else x[-1]}:
            raise ValueError("PRelu slope 不是标量或逐通道参数")
        return 0, 2 * n, "PRelu 比较及负分支乘法，按每元素约 2 次标量操作估算"
    if op == "Activation":
        factors = {0: 4, 1: 1, 2: 6, 3: 2, 5: 2, 6: 1, 7: 2, 14: 2}
        mode = attrs.get("mode", 1)
        if mode not in factors:
            raise ValueError(f"暂未估算 Activation mode={mode}")
        return 0, n * factors[mode], f"Activation mode={mode}，每元素约 {factors[mode]} 次运算"
    if op == "Eltwise":
        count = sum(bool(x) and not x.endswith(":-1") for x in data["inputs"])
        coeff = attrs.get("coeff", [])
        if attrs.get("mode", 1) not in (0, 1, 2) or count < 1 or any(c != 1 for c in coeff):
            raise ValueError("暂未估算此 Eltwise 模式/系数")
        return 0, n * (count - 1), "逐元素乘积/求和/最大值：N×(输入数量-1)"
    if op == "PoolingD":
        window = attrs.get("window", [1, 1])
        if attrs.get("global_pooling", False):
            x = _shape(inputs[0])
            if len(x) != 4 or inputs[0]["layout"] not in {"NCHW", "NC1HWC0"}:
                raise ValueError("全局池化缺少明确的 NCHW 逻辑形状")
            window = x[2:]
        mode = attrs.get("mode", 0)
        if mode not in (0, 1) or not window or any(d < 1 for d in window):
            raise ValueError("暂未估算此 PoolingD 模式/窗口")
        return 0, n * (prod(window) - (mode == 0)), "平均池化 N×窗口大小；最大池化 N×(窗口大小-1)"
    if op == "Softmax":
        dims = _shape(inputs[0])
        axis = attrs.get("axis", 0)
        if not -len(dims) <= axis < len(dims) or dims[axis] <= 0:
            raise ValueError("Softmax axis/形状无效")
        return 0, 5 * n - 2 * (n // dims[axis]), "稳定 Softmax，约 5N-2组数"
    raise ValueError(f"暂未实现 GE {op} 的计算量公式；保留名称、依赖和属性")


def statistics_nodes(graph):
    nodes, excluded = [], []
    for node_id, data in graph.nodes(data=True):
        if data["op_type"] in _STRUCTURAL or data["is_container"]:
            excluded.append(node_id)
            continue
        record = {"node_id": node_id, "name": data["name"], "original_name": data["original_name"],
                  "operator": data["op_type"], "hierarchy": list(data["hierarchy"]),
                  "hierarchy_path": data["hierarchy_path"], "hierarchy_reason": data["hierarchy_reason"],
                  "region_id": data["region_id"], "inputs": _resolved_inputs(graph, node_id),
                  "outputs": data["output_tensors"]}
        try:
            record["macs"], record["flops"], record["formula"] = _cost(graph, node_id)
        except (ValueError, KeyError, IndexError) as exc:
            record.update(macs=None, flops=None, reason=str(exc))
        nodes.append(record)
    return nodes, excluded
