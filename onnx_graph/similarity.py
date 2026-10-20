"""Group exported ONNX subgraphs for performance sampling; never delete models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from hashlib import sha256
import heapq
import json
import math
from pathlib import Path
import re

import networkx as nx
import numpy as np
import onnx
from onnx import AttributeProto as A, TensorProto as T, helper, numpy_helper


PACKAGE_DIR = Path(__file__).resolve().parent
INPUT_DIR = PACKAGE_DIR / "models/handfree_ainr/networkx/main_example/batch"
OUTPUT_DIR = None  # None: INPUT_DIR / grouping_<GROUP_MODE>
GROUP_MODE = "exact"  # exact: 配置同构；similar: 相似度达到阈值。
SIMILARITY_THRESHOLD = 0.90
SEPARATE_SLASH_COUNTS = True
WL_ROUNDS = 2

# 只忽略直接作为这些数据参数使用的浮点权重；形状、索引、量化参数等保留值。
_WEIGHT_PORTS = {
    "Conv": {1, 2}, "ConvTranspose": {1, 2}, "Gemm": {1, 2}, "MatMul": {0, 1},
    "BatchNormalization": {1, 2, 3, 4}, "InstanceNormalization": {1, 2},
    "LayerNormalization": {1, 2}, "PRelu": {1},
    "Add": {0, 1}, "Sub": {0, 1}, "Mul": {0, 1}, "Div": {0, 1},
}
_FLOAT_TYPES = {T.FLOAT, T.DOUBLE, T.FLOAT16, T.BFLOAT16, T.COMPLEX64, T.COMPLEX128}
_CONTROL_PORTS = {
    "Reshape": {1}, "Expand": {1}, "Tile": {1}, "Slice": {1, 2, 3, 4},
    "Resize": {1, 2, 3}, "Upsample": {1}, "Pad": {1, 2, 3}, "Clip": {1, 2},
    "Squeeze": {1}, "Unsqueeze": {1}, "Split": {1}, "TopK": {1},
    "Gather": {1}, "GatherElements": {1}, "GatherND": {1},
    "Scatter": {1}, "ScatterElements": {1}, "ScatterND": {1},
    "ConstantOfShape": {0}, "Range": {0, 1, 2}, "OneHot": {1, 2},
    "Dropout": {1, 2}, "Trilu": {1},
}


def _hash(value):
    return sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_key(tensor, ignore_values=False):
    result = [tensor.data_type, list(tensor.dims)]
    if ignore_values and tensor.data_type in _FLOAT_TYPES:
        return result + ["weight_values_ignored"]
    array = numpy_helper.to_array(tensor)
    if array.dtype.hasobject:  # 字符串数组的 tobytes() 包含指针，不能作为内容摘要。
        payload = _hash([v.hex() if isinstance(v, bytes) else str(v)
                         for v in array.reshape(-1)]).encode()
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return result + [sha256(payload).hexdigest()]


def _attribute_key(attribute, ignore_values=False):
    if attribute.type in (A.GRAPH, A.GRAPHS, A.SPARSE_TENSOR, A.SPARSE_TENSORS):
        raise ValueError("暂不比较控制流子图或稀疏属性，需要单独测试此 ONNX")
    if attribute.type == A.TENSOR:
        return [attribute.name, attribute.type, _tensor_key(attribute.t, ignore_values)]
    if attribute.type == A.TENSORS:
        return [attribute.name, attribute.type, [_tensor_key(t) for t in attribute.tensors]]
    value = deepcopy(attribute)
    value.ClearField("doc_string")
    return [value.name, value.SerializeToString(deterministic=True).hex()]


def _type_key(info):
    if info is None or not info.type.HasField("tensor_type"):
        return None, False, False
    tensor = info.type.tensor_type
    if not tensor.elem_type or not tensor.HasField("shape"):
        return [tensor.elem_type, None], False, False
    dims = []
    complete, dynamic = True, False
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
            dynamic = True
        else:
            dims.append(None)
            complete = False
    return [tensor.elem_type, dims], complete, dynamic


def wl_features(graph, label="config", rounds=WL_ROUNDS):
    """Directed, port-aware WL features; names and node IDs are never features."""
    labels = {n: data[label] for n, data in graph.nodes(data=True)}
    features = Counter((0, v) for v in labels.values())
    for step in range(1, rounds + 1):
        labels = {
            n: _hash([labels[n],
                      sorted((d["port"], labels[u]) for u, _, d in graph.in_edges(n, data=True)),
                      sorted((d["port"], labels[v]) for _, v, d in graph.out_edges(n, data=True))])
            for n in graph
        }
        features.update((step, v) for v in labels.values())
    return features


def feature_similarity(left, right):
    """Weighted Jaccard counts, in [0, 1]; unlike cosine, penalizes graph size."""
    if len(left) > len(right):
        left, right = right, left
    common = sum(min(count, right.get(key, 0)) for key, count in left.items())
    union = sum(left.values()) + sum(right.values()) - common
    return common / union if union else 1.0


def describe_model(source, rounds=WL_ROUNDS):
    model = onnx.load(source) if not isinstance(source, onnx.ModelProto) else deepcopy(source)
    if model.functions or model.graph.sparse_initializer or model.graph.quantization_annotation:
        raise ValueError("暂不比较局部函数、稀疏权重或量化注解，需要单独测试此 ONNX")
    if not model.graph.node:
        raise ValueError("ONNX 没有运算节点")
    onnx.checker.check_model(model)
    notes = []
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as error:
        notes.append(f"形状推断未完成: {type(error).__name__}: {error}")
    proto = model.graph
    infos = {v.name: v for v in [*proto.value_info, *proto.input, *proto.output]}
    initializers = {t.name: t for t in proto.initializer}
    inputs, outputs = {v.name for v in proto.input}, {v.name for v in proto.output}
    consumers = defaultdict(list)
    producers, protected = {}, set()
    for node in proto.node:
        for port, name in enumerate(node.input):
            consumers[name].append((node, port))
            if (node.domain or port in _CONTROL_PORTS.get(node.op_type, set())
                    or node.op_type.startswith("Reduce") and port > 0
                    or node.op_type in ("QuantizeLinear", "DequantizeLinear") and port > 0):
                protected.add(name)
        producers.update((name, node) for name in node.output if name)
    pending = list(protected)
    while pending:
        node = producers.get(pending.pop())
        if node is not None:
            for name in node.input:
                if name and name not in protected:
                    protected.add(name)
                    pending.append(name)

    def weight_only(name):
        return bool(consumers[name]) and name not in inputs | outputs | protected and all(
            not n.domain and port in _WEIGHT_PORTS.get(n.op_type, set())
            for n, port in consumers[name])

    for name, tensor in initializers.items():
        infos[name] = helper.make_tensor_value_info(name, tensor.data_type, list(tensor.dims))
    tensors = {name for node in proto.node for name in [*node.input, *node.output] if name}
    tensors.update(inputs | outputs | set(initializers))
    graph = nx.MultiDiGraph()
    complete, dynamic = True, False
    for name in sorted(tensors):
        dtype_shape, known, symbolic = _type_key(infos.get(name))
        complete &= known
        dynamic |= symbolic
        roles = [name in inputs, name in outputs, name in initializers]
        constant = _tensor_key(initializers[name], weight_only(name)) if name in initializers else None
        graph.add_node(("tensor", name), structure=_hash(["tensor", roles]),
                       config=_hash(["tensor", roles, dtype_shape, constant]))
    versions = {op.domain: op.version for op in model.opset_import}
    for index, node in enumerate(proto.node):
        ignore = (not node.domain and node.op_type == "Constant" and len(node.output) == 1
                  and weight_only(node.output[0]))
        attributes = [_attribute_key(a, ignore) for a in sorted(node.attribute, key=lambda a: a.name)]
        operation = [node.domain, node.op_type]
        graph.add_node(("op", index), structure=_hash(operation),
                       config=_hash([operation, versions.get(node.domain), attributes,
                                     [bool(v) for v in node.input], [bool(v) for v in node.output]]))
        for port, name in enumerate(node.input):
            if name:
                graph.add_edge(("tensor", name), ("op", index), port=port)
        for port, name in enumerate(node.output):
            if name:
                graph.add_edge(("op", index), ("tensor", name), port=port)
    if not complete:
        notes.append("有张量缺少类型/形状信息，保留为独立组")
    if dynamic:
        notes.append("含动态维度；同组测试仍需使用相同的实际尺寸")
    return dict(graph=graph, config=wl_features(graph, "config", rounds),
                structure=wl_features(graph, "structure", rounds), complete=complete,
                dynamic=dynamic, notes=notes, node_count=len(proto.node),
                operator_counts=dict(Counter(n.op_type for n in proto.node)))


def same_configuration(left, right):
    if not left["complete"] or not right["complete"] or left["config"] != right["config"]:
        return False
    # WL 特征相同并不能证明同构；还要核对方向、端口、多重边和全部配置标签。
    return nx.is_isomorphic(
        left["graph"], right["graph"],
        node_match=lambda a, b: a["config"] == b["config"],
        edge_match=lambda a, b: Counter(v["port"] for v in a.values())
        == Counter(v["port"] for v in b.values()))


def _exact_groups(items):
    buckets, groups = defaultdict(list), []
    for index, item in enumerate(items):
        key = tuple(sorted(item["config"].items()))
        for group in buckets[key]:
            if same_configuration(items[group[0]], item):
                group.append(index)
                break
        else:
            group = [index]
            groups.append(group)
            buckets[key].append(group)
    return groups


def complete_link_groups(scores, threshold, eligible=None):
    """Deterministic complete-link agglomeration; every pair meets threshold."""
    active = {i: [i] for i in range(len(scores))}
    eligible = [True] * len(scores) if eligible is None else eligible
    links = scores.copy()
    heap = [(-float(links[i, j]), i, j) for i in active for j in range(i + 1, len(scores))
            if eligible[i] and eligible[j] and links[i, j] >= threshold]
    heapq.heapify(heap)
    while heap:
        negative, i, j = heapq.heappop(heap)
        if i not in active or j not in active or -negative != links[i, j]:
            continue
        active[i].extend(active.pop(j))
        for k in active:
            if k != i:
                links[i, k] = links[k, i] = min(links[i, k], links[j, k])
                if eligible[k] and links[i, k] >= threshold:
                    heapq.heappush(heap, (-float(links[i, k]), min(i, k), max(i, k)))
    return [sorted(group) for _, group in sorted(active.items())]


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _manifest_names(input_dir):
    path = input_dir / "batch_export.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8-sig"))["records"]
    # 使用层级+文件名关联；复制到公司 PC 后清单中的旧绝对路径也不影响读取。
    return {(f"slash_{r['slash_count']}", r["onnx_file"].replace("\\", "/").rsplit("/", 1)[-1]):
            r["prefix"] for r in records if r.get("onnx_file") and r.get("status") in ("exported", "exists")}


def group_subgraphs(input_dir, output_dir=None, *, mode=GROUP_MODE,
                    threshold=SIMILARITY_THRESHOLD, separate_slash_counts=SEPARATE_SLASH_COUNTS,
                    rounds=WL_ROUNDS, model_paths=None):
    if mode not in ("exact", "similar"):
        raise ValueError("mode 必须是 exact 或 similar")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold 必须在 (0, 1] 之间")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 10:
        raise ValueError("rounds 必须是 1 到 10 的整数")
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else input_dir / f"grouping_{mode}"
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if input_dir == output_dir or input_dir.is_relative_to(output_dir):
        raise ValueError("输出目录不能等于输入目录或是其父目录")
    if model_paths is None:
        files = sorted(p for p in input_dir.rglob("*.onnx") if not p.is_relative_to(output_dir))
        if not files:
            raise FileNotFoundError(f"{input_dir} 中没有 .onnx 子图")
    else:
        # 批量导出只比较本次选中的文件，不让旧运行的模型成为当前筛选结果的代表。
        files = sorted({Path(p).resolve() for p in model_paths})
        if any(not p.is_relative_to(input_dir) or p.is_relative_to(output_dir)
               or p.suffix.lower() != ".onnx" or not p.is_file() for p in files):
            raise ValueError("model_paths 必须是输入目录内、报告目录外的现有 ONNX 文件")
    names = _manifest_names(input_dir)
    levels, failures = defaultdict(list), []
    for index, path in enumerate(files):
        parts = path.relative_to(input_dir).parts[:-1]
        slash = next((p for p in parts if re.fullmatch(r"slash_\d+", p)), None)
        if slash is None and re.fullmatch(r"slash_\d+", input_dir.name):
            slash = input_dir.name
        level = (slash or "unclassified") if separate_slash_counts else "all"
        try:
            item = describe_model(path, rounds)
            item.update(path=str(path), name=names.get((slash, path.name), path.stem))
            levels[level].append(item)
        except Exception as error:
            failures.append(dict(path=str(path), reason=f"{type(error).__name__}: {error}"))
        if (index + 1) % 25 == 0 or index + 1 == len(files):
            print(f"Read subgraphs: {index + 1}/{len(files)}")

    report = dict(input_dir=str(input_dir), output_dir=str(output_dir), mode=mode,
                  threshold=threshold, wl_rounds=rounds, separate_slash_counts=separate_slash_counts,
                  files=len(files), failures=failures, levels=[])
    for level, items in sorted(levels.items()):
        count = len(items)
        scores = np.eye(count, dtype=np.float64)
        structural = np.eye(count, dtype=np.float64)
        for i in range(count):
            for j in range(i + 1, count):
                scores[i, j] = scores[j, i] = feature_similarity(items[i]["config"], items[j]["config"])
                structural[i, j] = structural[j, i] = feature_similarity(
                    items[i]["structure"], items[j]["structure"])
        exact = _exact_groups(items)
        exact_ids = {i: f"E{number:04d}" for number, group in enumerate(exact, 1) for i in group}
        groups = exact if mode == "exact" else complete_link_groups(
            scores, threshold, [item["complete"] for item in items])
        members, representatives = [], []
        folder = output_dir / level
        for number, group in enumerate(groups, 1):
            representative = max(group, key=lambda i: sum(scores[i, j] for j in group))
            minimum = min(scores[i, j] for i in group for j in group)
            basis = ("单独保留" if len(group) == 1 else
                     "配置相同" if len({exact_ids[i] for i in group}) == 1 else "高度相似")
            for i in group:
                item, chosen = items[i], items[representative]
                record = dict(group=f"G{number:04d}", name=item["name"], path=item["path"],
                              node_count=item["node_count"], operator_counts=item["operator_counts"],
                              complete_shapes=item["complete"], dynamic_shapes=item["dynamic"],
                              exact_group=exact_ids[i], representative=chosen["name"],
                              representative_path=chosen["path"], is_representative=i == representative,
                              similarity=float(scores[i, representative]), minimum_similarity=float(minimum),
                              basis=basis, notes=item["notes"])
                members.append(record)
                if i == representative:
                    representatives.append(record)
        fields = ["分组", "子图名字", "ONNX文件", "节点数", "算子数量", "形状信息完整", "含动态维度",
                  "精确配置组", "代表子图", "代表ONNX", "是否代表", "与代表相似度",
                  "组内最低相似度", "分组依据", "提示"]
        for filename, records in [("groups.csv", members), ("representatives.csv", representatives)]:
            _write_csv(folder / filename, fields, (
                [r["group"], r["name"], r["path"], r["node_count"],
                 json.dumps(r["operator_counts"], ensure_ascii=False), r["complete_shapes"],
                 r["dynamic_shapes"], r["exact_group"], r["representative"], r["representative_path"],
                 r["is_representative"], r["similarity"], r["minimum_similarity"], r["basis"],
                 "；".join(r["notes"])] for r in records))
        _write_csv(folder / "similarity.csv",
                   ["子图A", "子图B", "ONNX文件A", "ONNX文件B", "结构相似度", "配置相似度", "配置相同"],
                   ([items[i]["name"], items[j]["name"], items[i]["path"], items[j]["path"],
                     float(structural[i, j]), float(scores[i, j]), exact_ids[i] == exact_ids[j]]
                    for i in range(count) for j in range(i + 1, count)))
        report["levels"].append(dict(level=level, models=count, exact_groups=len(exact),
                                     groups=len(groups), representatives=representatives, members=members))
        print(f"{level}: {count} models -> {len(groups)} groups ({len(exact)} exact configurations)")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "failures.csv", ["ONNX文件", "原因"],
               ([r["path"], r["reason"]] for r in failures))
    (output_dir / "grouping.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Grouping report: {output_dir}; failed: {len(failures)} (keep these models for separate testing)")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=["exact", "similar"], default=GROUP_MODE)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--merge-levels", action="store_true", help="允许不同 slash 层级一起比较")
    args = parser.parse_args(argv)
    return group_subgraphs(args.input_dir, args.output_dir, mode=args.mode, threshold=args.threshold,
                           separate_slash_counts=SEPARATE_SLASH_COUNTS and not args.merge_levels)


if __name__ == "__main__":
    main()
