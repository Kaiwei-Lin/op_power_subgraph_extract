"""NodeProto is authoritative; NetworkX edges and hierarchy are derived indexes."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import re
from typing import Mapping

import networkx as nx
import onnx
from onnx import AttributeProto, ModelProto, NodeProto, ValueInfoProto, helper


def _flat(name: str) -> str:
    return name.strip("/*").replace("/", "_").replace("*", "_")


def _tensor_stem(name: str) -> str:
    return re.sub(r"(?:_output_\d+|:\d+)$", "", name)


def _name_candidate(node: NodeProto) -> tuple[list[str], str]:
    outputs = {_tensor_stem(t) for t in node.output if "/" in t}
    exact = sorted(t for t in outputs if _flat(t) == _flat(node.name))
    if exact:
        return exact, "output"

    # Input paths belong to upstream nodes: only borrow a parent scope when the
    # remaining name is this node's operator (possibly with a numeric suffix).
    candidates = []
    for tensor in [*node.output, *node.input]:
        parts = _tensor_stem(tensor).split("/")
        for end in range(1, len(parts)):
            scope = "/".join(parts[:end])
            prefix = _flat(scope) + "_"
            if scope.strip("/") and _flat(node.name).startswith(prefix):
                tail = _flat(node.name)[len(prefix):]
                if re.fullmatch(re.escape(node.op_type) + r"(?:_+\d+)?", tail):
                    candidates.append(scope + "/" + tail)
    return sorted(set(candidates)), "io_scope"


def _fill_graph_names(graph: onnx.GraphProto, fallback: str = "onnx_graph") -> None:
    """Graph names are required by the checker, unlike optional operator names."""
    if not graph.name.strip():
        graph.name = fallback
    for node_index, node in enumerate(graph.node):
        for attr_index, attr in enumerate(node.attribute):
            graphs = [attr.g] if attr.type == AttributeProto.GRAPH and attr.HasField("g") else attr.graphs
            for graph_index, subgraph in enumerate(graphs):
                _fill_graph_names(subgraph, f"{fallback}_node_{node_index}_attr_{attr_index}_{graph_index}")


def repair_node_names(model: ModelProto) -> tuple[ModelProto, list[dict]]:
    """Return an independent model and a per-node report; never rename tensors.

    Recover / boundaries only with matching tensor-path evidence. Empty or
    whitespace-only names are skipped, preserving unnamed boundary operators.
    Existing / names, ambiguous paths, and name collisions are left unchanged.
    Operator renaming applies to the main graph. Required GraphProto names are
    filled when missing, including control-flow subgraphs.
    """
    result = deepcopy(model)
    _fill_graph_names(result.graph)
    original_names = Counter(n.name for n in result.graph.node if n.name)
    proposals = []
    for index, node in enumerate(result.graph.node):
        candidates, source = [], ""
        if not node.name.strip():
            status = "skipped_unnamed"
        elif "/" in node.name:
            status = "already_hierarchical"
        else:
            candidates, source = _name_candidate(node)
            status = "renamed" if len(candidates) == 1 else "ambiguous" if candidates else "unresolved"
        proposals.append(dict(node_id=index, old_name=node.name, new_name=node.name,
                              status=status, source=source, candidates=candidates))
    counts = Counter(r["candidates"][0] for r in proposals if r["status"] == "renamed")
    for node, record in zip(result.graph.node, proposals):
        if record["status"] != "renamed":
            continue
        target = record["candidates"][0]
        if counts[target] > 1 or original_names[target]:
            record["status"] = "collision"
        else:
            node.name = record["new_name"] = target
    return result, proposals


def _node_inputs(node: NodeProto):
    """Include lexical captures in If/Loop/Scan GraphProto attributes."""
    yield from ((name, index) for index, name in enumerate(node.input) if name)
    captures = set(node.input)
    for attr in node.attribute:
        graphs = [attr.g] if attr.type == AttributeProto.GRAPH else attr.graphs
        for graph in graphs:
            defined = {v.name for v in graph.input}
            defined.update(t.name for t in graph.initializer)
            defined.update(t.values.name for t in graph.sparse_initializer)
            defined.update(t for n in graph.node for t in n.output if t)
            used = [name for n in graph.node for name, _ in _node_inputs(n)]
            used.extend(v.name for v in graph.output)
            for name in used:
                if name and name not in defined and name not in captures:
                    captures.add(name)
                    yield name, None


def refresh_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Rebuild edges and read-only convenience fields after editing node['proto']."""
    if graph.graph.get("source_format") == "omc":
        raise ValueError("OMC 图仅支持读取分析和前缀提取；refresh_graph 只适用于 ONNX NodeProto")
    if nx.is_frozen(graph):
        graph = graph.copy()
    producers = {}
    hierarchy = defaultdict(list)
    graph.remove_edges_from(list(graph.edges(keys=True)))
    for node_id, data in graph.nodes(data=True):
        node = data["proto"]
        parts = tuple(part for part in node.name.split("/") if part) if node.name.strip() else ()
        data.update(name=node.name, op_type=node.op_type, domain=node.domain,
                    inputs=tuple(node.input), outputs=tuple(node.output),
                    attributes=tuple(node.attribute), doc_string=node.doc_string,
                    hierarchy=parts, scope="/".join(parts[:-1]))
        hierarchy[""].append(node_id)
        for end in range(1, len(parts) + 1):
            hierarchy["/".join(parts[:end])].append(node_id)
        for output_index, name in enumerate(node.output):
            if not name:
                continue
            if name in producers:
                raise ValueError(f"Multiple producers for tensor {name!r}")
            producers[name] = (node_id, output_index)
    for target, data in graph.nodes(data=True):
        for tensor, input_index in _node_inputs(data["proto"]):
            if tensor in producers:
                source, output_index = producers[tensor]
                graph.add_edge(source, target, tensor=tensor, input_index=input_index,
                               output_index=output_index, implicit=input_index is None)
    graph.graph["hierarchy"] = dict(hierarchy)
    return graph


def onnx_to_networkx(source: str | Path | ModelProto, *, repair_names: bool = False) -> nx.MultiDiGraph:
    """One integer-ID vertex per ONNX operator; one edge per input connection.

    graph.graph['model'] holds all ModelProto/GraphProto fields, initializers,
    value_info and local functions. Every vertex holds its complete NodeProto.
    Loading a path also loads external tensor data, making the result portable.
    """
    model = deepcopy(source) if isinstance(source, ModelProto) else onnx.load(str(source))
    original_names = [n.name for n in model.graph.node]
    report = []
    if repair_names:
        model, report = repair_node_names(model)
    graph = nx.MultiDiGraph()
    graph.graph.update(model=model, original_node_ids=tuple(range(len(model.graph.node))),
                       rename_report=report)
    for index, node in enumerate(model.graph.node):
        graph.add_node(index, proto=deepcopy(node), original_name=original_names[index], original_index=index)
    return refresh_graph(graph)


def nodes_by_prefix(graph: nx.MultiDiGraph, prefix: str) -> list:
    """Match a complete / path component; leading/trailing / are optional."""
    if graph.graph.get("source_format") == "omc":
        parts = tuple(p for p in prefix.split("/") if p)
        return [i for i, d in graph.nodes(data=True) if d["hierarchy"][:len(parts)] == parts]
    graph = refresh_graph(graph)
    return list(graph.graph["hierarchy"].get(prefix.strip("/"), []))


def extract_prefix(graph: nx.MultiDiGraph, prefix: str) -> nx.MultiDiGraph:
    """Copy the induced subgraph, retaining source metadata for boundary export."""
    selected = nodes_by_prefix(graph, prefix)
    if not selected:
        kind = "OMC" if graph.graph.get("source_format") == "omc" else "ONNX"
        raise ValueError(f"No {kind} nodes match hierarchy prefix {prefix!r}")
    result = deepcopy(graph.subgraph(selected).copy())
    if graph.graph.get("source_format") == "omc":
        result.graph["hierarchy"] = {p: [i for i in ids if i in result]
                                     for p, ids in result.graph["hierarchy"].items()}
        return result
    return refresh_graph(result)


def _value_infos(model: ModelProto) -> dict[str, ValueInfoProto]:
    infos = {v.name: v for v in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
    for tensor in model.graph.initializer:
        infos.setdefault(tensor.name, helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims))
    for tensor in model.graph.sparse_initializer:
        infos.setdefault(tensor.values.name, helper.make_tensor_value_info(
            tensor.values.name, tensor.values.data_type, tensor.dims))
    return infos


def _has_type(info: ValueInfoProto | None) -> bool:
    if info is None or not info.HasField("type"):
        return False
    kind = info.type.WhichOneof("value")
    if kind in ("tensor_type", "sparse_tensor_type"):
        value = getattr(info.type, kind)
        return bool(value.elem_type) and value.HasField("shape")
    return kind is not None


def networkx_to_onnx(
    graph: nx.MultiDiGraph, *, infer_shapes: bool = True,
    value_info: Mapping[str, ValueInfoProto] | None = None, check: bool = True,
) -> ModelProto:
    """Export current NodeProtos in dependency order, deriving subgraph I/O.

    For a full graph, preserve model fields except missing required graph names,
    which receive deterministic defaults, without shape inference. For a
    selection, retain required weights and expose external activations as inputs.
    Missing boundary types raise an error; pass explicit value_info for custom
    operators that ONNX shape inference cannot handle.
    """
    if graph.graph.get("source_format") == "omc":
        raise ValueError("不能将 OMC 编译图直接导出为可执行 ONNX；请用原始 ONNX 提取后重新编译")
    if not graph and graph.graph["original_node_ids"]:
        raise ValueError("Cannot export an empty ONNX graph")
    graph = refresh_graph(graph)
    positions = {node_id: i for i, node_id in enumerate(graph)}
    try:
        order = list(nx.lexicographical_topological_sort(graph, key=positions.__getitem__))
    except nx.NetworkXUnfeasible as error:
        raise ValueError("ONNX dependencies contain a cycle") from error
    template = graph.graph["model"]
    result = deepcopy(template)
    nodes = [graph.nodes[i]["proto"] for i in order]
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    _fill_graph_names(result.graph)
    selected_indices = {data.get("original_index") for _, data in graph.nodes(data=True)}
    if len(graph) == len(graph.graph["original_node_ids"]) and selected_indices == set(graph.graph["original_node_ids"]):
        if check:
            onnx.checker.check_model(result)
        return result
    if template.training_info:
        raise ValueError("Subgraph extraction of training bindings is unsupported; export the full graph")

    produced = {t for n in nodes for t in n.output if t}
    required = dict.fromkeys(t for n in nodes for t, _ in _node_inputs(n))
    original_outputs = {v.name for v in template.graph.output}
    external_uses = {
        t for i, n in enumerate(template.graph.node) if i not in selected_indices
        for t, _ in _node_inputs(n)
    }
    outputs = list(dict.fromkeys(
        t for n in nodes for t in n.output
        if t and (t in original_outputs or t in external_uses or t not in required)
    ))
    # Quantization annotations can refer to scale/zero-point initializers that
    # are not explicit node inputs; keep these dependencies as well.
    relevant = produced | set(required)
    annotations = []
    for annotation in template.graph.quantization_annotation:
        if annotation.tensor_name in relevant:
            annotations.append(annotation)
            required.update(dict.fromkeys(p.value for p in annotation.quant_parameter_tensor_names))
    weights = {t.name for t in template.graph.initializer}
    weights.update(t.values.name for t in template.graph.sparse_initializer)
    external = set(required) - produced
    inputs = [v.name for v in template.graph.input if v.name in external]
    inputs.extend(t for t in required if t in external - weights and t not in inputs)

    infos = _value_infos(template)
    overrides = dict(value_info or {})
    for name, info in overrides.items():
        if name != info.name:
            raise ValueError(f"value_info key {name!r} does not match {info.name!r}")
    inference_error = None
    if infer_shapes and any(not _has_type(overrides.get(t, infos.get(t))) for t in inputs + outputs):
        try:
            context = deepcopy(template)
            for _, data in graph.nodes(data=True):
                i = data.get("original_index")
                if isinstance(i, int) and 0 <= i < len(context.graph.node):
                    context.graph.node[i].CopyFrom(data["proto"])
            _fill_graph_names(context.graph)
            inferred = _value_infos(onnx.shape_inference.infer_shapes(context))
            infos.update({name: info for name, info in inferred.items() if _has_type(info)})
        except (onnx.shape_inference.InferenceError, RuntimeError, ValueError) as error:
            inference_error = str(error)
    infos.update(overrides)
    missing = [name for name in inputs + outputs if not _has_type(infos.get(name))]
    if missing:
        detail = f" Shape inference: {inference_error}" if inference_error else ""
        raise ValueError(f"Missing boundary type/shape for {missing}; supply value_info with ValueInfoProto objects.{detail}")

    for field in ("input", "output", "initializer", "sparse_initializer", "value_info", "quantization_annotation"):
        result.graph.ClearField(field)
    result.graph.input.extend(infos[name] for name in inputs)
    result.graph.output.extend(infos[name] for name in outputs)
    result.graph.initializer.extend(t for t in template.graph.initializer if t.name in required)
    result.graph.sparse_initializer.extend(t for t in template.graph.sparse_initializer if t.values.name in required)
    retained = produced | set(required)
    result.graph.value_info.extend(info for name, info in infos.items() if name in retained - set(inputs) - set(outputs))
    result.graph.quantization_annotation.extend(annotations)
    if check:
        onnx.checker.check_model(result)
    return result
