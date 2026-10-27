"""把运行时算子的平均耗时映射到 ONNX 节点；不猜测融合算子的归属。"""

import csv
from collections import defaultdict
from math import fsum, isfinite
from pathlib import Path
import re

if __package__:
    from .core import _alias_flat, _flat, load_name_aliases
else:
    from core import _alias_flat, _flat, load_name_aliases


def _match_name(name, exact, flattened, aliases):
    # 原始名称可能本身含括号，因此先精确匹配，再尝试去掉运行时类型后缀。
    stripped = re.sub(r"\([^()]*\)$", "", name).strip()
    for candidate, method in [(name, "exact"), (stripped, "type_suffix_removed")]:
        if candidate and candidate in exact:
            return sorted(exact[candidate]), method
    for candidate in (name, stripped):
        key = _flat(candidate)
        if key and key in flattened:
            return sorted(flattened[key]), "unique_flattened_name"
    for candidate in (name, stripped):
        key = _alias_flat(candidate, aliases)
        if key and key in flattened:
            return sorted(flattened[key]), "alias_flattened_name"
    return [], "unmatched"


def attach_profile(nodes, csv_path, *, name_aliases=None):
    """每个运行算子多条记录取均值；同一 ONNX 节点的不同运行算子均值相加。"""
    path = Path(csv_path)
    aliases = load_name_aliases(name_aliases)
    exact, flattened = defaultdict(set), defaultdict(set)
    for node in nodes:
        node.update(exec_time=None, exec_time_samples=0, profile_entries=[])
        for name in {node["name"].strip(), node["original_name"].strip()} - {""}:
            exact[name].add(node["node_id"])
            flattened[_flat(name)].add(node["node_id"])
            flattened[_alias_flat(name, aliases)].add(node["node_id"])

    groups, models = {}, set()
    rows_read = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        fields = [x.strip().lower() for x in next(reader, [])]
        if not fields or len(set(fields)) != len(fields) or "" in fields:
            raise ValueError("运行时间 CSV 的表头为空或有重复/空字段")
        if "total time(us)" not in fields or not {"op name", "onnx_node_name"}.intersection(fields):
            raise ValueError("运行时间 CSV 需要 'op name'（或 'onnx_node_name'）及 'total time(us)' 列；时间单位为微秒")
        for line, values in enumerate(reader, 2):
            if not values or not any(x.strip() for x in values):
                continue
            if len(values) != len(fields):
                raise ValueError(f"运行时间 CSV 第 {line} 行的字段数量与表头不同")
            row = dict(zip(fields, (x.strip() for x in values)))
            rows_read += 1
            if row.get("model name"):
                models.add(row["model name"])
            runtime_name = row.get("op name") or row.get("onnx_node_name", "")
            lookup_name = row.get("onnx_node_name") or runtime_name
            key = (runtime_name, lookup_name)
            group = groups.setdefault(key, {"profile_op_name": runtime_name, "lookup_name": lookup_name,
                                            "rows": 0, "samples": [], "first_csv_row": line})
            group["rows"] += 1
            value = row["total time(us)"]
            if value:
                try:
                    duration = float(value)
                except ValueError as exc:
                    raise ValueError(f"运行时间 CSV 第 {line} 行的 total time(us) 不是数值: {value!r}") from exc
                if not isfinite(duration) or duration < 0:
                    raise ValueError(f"运行时间 CSV 第 {line} 行的时间必须为有限非负数: {value!r}")
                group["samples"].append(duration)
    if not rows_read:
        raise ValueError("运行时间 CSV 没有数据行")
    if len(models) > 1:
        raise ValueError(f"运行时间 CSV 包含多个模型，请先筛选出当前模型: {sorted(models)}")

    by_id = {n["node_id"]: n for n in nodes}
    measurements = defaultdict(list)
    entries = []
    for group in groups.values():
        samples = group.pop("samples")
        candidates, method = _match_name(group["lookup_name"], exact, flattened, aliases)
        mean = fsum(x / len(samples) for x in samples) if samples else None
        status = "matched" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched"
        if status == "matched" and mean is None:
            status = "missing_time"
        group.update(status=status, match_method=method, candidate_node_ids=candidates,
                     sample_count=len(samples), mean_exec_time=mean)
        if status == "matched":
            node = by_id[candidates[0]]
            node["profile_entries"].append(len(entries))
            node["exec_time_samples"] += len(samples)
            measurements[node["node_id"]].append(mean)
        entries.append(group)
    for node in nodes:
        if node["node_id"] in measurements:
            node["exec_time"] = fsum(measurements[node["node_id"]])
        node["arithmetic_intensity"] = intensity(node["flops"], node["exec_time"])

    unmatched_ids = [n["node_id"] for n in nodes if n["exec_time"] is None]
    return {
        "source": str(path.resolve()), "exec_time_unit": "us",
        "arithmetic_intensity_unit": "FLOPs/us", "models": sorted(models),
        "row_count": rows_read, "operator_count": len(entries),
        "matched_node_count": len(nodes) - len(unmatched_ids), "unmatched_node_ids": unmatched_ids,
        "matched_operator_count": sum(e["status"] == "matched" for e in entries),
        "unmatched_operator_count": sum(e["status"] != "matched" for e in entries),
        "sample_counts": sorted({e["sample_count"] for e in entries}), "entries": entries,
        "conventions": [
            "时间单位 us；同一运行算子的多条有效记录取平均，再对匹配到子图的算子均值求和。",
            "保留全部采样（包括首轮）；若不同算子的采样次数不同，仍各自取平均，次数记录在 JSON 中。",
            "按修复前/后的名称匹配，允许去掉末尾 (运行时类型)；/、*、.、_ 扁平化仅接受唯一候选。",
            "可选前缀别名与名称修复共用；最长前缀优先、仅替换一次，歧义不分配时间。",
            "缺失、歧义或无时间的节点不补零；exec_time 完整值留空，known_exec_time 保留已匹配部分。",
            "未匹配的编译器插入/融合算子不分配给邻近节点；名称匹配不能证明融合范围。",
            "exec_time 是算子时间之和，不是考虑并行、调度及传输后的子图端到端延迟。",
            "算术强度按用户定义为 FLOPs/exec_time，单位 FLOPs/us，表示计算吞吐率；不是 FLOPs/Byte。",
            "仅在 FLOPs 已知、全部节点都有时间且 exec_time>0 时计算该比值；请确保 profiling 的输入尺寸与 FLOPs 配置一致。",
        ],
    }


def intensity(flops, exec_time):
    return flops / exec_time if flops is not None and exec_time is not None and exec_time > 0 else None


def time_totals(nodes, flops):
    missing = [n["node_id"] for n in nodes if n["exec_time"] is None]
    known = fsum(n["exec_time"] for n in nodes if n["exec_time"] is not None)
    duration = None if missing else known
    return {"exec_time": duration, "known_exec_time": known,
            "exec_time_matched_nodes": len(nodes) - len(missing),
            "exec_time_missing_node_ids": missing, "exec_time_complete": not missing,
            "arithmetic_intensity": intensity(flops, duration)}
