"""修改顶部配置，一次完成统计、筛选、导出和去重，或按单个 PREFIX 导出。"""

from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import operator
from pathlib import Path
import pickle
import re

import onnx

# 同时支持直接运行 main.py 和 python -m onnx_graph.main。
if __package__:
    from .core import extract_prefix, networkx_to_onnx, nodes_by_prefix, onnx_to_networkx
    from .similarity import group_subgraphs
    from .stats import analyze_subgraphs, write_reports
else:
    from core import extract_prefix, networkx_to_onnx, nodes_by_prefix, onnx_to_networkx
    from similarity import group_subgraphs
    from stats import analyze_subgraphs, write_reports


PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_DIR / "models/handfree_ainr/handfree_ainr.onnx"
PREFIX = "MobilenetV3/expanded_conv_1"
REPAIR_NAMES = True  # 自动统计和导出共用；空名称始终跳过名称处理。
OUTPUT_DIR = MODEL_PATH.parent / "networkx/main_example"

# True: 统计、筛选、批量导出；False: 使用原来的 PREFIX 示例。
BATCH_FROM_STATS = True
GENERATE_STATS = True  # False: 跳过统计，读取 STATS_DIR 中已有的全部 stats_<数字>_slash.csv。
STATS_DIR = MODEL_PATH.parent / "networkx" / (
    MODEL_PATH.stem + ("_omc_stats" if MODEL_PATH.suffix.lower() in (".omc", ".om") else "_stats"))
SLASH_COUNTS = [1, 2]  # 本次统计及导出的层级；前导 / 不计数。
PROFILE_CSV = MODEL_PATH.parent / "handfree_ainr_fuse_20261020_102316_op.csv"
# PROFILE_CSV = None  # 不需要时间统计时设为 None。
BATCH_SIZE = 1  # 仅用于统计估算：将输入的未知第 0 维视为 batch；None 禁用。
INPUT_SHAPES = {}  # 仅用于统计估算，例如 {"input": [1, 3, 224, 224]}，不改写导出 ONNX。
GE_PROTO_PATH = None  # OMC 统计可选：DDK 的 ge_ir_pb2.py；ONNX 不需要。

# 默认不过滤。每个内层列表是 AND，各个内层列表之间是 OR；数值单位沿用 CSV。
FILTERS = []
# 例：exec_time > 100 us
# FILTERS = [[("exec_time", ">", 100)]]
# 例：exec_time > 100 us 且 节点数 > 5
# FILTERS = [[("exec_time", ">", 100), ("节点数", ">", 5)]]
# 例：exec_time > 100 us 或 节点数 > 5
# FILTERS = [[("exec_time", ">", 100)], [("节点数", ">", 5)]]
# 例：仅按已匹配部分的耗时筛选，允许子图的完整 exec_time 未知
# FILTERS = [[("已知exec_time", ">", 100)]]

# 批量导出后只保留本次分组的代表 ONNX；False 则保留全部导出模型。
KEEP_REPRESENTATIVES_ONLY = True
GROUP_MODE = "exact"  # exact: 配置同构；similar: 按下面的配置相似度阈值分组。
SIMILARITY_THRESHOLD = 0.90
SEPARATE_SLASH_COUNTS = True

_COMPARISONS = {"==": operator.eq, "!=": operator.ne, ">": operator.gt,
                ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def _number(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"不是有效数值: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"不是有限数值: {value!r}")
    return number


def _prepare_filters(filters):
    result = []
    for group in filters:
        if not group:
            raise ValueError("条件组不能为空；如需不过滤，请设置 FILTERS = []")
        prepared = []
        for condition in group:
            if not isinstance(condition, (tuple, list)) or len(condition) != 3:
                raise ValueError(f"条件应为 (列名, 比较符, 阈值): {condition!r}")
            column, comparison, expected = condition
            if not isinstance(column, str) or not column.strip() or comparison not in _COMPARISONS:
                raise ValueError(f"无效的列名或比较符: {condition!r}")
            if isinstance(expected, (int, float, Decimal)) and not isinstance(expected, bool):
                expected = _number(expected)
            elif not isinstance(expected, str) or comparison not in ("==", "!="):
                raise ValueError(f"数值阈值请使用数字；文本只支持 == 和 !=: {condition!r}")
            prepared.append((column.strip(), comparison, expected))
        result.append(prepared)
    return result


def _matches_filters(row, groups):
    def matches(condition):
        column, comparison, expected = condition
        actual = row[column]
        if isinstance(expected, Decimal):
            if not actual:
                return False  # 未知时间/计算量不是 0，也不能通过 != 0 等数值条件。
            actual = _number(actual)
        return _COMPARISONS[comparison](actual, expected)

    return not groups or any(all(matches(c) for c in group) for group in groups)


def _write_batch_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report["counts"] = dict(Counter(r["status"] for r in report["records"]))
    summary_path = output_dir / "representatives_summary.csv"
    report["representatives_summary"] = str(summary_path.resolve())
    (output_dir / "batch_export.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (output_dir / "batch_export.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["统计文件", "CSV行号", "斜杠数", "子图前缀", "状态", "节点数",
                         "ONNX文件", "输入张量", "输出张量", "原因",
                         "去重状态", "分组", "代表ONNX", "与代表相似度", "去重说明"])
        for r in report["records"]:
            writer.writerow([r["csv_file"], r["csv_row"], r["slash_count"], r["prefix"],
                             r["status"], r["node_count"], r["onnx_file"],
                             json.dumps(r["inputs"], ensure_ascii=False),
                             json.dumps(r["outputs"], ensure_ascii=False), r["reason"],
                             r.get("dedup_status", ""), r.get("group", ""),
                             r.get("representative_onnx", ""), r.get("similarity", ""),
                             r.get("dedup_reason", "")])
    group_sizes = Counter(r["group"] for r in report["records"] if r.get("group"))
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["斜杠数", "分组", "子图名字", "ONNX文件", "节点数", "同组子图数",
                         "已知exec_time", "exec_time", "exec_time_unit",
                         "时间匹配节点数", "时间未匹配节点数"])
        for r in report["records"]:
            if r.get("dedup_status") != "representative":
                continue
            values = r["statistics"]
            writer.writerow([r["slash_count"], r["group"], r["prefix"], r["onnx_file"], r["node_count"],
                             group_sizes[r["group"]], values.get("已知exec_time", ""),
                             values.get("exec_time", ""), values.get("exec_time_unit", ""),
                             values.get("时间匹配节点数", ""), values.get("时间未匹配节点数", "")])


def _retain_representatives(report, output_dir, *, mode, threshold, separate_slash_counts):
    root = output_dir.resolve()
    candidates = {Path(r["onnx_file"]).resolve(): r for r in report["records"]
                  if r["status"] in ("exported", "exists")}
    if Path(report["model"]).resolve() in candidates:
        raise ValueError("原始模型不能作为待清理的子图，未删除任何模型")
    if any(not p.is_relative_to(root) or p.suffix.lower() != ".onnx" or not p.is_file()
           for p in candidates):
        raise ValueError("去重文件必须位于本次批量输出目录内，未删除任何模型")
    snapshots = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in candidates}
    grouped = group_subgraphs(root, mode=mode, threshold=threshold,
                              separate_slash_counts=separate_slash_counts, model_paths=candidates)
    members = {Path(m["path"]).resolve(): (level["level"], m)
               for level in grouped["levels"] for m in level["members"]}
    failures = {Path(r["path"]).resolve(): r["reason"] for r in grouped["failures"]}
    representatives = {p for p, (_, m) in members.items() if m["is_representative"]}
    if set(members) & set(failures) or (set(members) | set(failures)) != set(candidates):
        raise ValueError("分组清单与本次导出文件不一致，未删除任何模型")
    if any(Path(m["representative_path"]).resolve() not in representatives
           for _, m in members.values()):
        raise ValueError("分组清单缺少代表模型，未删除任何模型")
    # 先核对所有文件，再逐文件删除；不递归清理目录或相信 CSV 中的任意路径。
    if any(p.resolve() != p or not p.is_file()
           or (p.stat().st_size, p.stat().st_mtime_ns) != snapshots[p] for p in candidates):
        raise ValueError("分组期间模型文件发生变化，未删除任何模型")
    for path, record in candidates.items():
        if path in failures:
            record.update(dedup_status="kept_unchecked", dedup_reason=failures[path])
            continue
        level, member = members[path]
        record.update(group=f"{level}/{member['group']}",
                      representative_onnx=member["representative_path"],
                      similarity=member["similarity"], dedup_reason="；".join(member["notes"]))
        if path in representatives:
            record["dedup_status"] = "representative"
            continue
        try:
            path.unlink()
        except OSError as error:
            record.update(dedup_status="cleanup_failed", dedup_reason=f"未删除，仍保留模型: {error}")
            continue
        record.update(export_status=record["status"], status="deduplicated", dedup_status="removed",
                      reason=f"已去重；代表 ONNX: {member['representative_path']}")
    return dict(status="completed", mode=mode, threshold=threshold,
                separate_slash_counts=separate_slash_counts,
                grouping_report=str(Path(grouped["output_dir"]) / "grouping.json"),
                counts=dict(Counter(r["dedup_status"] for r in candidates.values())))


def export_from_stats(model_path, stats_dir, output_dir, *, filters=None, repair_names=False,
                      deduplicate=False, group_mode="exact", similarity_threshold=0.90,
                      separate_slash_counts=True, csv_paths=None):
    """读取单个模型的分层 CSV；组内 AND、组间 OR；返回逐行导出清单。"""
    model_path, stats_dir, output_dir = map(Path, (model_path, stats_dir, output_dir))
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("批量导出需要原始 .onnx，不能将 OMC 编译图直接还原为 ONNX")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    groups = _prepare_filters(filters or [])
    paths = stats_dir.glob("stats_*_slash.csv") if csv_paths is None else [Path(p) for p in csv_paths]
    if csv_paths is not None and any(
            not p.is_file() or p.resolve().parent != stats_dir.resolve()
            or not re.fullmatch(r"stats_(\d+)_slash\.csv", p.name) for p in paths):
        raise ValueError("csv_paths 必须是统计目录内已有的 stats_<数字>_slash.csv")
    files = sorted(
        (int(match[1]), path)
        for path in paths
        if path.is_file() and (match := re.fullmatch(r"stats_(\d+)_slash\.csv", path.name))
    )
    if not files:
        raise FileNotFoundError(f"{stats_dir} 中没有 stats_<数字>_slash.csv")

    required = {"子图名字", "节点数"} | {c[0] for group in groups for c in group}
    records = []
    for depth, path in files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            fields = [f.strip() for f in next(reader, [])]
            if not fields or "" in fields or len(set(fields)) != len(fields):
                raise ValueError(f"{path}: CSV 表头为空或有重复/空字段")
            missing = required - set(fields)
            if missing:
                raise ValueError(f"{path}: 缺少列 {sorted(missing)}；请检查 FILTERS 或重新生成统计")
            for values in reader:
                if not values or not any(v.strip() for v in values):
                    continue
                if len(values) != len(fields):
                    raise ValueError(f"{path}:{reader.line_num}: CSV 字段数量与表头不一致")
                row = dict(zip(fields, (v.strip() for v in values)))
                prefix = "/".join(p for p in row["子图名字"].split("/") if p)
                record = dict(csv_file=str(path.resolve()), csv_row=reader.line_num,
                              slash_count=depth, prefix=prefix, statistics=row,
                              status="selected", reason="", onnx_file="", node_count=None,
                              inputs=[], outputs=[])
                try:
                    if not prefix:
                        raise ValueError("子图前缀为空，不能将空前缀当作整图导出")
                    if prefix.count("/") != depth:
                        raise ValueError("子图前缀的 / 数量与统计文件名不一致")
                    if not _matches_filters(row, groups):
                        values = {c[0]: row[c[0]] or "<空>" for group in groups for c in group}
                        record.update(status="filtered", reason=(
                            f"未满足筛选条件 {filters!r}；实际值: "
                            + json.dumps(values, ensure_ascii=False)))
                except ValueError as error:
                    record.update(status="failed", reason=str(error))
                records.append(record)

    graph = onnx_to_networkx(model_path, repair_names=repair_names) if any(
        r["status"] == "selected" for r in records) else None
    seen = set()
    for record in records:
        if record["status"] != "selected":
            continue
        prefix, depth = record["prefix"], record["slash_count"]
        created = False
        try:
            row = record["statistics"]
            ids = nodes_by_prefix(graph, prefix)
            if not ids:
                raise ValueError("ONNX 中找不到该前缀；请检查模型及 REPAIR_NAMES 是否与统计时一致")
            record["node_count"] = len(ids)
            if _number(row["节点数"]) != len(ids):
                raise ValueError(f"CSV 节点数 {row['节点数']} 与 ONNX 的 {len(ids)} 不一致；请重新生成同一模型的统计")
            counts = Counter((f"{graph.nodes[i]['domain']}::{graph.nodes[i]['op_type']}"
                              if graph.nodes[i]["domain"] else graph.nodes[i]["op_type"]) for i in ids)
            for column, value in row.items():
                if column.endswith("数量") and _number(value) != counts[column[:-2]]:
                    raise ValueError(f"{column} 与 ONNX 不一致；不能把 OMC/其他模型的统计直接用于当前 ONNX")
            key = (depth, prefix)
            if key in seen:
                record.update(status="duplicate", reason="相同层级、前缀已在本批次处理")
                continue
            seen.add(key)
            # 不将原始前缀直接当作路径；摘要区分 /、_、特殊字符和大小写。
            stem = re.sub(r"[^\w.-]", "_", prefix)[:60].strip(".") or "subgraph"
            digest = sha256(prefix.encode("utf-8")).hexdigest()[:12]
            target = output_dir / f"slash_{depth}" / f"sg_{stem}_{digest}.onnx"
            record["onnx_file"] = str(target.resolve())
            if target.exists():
                record.update(status="exists", reason="已有文件保留；需重新导出时请使用新的输出目录")
                continue
            sub_model = networkx_to_onnx(extract_prefix(graph, prefix))
            payload = sub_model.SerializeToString()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                created = True
                stream.write(payload)
            record.update(status="exported", inputs=[v.name for v in sub_model.graph.input],
                          outputs=[v.name for v in sub_model.graph.output])
        except Exception as error:
            # 单个子图失败仍继续处理其他行；仅清理本次未写完的新文件。
            if created:
                target.unlink(missing_ok=True)
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")

    report = dict(model=str(model_path.resolve()), stats_dir=str(stats_dir.resolve()),
                  csv_files=[str(path.resolve()) for _, path in files],
                  repair_names=repair_names, filters=filters or [],
                  counts=dict(Counter(r["status"] for r in records)), records=records)
    _write_batch_report(report, output_dir)
    if deduplicate:
        try:
            report["deduplication"] = _retain_representatives(
                report, output_dir, mode=group_mode, threshold=similarity_threshold,
                separate_slash_counts=separate_slash_counts)
        except Exception as error:
            report["deduplication"] = dict(status="failed", reason=f"{type(error).__name__}: {error}")
        finally:
            _write_batch_report(report, output_dir)
    print(f"Batch export: {len(files)} CSV files, {len(records)} rows; {report['counts']}")
    print(f"ONNX output directory: {output_dir.resolve()}")
    if not records:
        print("CSV 没有子图数据行，只有表头或空行；请检查统计目录和层级。")
    elif not report["counts"].get("exported") and not report["counts"].get("deduplicated"):
        print("本次未生成新的 ONNX 文件；下方显示部分原因，完整记录见导出清单。")
    for status in ("failed", "filtered", "exists"):
        examples = [r for r in records if r["status"] == status][:3]
        for r in examples:
            print(f"[{status}] {r['prefix']}: {r['reason'][:1000]}")
    if deduplicate:
        result = report["deduplication"]
        print(f"Deduplication: {result.get('counts', result.get('reason'))}")
        if result["status"] == "completed":
            print(f"Grouping report: {result['grouping_report']}")
        print(f"Representative summary: {report['representatives_summary']}")
    print(f"Export manifest: {output_dir / 'batch_export.csv'}")
    return report


def main():
    if BATCH_FROM_STATS:
        csv_paths = None
        if GENERATE_STATS:
            print(f"Generate statistics: {MODEL_PATH}")
            statistics = analyze_subgraphs(
                MODEL_PATH, SLASH_COUNTS, repair_names=REPAIR_NAMES, batch_size=BATCH_SIZE,
                input_shapes=INPUT_SHAPES, profile_csv=PROFILE_CSV, ge_proto_path=GE_PROTO_PATH)
            csv_paths = write_reports(statistics, STATS_DIR)
            print(f"Statistics report: {Path(STATS_DIR).resolve() / 'stats.json'}")
            for level, path in zip(statistics["levels"], csv_paths):
                print(f"slash_{level['slash_count']}: {level['subgraph_count']} subgraphs; {path.resolve()}")
            if statistics["source_format"] == "OMC":
                print("OMC 统计已完成；OMC 编译图不能直接导出为可执行 ONNX，后续导出与去重未运行。")
                return statistics
        return export_from_stats(MODEL_PATH, STATS_DIR, OUTPUT_DIR / "batch",
                                 filters=FILTERS, repair_names=REPAIR_NAMES,
                                 deduplicate=KEEP_REPRESENTATIVES_ONLY, group_mode=GROUP_MODE,
                                 similarity_threshold=SIMILARITY_THRESHOLD,
                                 separate_slash_counts=SEPARATE_SLASH_COUNTS, csv_paths=csv_paths)
    graph = onnx_to_networkx(MODEL_PATH, repair_names=REPAIR_NAMES)
    print(f"Full graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # 导出时会校验模型；缺失的 graph.name 会补齐，不能要求与原始无效模型逐字节相同。
    restored = networkx_to_onnx(graph)
    print(f"ONNX graph name: {restored.graph.name}")

    node_ids = nodes_by_prefix(graph, PREFIX)
    print(f"Prefix: {PREFIX} ({len(node_ids)} nodes)")
    for node_id in node_ids:
        data = graph.nodes[node_id]
        print(f"  [{node_id}] {data['op_type']}: {data['name']}")
        print(f"    inputs:  {data['inputs']}")
        print(f"    outputs: {data['outputs']}")

    subgraph = extract_prefix(graph, PREFIX)
    # 在这里添加自己的分析或编辑逻辑；修改节点时使用 subgraph.nodes[id]["proto"]。
    sub_model = networkx_to_onnx(subgraph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = OUTPUT_DIR / "full_graph.nx.pkl"
    with graph_path.open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = OUTPUT_DIR / "subgraph.onnx"
    onnx.save(sub_model, model_path)

    if REPAIR_NAMES:
        report_path = OUTPUT_DIR / "rename_report.json"
        report_path.write_text(
            json.dumps(graph.graph["rename_report"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Rename report: {report_path}")

    print(f"NetworkX file: {graph_path}")
    print(f"Subgraph ONNX: {model_path}")
    print(f"Subgraph inputs:  {[v.name for v in sub_model.graph.input]}")
    print(f"Subgraph outputs: {[v.name for v in sub_model.graph.output]}")


if __name__ == "__main__":
    main()
