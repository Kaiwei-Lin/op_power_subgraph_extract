"""批量统计已编译的子图 OMC；每个文件按整个运算图统计，不要求内部保留层级。"""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

if __package__:
    from .merge_power import TRANSFER_FIELDS, transfer_statistics
    from .stats import analyze_subgraphs, write_reports
else:
    from merge_power import TRANSFER_FIELDS, transfer_statistics
    from stats import analyze_subgraphs, write_reports


PACKAGE_DIR = Path(__file__).resolve().parent
OMC_DIR = PACKAGE_DIR / "models/handfree_ainr/omc"
SUMMARY_CSV = None  # 可选：代表子图 representatives_summary.csv；不提供则统计目录下全部 OMC/OM。
OUTPUT_DIR = None  # 默认 OMC_DIR / compiled_stats；必须是新目录。
GE_PROTO_PATH = None  # DDK ge_ir_pb2.py，也可设置 ONNX_GRAPH_GE_PROTO。

_FIELDS = ["OMC文件", "OMC状态", "OMC原因", "OMC统计JSON", "OMC统计TXT", "OMC_SHA256",
           "OMC_节点数", "OMC_MACs", "OMC_FLOPs", "OMC_已知MACs", "OMC_已知FLOPs",
           "OMC_计算量完整", "OMC_未知计算量节点数", *["OMC_" + f for f in TRANSFER_FIELDS]]


def _key(filename):
    stem = Path(str(filename).replace("\\", "/")).stem
    stem = re.sub(r"_fuse_\d{8}_\d{6}$", "", stem)
    return re.sub(r"_fixed$", "", stem)


def _slash(path, root):
    parts = [root.name, *path.relative_to(root).parts[:-1]]
    levels = {str(int(m[1])) for part in parts if (m := re.fullmatch(r"slash_?(\d+)", part, re.I))}
    return next(iter(levels)) if len(levels) == 1 else ""


def number_models(omc_dir, numbering_csv=None):
    """按 slash 数字及文件名（忽略大小写）编号；可用既有编号表固定对应关系。"""
    root = Path(omc_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".omc", ".om"}]
    paths.sort(key=lambda p: (int(_slash(p, root)) if _slash(p, root) else float("inf"),
                             p.name.casefold(), p.relative_to(root).as_posix()))
    if not paths:
        raise FileNotFoundError(f"{root}: 没有 OMC/OM 文件")
    records = [dict(编号=i, 斜杠数=_slash(p, root), OMC相对路径=p.relative_to(root).as_posix(),
                    OMC_SHA256=sha256(p.read_bytes()).hexdigest()) for i, p in enumerate(paths, 1)]
    if numbering_csv is None:
        return records
    with Path(numbering_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if (not {"编号", "斜杠数", "OMC相对路径", "OMC_SHA256"}.issubset(fields)
                or len(set(fields)) != len(fields) or any(not f.strip() for f in fields)):
            raise ValueError("编号表需要 编号、斜杠数、OMC相对路径、OMC_SHA256 列")
        saved = list(reader)
    inventory = {r["OMC相对路径"]: r for r in records}
    result, seen = [], set()
    for row in saved:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("编号表字段数量与表头不一致")
        relative = row["OMC相对路径"].replace("\\", "/")
        if relative in seen or relative not in inventory:
            raise ValueError(f"编号表文件重复或不在当前 OMC 目录：{relative}")
        expected = inventory[relative]
        if row["OMC_SHA256"] != expected["OMC_SHA256"] or row["斜杠数"] != expected["斜杠数"]:
            raise ValueError(f"OMC 内容或层级与编号表不一致：{relative}")
        seen.add(relative)
        result.append({**expected, "编号": int(row["编号"])})
    if seen != set(inventory) or sorted(r["编号"] for r in result) != list(range(1, len(records) + 1)):
        raise ValueError("编号表必须完整覆盖当前 OMC，且编号连续、不重复，从 1 开始")
    return sorted(result, key=lambda r: r["编号"])


def _jobs(root, files, summary_csv):
    if summary_csv is None:
        return ["斜杠数", "子图名字"], [(dict(斜杠数=_slash(p, root), 子图名字=_key(p.name)), [p]) for p in files], []
    with Path(summary_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if (not fields or any(not f.strip() for f in fields) or len(set(fields)) != len(fields)
                or not {"ONNX文件", "子图名字"}.issubset(fields)):
            raise ValueError("代表总表需要唯一、非空的表头，且包含 ONNX文件、子图名字")
        if (set(fields) & set(_FIELDS)) - {"OMC文件"}:
            raise ValueError("请使用原始 representatives_summary.csv；总表已包含 OMC 统计列")
        rows = list(reader)
    index = defaultdict(list)
    for path in files:
        index[_key(path.name)].append(path)
    jobs, used = [], set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("代表总表存在字段数量与表头不一致的行")
        if not row["ONNX文件"].strip() or not row["子图名字"].strip():
            raise ValueError("代表总表的 ONNX文件、子图名字不能为空")
        explicit = row.get("OMC文件", "").strip()
        if explicit:
            target = Path(explicit.replace("\\", "/"))
            target = (root / target).resolve() if not target.is_absolute() else target.resolve()
            candidates = [p for p in files if p.resolve() == target]
        else:
            candidates = index[_key(row["ONNX文件"])]
            level = row.get("斜杠数", "").strip()
            if level:
                candidates = [p for p in candidates if not _slash(p, root) or _slash(p, root) == level]
        used.update(candidates)
        jobs.append((row, candidates))
    return fields, jobs, [str(p) for p in files if p not in used]


def analyze_compiled_subgraphs(omc_dir, output_dir, *, summary_csv=None, ge_proto_path=None):
    """以 OMC 整图 total 汇总；保留输入总表的 ONNX 统计和时间列，不覆盖其来源。"""
    root, output = Path(omc_dir).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".omc", ".om"})
    if not files:
        raise FileNotFoundError(f"{root}: 没有 OMC/OM 文件")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{output}")
    fields, jobs, unused = _jobs(root, files, summary_csv)
    counts = Counter(paths[0] for _, paths in jobs if len(paths) == 1)
    output.mkdir(parents=True)
    rows, operators = [], set()
    for original, candidates in jobs:
        row = dict(original)
        row.update({f: "" for f in _FIELDS})
        row["OMC文件"] = json.dumps([str(p) for p in candidates], ensure_ascii=False)
        if not candidates:
            row.update(OMC状态="missing", OMC原因="找不到对应 OMC；可在原总表中添加 OMC文件 列指定相对路径")
        elif len(candidates) != 1 or counts[candidates[0]] > 1:
            row.update(OMC状态="ambiguous", OMC原因="存在多个候选或同一 OMC 对应多行；请检查文件及映射")
        else:
            path = candidates[0]
            row["OMC文件"] = str(path)
            try:
                report = analyze_subgraphs(path, [0], repair_names=False, hierarchy_source="outputs",
                                           ge_proto_path=ge_proto_path)
                # 一个文件就是一个已编译子图；内部名称层级不再参与分组。
                report["levels"] = []
                _, transfer = transfer_statistics(report)
                report["compiled_scope"] = "whole_omc"
                report["data_transfer"] = transfer
                report["conventions"] = [*report["conventions"],
                    "本报告统计整个子图 OMC；不按编译后名称再切分，也不合计其他 OMC。",
                    "数据转移量为逐算子输入（含权重/偏置）及输出的逻辑字节数之和；共享及中间张量按使用计入，非实测 DDR 流量。"]
                relative = str(path.relative_to(root))
                stem = re.sub(r"[^\w.-]", "_", path.stem)[:70]
                detail = output / "models" / f"{stem}_{sha256(relative.encode('utf-8')).hexdigest()[:12]}"
                write_reports(report, detail)
                with (detail / "stats.txt").open("a", encoding="utf-8") as stream:
                    stream.write("\n数据转移量（逻辑字节）:\n" + "\n".join(f"{k}: {v}" for k, v in transfer.items()) + "\n")
                total = report["total"]
                row.update(OMC状态="ok", OMC统计JSON=str(detail / "stats.json"), OMC统计TXT=str(detail / "stats.txt"),
                           OMC_SHA256=sha256(path.read_bytes()).hexdigest(), OMC_节点数=total["node_count"],
                           OMC_MACs=total["macs"], OMC_FLOPs=total["flops"], OMC_已知MACs=total["known_macs"],
                           OMC_已知FLOPs=total["known_flops"], OMC_计算量完整="是" if total["complete"] else "否",
                           OMC_未知计算量节点数=len(total["unknown_node_ids"]))
                row.update({"OMC_" + k: v for k, v in transfer.items()})
                row["operator_counts"] = total["operator_counts"]
                operators.update(total["operator_counts"])
            except Exception as error:
                row.update(OMC状态="failed", OMC原因=f"{type(error).__name__}: {error}")
        rows.append(row)
    operator_fields = [f"OMC_{op}数量" for op in sorted(operators)]
    headers = list(dict.fromkeys([*fields, *_FIELDS, *operator_fields]))
    for row in rows:
        for op in sorted(operators):
            row[f"OMC_{op}数量"] = row.get("operator_counts", {}).get(op, 0) if row["OMC状态"] == "ok" else ""
    result = dict(omc_dir=str(root), summary_csv=str(Path(summary_csv).resolve()) if summary_csv is not None else None,
                  source_format="OMC", scope="whole_omc_per_row", counts=dict(Counter(r["OMC状态"] for r in rows)),
                  unused_omc_files=unused, records=rows)
    (output / "omc_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "omc_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omc-dir", type=Path, default=OMC_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV, help="可选 representatives_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ge-proto", type=Path, default=GE_PROTO_PATH)
    args = parser.parse_args(argv)
    output = args.output_dir if args.output_dir is not None else args.omc_dir / "compiled_stats"
    try:
        result = analyze_compiled_subgraphs(args.omc_dir, output, summary_csv=args.summary, ge_proto_path=args.ge_proto)
    except (OSError, ValueError) as error:
        parser.exit(1, f"统计失败：{error}\n")
    print(f"OMC statistics: {result['counts']}")
    print(f"未关联的 OMC: {len(result['unused_omc_files'])}")
    print(f"总表: {Path(output).resolve() / 'omc_summary.csv'}")
    return result


if __name__ == "__main__":
    main()
