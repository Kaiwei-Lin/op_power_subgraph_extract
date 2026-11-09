"""导出经核验的六路 perfstat 测量，并为 merge_power 提供按 SHA256/DDR 频率的关联。"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys


SCOPE = "npu_aicore_pair_inferred"
SELECTED_CHANNELS = ("aicore_m0", "aicore_m1")
NUMERIC_FIELDS = (
    "实测DDR频率_MHz", "实测采样时长_s", "实测采样点数", "实测窗口推理次数",
    "实测读带宽_GiBps", "实测写带宽_GiBps", "实测总带宽_GiBps",
    "实测采样读字节数", "实测采样写字节数", "实测采样总字节数",
    "实测每次推理读字节数", "实测每次推理写字节数", "实测每次推理总字节数",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return result


def _number(value, name, *, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} 必须为{'整数' if integer else '数值'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} 必须为有限{'正数' if positive else '非负数'}")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} 必须为小写 SHA256")


def _evidence(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 需要非空证据说明")


def validate_record(record):
    """Portable summary contract. Raw capture verification happens during export."""
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("scope") != SCOPE:
        raise ValueError("不支持的 bandwidth.json 版本或监测范围；当前要求 aicore_m0 + aicore_m1，"
                         "请从原始采样重新导出，不能将旧 IBQE 结果直接改名")
    for name in ("model_sha256", "data_sha256"):
        _hash(record.get(name), name)
    frequency = record.get("ddr_frequency_mhz")
    if frequency is not None:
        _number(frequency, "ddr_frequency_mhz", positive=True, integer=True)
        _evidence(record.get("ddr_frequency_evidence"), "ddr_frequency_evidence")
    seconds = _number(record.get("capture_seconds"), "capture_seconds", positive=True)
    samples = _number(record.get("sample_count"), "sample_count", positive=True, integer=True)
    ticks = _number(record.get("period_ticks"), "period_ticks", positive=True, integer=True)
    clock = _number(record.get("sampling_clock_hz"), "sampling_clock_hz", positive=True)
    elapsed_ticks = _number(samples * ticks, "elapsed_ticks", positive=True, integer=True)
    if samples < 2 or not math.isclose(seconds, elapsed_ticks / clock, rel_tol=1e-9):
        raise ValueError("采样时长与采样点数、ticks、采样时钟不一致")
    for direction in ("read", "write"):
        _number(record.get(direction + "_bytes"), direction + "_bytes", integer=True)
        if not math.isfinite(record[direction + "_bytes"] / seconds / 2**30):
            raise ValueError("带宽超出可表示范围")
    total = _number(record["read_bytes"] + record["write_bytes"], "total_bytes", integer=True)
    if not math.isfinite(total / seconds / 2**30):
        raise ValueError("合计带宽超出可表示范围")
    valid = record.get("selected_valid_samples", {})
    if not isinstance(valid, dict):
        raise ValueError("selected_valid_samples 必须为对象")
    for name in SELECTED_CHANNELS:
        count = _number(valid.get(name), name + " valid samples", integer=True)
        if not 2 <= count <= samples:
            raise ValueError(f"{name} 有效样本不足或超过总数，不能标记有效零带宽")
    validation = record.get("validation")
    if not isinstance(validation, dict) or any(validation.get(k) != "PASS" for k in ("isolation", "overlap")):
        raise ValueError("需要隔离及采样与推理重叠验证均为 PASS")
    calls = record.get("inferences_in_capture")
    method = record.get("inference_count_method")
    if calls is not None:
        if method not in ("exact", "estimated"):
            raise ValueError("窗口推理次数需要 exact/estimated 口径")
        _number(calls, "inferences_in_capture", positive=True, integer=(method == "exact"))
        _evidence(record.get("inference_count_evidence"), "inference_count_evidence")
        if not math.isfinite(total / calls):
            raise ValueError("每次推理数据量超出可表示范围")
    elif method is not None:
        raise ValueError("有推理次数口径但缺少窗口内推理次数")


def export_capture(capture_dir, model_path=None, *, ddr_mhz=None, ddr_evidence=None):
    """Recompute bytes from raw data; never import screenshot-reproduction totals."""
    if __package__:
        from .perfstat.analyze import analyze
    else:
        from perfstat.analyze import analyze
    capture = Path(capture_dir).resolve()
    invocation = read_json(capture / "invocation.json")
    if invocation.get("kind") != "model":
        raise ValueError("仅导出模型运行采样，空闲对照不能作为模型带宽")
    model = Path(model_path or invocation["model"]).resolve()
    model_hash = sha256(model)
    if model_hash != invocation.get("model_sha256"):
        raise ValueError("模型 SHA256 与 invocation.json 不一致")
    sources = ["invocation.json", "kernel_perfstat.log", "capture.log", "isolation_validation.json"]
    isolation = read_json(capture / "isolation_validation.json")
    overlap_files = [p for p in ("overlap_validation.json", "window_progress_validation.json")
                     if (capture / p).is_file()]
    if not overlap_files:
        raise ValueError("缺少采样与模型运行重叠验证文件")
    overlap = [read_json(capture / p) for p in overlap_files]
    if isolation.get("status") != "PASS" or any(p.get("status") != "PASS" for p in overlap):
        raise ValueError("隔离或采样与模型运行重叠验证未通过")
    sources.extend(overlap_files)
    sources.extend(p for p in ("run.log", "input_manifest.json", "npu_state_before.txt", "npu_state_at_start.txt",
                              "npu_state_at_stop.txt", "npu_state_after.txt") if (capture / p).is_file())
    capture_log = (capture / "capture.log").read_text(encoding="utf-8-sig")
    returns = re.findall(r"\bret\s*=\s*0x([0-9a-fA-F]+)", capture_log)
    if len(returns) != 2 or any(int(x, 16) for x in returns):
        raise ValueError("capture.log 需要本次 start/stop 的两个 ret = 0x0")
    kernel = (capture / "kernel_perfstat.log").read_text(encoding="utf-8-sig")
    clocks = set(re.findall(r"\bclk_perf_stat is (\d+)", kernel))
    if len(clocks) != 1:
        raise ValueError("内核日志需要本次采样的唯一 clk_perf_stat 时钟")
    result = analyze((capture / "perf_data_hs.data").read_bytes(), int(clocks.pop()))
    previous = capture / "counter_bandwidth.json"
    if previous.is_file():
        if read_json(previous).get("sha256") != result["sha256"]:
            raise ValueError("已有 counter_bandwidth.json 与原始 .data 的 SHA256 不一致")
        sources.append(previous.name)
    channels = {c["name"]: c for c in result["channels"]}
    record = dict(
        schema_version=1, model_sha256=model_hash, model_path=str(model),
        ddr_frequency_mhz=ddr_mhz, ddr_frequency_evidence=ddr_evidence,
        scope=SCOPE, capture_seconds=result["duration_seconds"], sample_count=result["sample_count"],
        period_ticks=result["period_ticks"], sampling_clock_hz=result["sampling_clock_hz"],
        data_sha256=result["sha256"],
        read_bytes=sum(channels[n]["read_bytes"] for n in SELECTED_CHANNELS),
        write_bytes=sum(channels[n]["write_bytes"] for n in SELECTED_CHANNELS),
        selected_valid_samples={n: result["sample_count"] - channels[n]["invalid_samples"]
                                for n in SELECTED_CHANNELS},
        inferences_in_capture=None, inference_count_method=None, inference_count_evidence=None,
        validation={"isolation": "PASS", "overlap": "PASS"},
        conditions={k: invocation.get(k) for k in ("target", "profiling", "pmu", "dev_perf", "input_mode")},
        capture_directory=str(capture), channels=result["channels"],
        conservation=result["conservation"], limitations=result["limitations"],
    )
    if (capture / "inference_window.json").is_file():
        sources.append("inference_window.json")
        inference = read_json(capture / "inference_window.json")
        for key in ("inferences_in_capture", "inference_count_method", "inference_count_evidence"):
            record[key] = inference.get(key)
    record["evidence_files"] = {name: sha256(capture / name) for name in sources}
    validate_record(record)
    return record


def load_bandwidth_results(directory):
    index = defaultdict(list)
    if directory is None:
        return index
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(root.rglob("bandwidth.json"))
    if not paths:
        raise ValueError(f"实测目录内未找到 bandwidth.json：{root}")
    for path in paths:
        try:
            record = read_json(path)
            validate_record(record)
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
        key = (record["model_sha256"], record.get("ddr_frequency_mhz"))
        index[key].append((path.resolve(), record))
    return index


def bandwidth_fields(index, model_hash, frequency):
    row = dict.fromkeys(NUMERIC_FIELDS)
    row.update(实测带宽状态="missing", 实测带宽原因="没有 SHA256 和 DDR 频率均匹配的测量",
               实测监测范围="", 实测带宽文件="", 实测数据SHA256="", 实测每次量口径="")
    matches = index.get((model_hash, frequency), [])
    if not matches:
        return row
    row["实测带宽文件"] = "; ".join(str(p) for p, _ in matches)
    if len(matches) != 1:
        row.update(实测带宽状态="ambiguous", 实测带宽原因="同一模型和 DDR 频率有多个测量，请明确选择")
        return row
    _, record = matches[0]
    seconds, calls = record["capture_seconds"], record.get("inferences_in_capture")
    row.update(实测带宽状态="ok", 实测带宽原因="",
               实测监测范围="NPU aicore_m0 + aicore_m1（接口计数口径）",
               实测数据SHA256=record["data_sha256"], 实测DDR频率_MHz=frequency,
               实测采样时长_s=seconds, 实测采样点数=record["sample_count"],
               实测窗口推理次数=calls, 实测每次量口径=record.get("inference_count_method") or "unknown")
    for name, amount in (("读", record["read_bytes"]), ("写", record["write_bytes"]),
                         ("总", record["read_bytes"] + record["write_bytes"])):
        row[f"实测{name}带宽_GiBps"] = amount / seconds / 2**30
        row[f"实测采样{name}字节数"] = amount
        row[f"实测每次推理{name}字节数"] = amount / calls if calls is not None else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="OMC 路径，默认读取 invocation.json")
    parser.add_argument("--ddr-mhz", type=int, help="采样时已确认的 DDR MHz；未知时省略")
    parser.add_argument("--ddr-evidence", help="固定 DDR 频率及采样期间验证的证据")
    parser.add_argument("--output", type=Path, help="默认 capture-dir/bandwidth.json；拒绝覆盖")
    args = parser.parse_args()
    output = args.output or args.capture_dir / "bandwidth.json"
    if output.name != "bandwidth.json":
        parser.error("输出文件必须命名为 bandwidth.json，供 --bandwidth-dir 递归收集")
    try:
        result = export_capture(args.capture_dir, args.model, ddr_mhz=args.ddr_mhz,
                                ddr_evidence=args.ddr_evidence)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"导出失败：{error}\n")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
