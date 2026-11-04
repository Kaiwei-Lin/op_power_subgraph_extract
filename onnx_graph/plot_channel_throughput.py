"""将 summary 中各 Channel 的平均功耗与数据移动吞吐率画在同一张图上。"""

import argparse
import math
from pathlib import Path
import re

if __package__:
    from .plot_ddr_power import read_table, power_watts
    from .plot_power_throughput import column
else:
    from plot_ddr_power import read_table, power_watts
    from plot_power_throughput import column


CHANNELS = ["VBUCK0_PERI", "VBUCK1_1V12", "DDRPHY"]


def channel_points(headers, rows, channels=None, *, ddr_frequency=None, throughput_unit=None,
                   bare_power_unit="W"):
    throughput_col = column(headers, ["数据移动吞吐", "数据移动吞吐率", "数据移动吞吐率(B/s)"])
    frequency_col = next((h for h in ("DDR频率_MHz", "DDR频率 MHz") if h in headers), None)
    if ddr_frequency is not None:
        if not math.isfinite(ddr_frequency) or ddr_frequency <= 0:
            raise ValueError("DDR 频率必须是有限正数（MHz）")
        if frequency_col is None:
            raise ValueError("此表没有 DDR 频率列，无法按频率筛选")
    if channels is None:
        channels = CHANNELS
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Channel 列表不能为空或重复，请用 --channels 指定")
    columns = {c: column(headers, [c]) for c in channels}
    labels = {c: re.split(r"\[W\]|[（(]", c, maxsplit=1)[0].strip() for c in channels}
    if any(not c for c in labels.values()) or len(set(labels.values())) != len(labels):
        raise ValueError("去掉单位和括号说明后的 Channel 名称为空或重复")
    points = {labels[c]: [] for c in channels}
    for index, row in enumerate(rows, 2):
        if ddr_frequency is not None:
            raw_frequency = row[frequency_col]
            if raw_frequency is None or str(raw_frequency).strip() == "":
                continue
            if float(raw_frequency) != ddr_frequency:
                continue
        raw_x = row[throughput_col]
        if raw_x is None or str(raw_x).strip() == "":
            continue
        unit = "B/s" if "(B/s)" in throughput_col else row.get("数据移动吞吐单位")
        if unit and throughput_unit and unit != throughput_unit:
            raise ValueError("指定的吞吐率单位与表中单位冲突")
        unit = unit or throughput_unit
        if unit not in {"B/s", "MB/s", "GB/s"}:
            raise ValueError("吞吐率单位不明确，请指定 --throughput-unit B/s、MB/s 或 GB/s")
        x = float(raw_x)
        if isinstance(raw_x, bool) or not math.isfinite(x) or x < 0:
            raise ValueError(f"第 {index} 行吞吐率无效：{raw_x!r}")
        x *= {"B/s": 1e-9, "MB/s": 1e-3, "GB/s": 1}[unit]
        for channel, field in columns.items():
            y = power_watts(row[field], bare_power_unit)
            if math.isfinite(y):
                points[labels[channel]].append((x, y))
    if not any(points.values()):
        raise ValueError("筛选后没有有效的吞吐率/功耗配对；请检查时长、功耗状态或筛选条件")
    return points


def plot_summary(source, output_dir=None, channels=None, *, sheet=None, ddr_frequency=None,
                 throughput_unit=None, bare_power_unit="W", display_power_unit="mW"):
    headers, rows = read_table(source, sheet)
    points = channel_points(headers, rows, channels, ddr_frequency=ddr_frequency,
                            throughput_unit=throughput_unit, bare_power_unit=bare_power_unit)
    if display_power_unit not in {"W", "mW"}:
        raise ValueError("功耗显示单位仅支持 W 或 mW")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        colors = plt.get_cmap("tab10" if len(points) <= 10 else "tab20" if len(points) <= 20 else "turbo")
        scale = 1000 if display_power_unit == "mW" else 1
        for i, (channel, values) in enumerate(points.items()):
            values = sorted(values, key=lambda p: p[0])
            color = colors(i if len(points) <= 20 else i / max(len(points) - 1, 1))
            ax.plot([p[0] for p in values], [p[1] * scale for p in values],
                    label=channel, color=color, marker="o", markersize=4, linewidth=1.5, alpha=.85)
        title = "Mean channel power vs. data movement throughput"
        if ddr_frequency is not None:
            title += f"\nDDR frequency: {ddr_frequency:g} MHz"
        ax.set_title(title, pad=12)
        ax.set_xlabel("Data Movement Throughput (GB/s)")
        ax.set_ylabel(f"Average Power ({display_power_unit})")
        ax.grid(alpha=.2, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Channel", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        output = Path(output_dir) if output_dir is not None else Path(source).resolve().parent
        output.mkdir(parents=True, exist_ok=True)
        suffix = f"_ddr_{ddr_frequency:g}MHz" if ddr_frequency is not None else ""
        files = [output / f"channels_power_vs_throughput{suffix}.{ext}" for ext in ("png", "pdf")]
        for path in files:
            fig.savefig(path, dpi=300)
        plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="默认 summary 所在目录，覆盖同名图")
    parser.add_argument("--channels", nargs="+", default=CHANNELS,
                        help="默认 VBUCK0_PERI VBUCK1_1V12 DDRPHY；可增删")
    parser.add_argument("--sheet", help="XLSX 默认读取第一张工作表")
    parser.add_argument("--ddr-frequency", type=float, help="只画指定 DDR 频率（MHz）")
    parser.add_argument("--throughput-unit", choices=["B/s", "MB/s", "GB/s"], help="表中未注明时的输入吞吐率单位")
    parser.add_argument("--power-unit", choices=["W", "mW", "uW"], default="W", help="无单位功耗值的输入单位")
    parser.add_argument("--display-power-unit", choices=["W", "mW"], default="mW")
    args = parser.parse_args()
    files = plot_summary(args.summary, args.output_dir, args.channels, sheet=args.sheet,
                         ddr_frequency=args.ddr_frequency, throughput_unit=args.throughput_unit,
                         bare_power_unit=args.power_unit, display_power_unit=args.display_power_unit)
    for path in files:
        print(path.resolve())


if __name__ == "__main__":
    main()
