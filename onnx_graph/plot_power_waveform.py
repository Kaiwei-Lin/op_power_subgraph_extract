"""读取详细 db XLSX 的第一张工作表，绘制指定 Channel 的原始功耗波形。"""

import argparse
import math
from pathlib import Path
import re

if __package__:
    from .merge_power import _sample_number, load_workbook
else:
    from merge_power import _sample_number, load_workbook


def read_waveform(source, channel):
    """时间戳输入为 us；返回从首个采样开始的 us 和功耗 W，空功耗保留为 NaN。"""
    if load_workbook is None:
        raise ValueError("读取 XLSX 需要：python -m pip install openpyxl")
    source = Path(source)
    channel = channel.strip()
    if not channel:
        raise ValueError("Channel 名称不能为空")
    with source.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        try:
            sheet = book.worksheets[0]
            sheet.reset_dimensions()
            rows = sheet.iter_rows(values_only=True)
            header = next((r for r in rows if any(v is not None and str(v).strip() for v in r)), ())
            columns = [(i, str(h).split("[W]", 1)[0].strip(), str(h).strip())
                       for i, h in enumerate(header) if i > 0 and "[W]" in str(h)]
            matches = [c for c in columns if c[2] == channel]
            if not matches:
                matches = [c for c in columns if c[1] == channel]
            if len(matches) != 1:
                raise ValueError(f"找不到唯一的功耗列 {channel!r}；可选 Channel：{[c[1] for c in columns]}")
            column, label, _ = matches[0]
            times, powers = [], []
            first = previous = None
            for index, row in enumerate(rows, 2):
                if not any(v is not None and str(v).strip() for v in row):
                    continue
                timestamp = _sample_number(row[0], f"第 {index} 行时间戳 (us)")
                if previous is not None and timestamp <= previous:
                    raise ValueError(f"第 {index} 行时间戳重复或倒序，需要严格递增")
                first = timestamp if first is None else first
                elapsed = float(timestamp - first)
                if not math.isfinite(elapsed):
                    raise ValueError(f"第 {index} 行时间戳跨度超出范围")
                previous = timestamp
                raw_power = row[column] if column < len(row) else None
                power = math.nan
                if raw_power is not None and str(raw_power).strip():
                    power = float(_sample_number(raw_power, f"第 {index} 行 {label}[W]"))
                    if not math.isfinite(power):
                        raise ValueError(f"第 {index} 行功耗超出范围")
                times.append(elapsed)
                powers.append(power)
            if not any(math.isfinite(p) for p in powers):
                raise ValueError(f"{channel} 没有有效功耗采样值")
            return times, powers
        finally:
            book.close()


def plot_waveform(source, channel, output_dir=None, *, time_unit="ms", power_unit="mW"):
    if time_unit not in {"us", "ms", "s"} or power_unit not in {"W", "mW"}:
        raise ValueError("不支持的时间或功耗显示单位")
    times, powers = read_waveform(source, channel)
    times = [t / {"us": 1, "ms": 1000, "s": 1_000_000}[time_unit] for t in times]
    powers = [p * (1000 if power_unit == "mW" else 1) for p in powers]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = Path(source)
    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        ax.plot(times, powers, linewidth=1, color="#2673b8", label=channel,
                marker="o" if sum(math.isfinite(p) for p in powers) == 1 else None)
        ax.set_title(f"{source.name}\n{channel} power waveform", pad=12)
        ax.set_xlabel(f"Elapsed time ({time_unit})")
        ax.set_ylabel(f"Power ({power_unit})")
        ax.grid(alpha=.2, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        output = Path(output_dir) if output_dir is not None else source.resolve().parent
        output.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.-]", "_", source.stem)[:80]
        name = re.sub(r"[^\w.-]", "_", channel)[:80]
        files = [output / f"{stem}_{name}_waveform.{ext}" for ext in ("png", "pdf")]
        for path in files:
            fig.savefig(path, dpi=300)
        plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="详细功耗 XLSX，例如 1_db.xlsx、1-1db.xlsx")
    parser.add_argument("--channel", required=True, help="例如 DDRPHY，或完整列名 DDRPHY[W]")
    parser.add_argument("--output-dir", type=Path, help="默认输入文件所在目录，覆盖同名图")
    parser.add_argument("--time-unit", choices=["us", "ms", "s"], default="ms", help="从首个采样开始的时间显示单位")
    parser.add_argument("--power-unit", choices=["W", "mW"], default="mW", help="显示单位；原始 [W] 输入始终按 W 解析")
    args = parser.parse_args()
    files = plot_waveform(args.db, args.channel, args.output_dir, time_unit=args.time_unit, power_unit=args.power_unit)
    for path in files:
        print(path.resolve())


if __name__ == "__main__":
    main()
