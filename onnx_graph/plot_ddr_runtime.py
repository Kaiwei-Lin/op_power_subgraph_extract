"""根据 summary CSV/XLSX 绘制各模型 DDR 频率—执行时间图及全部模型对比图。"""

import argparse
from hashlib import sha256
import math
from pathlib import Path
import re
import textwrap

if __package__:
    from .plot_ddr_power import read_table, FREQUENCIES
    from .plot_power_throughput import column
else:
    from plot_ddr_power import read_table, FREQUENCIES
    from plot_power_throughput import column


def prepare_runtime_series(headers, rows, time_unit=None):
    model_col = column(headers, ["OMC相对路径", "OMC名字", "OMC/模型名称（照片辨识）", "OMC文件"])
    frequency_col = column(headers, ["DDR频率_MHz", "DDR频率 MHz"])
    time_col = column(headers, ["运行时长", "执行时间"])
    header_unit = re.search(r"[（(](s|ms|us|µs|μs)[)）]$", time_col)
    groups = {}
    for index, row in enumerate(rows, 2):
        raw_frequency = row[frequency_col]
        if raw_frequency is None or str(raw_frequency).strip() == "":
            continue
        frequency = float(raw_frequency)
        if frequency not in FREQUENCIES:
            raise ValueError(f"第 {index} 行 DDR 频率不支持：{raw_frequency!r}")
        model = str(row[model_col] or "").strip()
        if not model:
            raise ValueError(f"第 {index} 行模型名称为空")
        points = groups.setdefault(model, {})
        if frequency in points:
            raise ValueError(f"模型 {model} 的 {frequency:g} MHz 重复")
        raw = row[time_col]
        if raw is None or str(raw).strip() == "":
            points[frequency] = math.nan
            continue
        declared = row.get("运行时长单位") or row.get("执行时间单位") or (header_unit[1] if header_unit else None)
        declared = str(declared).strip().replace("µ", "u").replace("μ", "u") if declared else None
        if declared and time_unit and declared != time_unit:
            raise ValueError("指定的时间单位与表中单位冲突")
        unit = declared or time_unit
        if unit not in {"s", "ms", "us"}:
            raise ValueError("执行时间单位不明确，请用 --time-unit s/ms/us 指定输入单位")
        duration = float(raw)
        if isinstance(raw, bool) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"第 {index} 行执行时间无效：{raw!r}")
        duration *= {"s": 1000, "ms": 1, "us": .001}[unit]
        if not math.isfinite(duration):
            raise ValueError(f"第 {index} 行执行时间换算后超出范围")
        points[frequency] = duration
    if not any(math.isfinite(t) for points in groups.values() for t in points.values()):
        raise ValueError("没有有效的模型/DDR 频率/执行时间记录")
    return groups


def plot_summary(source, output_dir=None, *, sheet=None, time_unit=None):
    headers, rows = read_table(source, sheet)
    groups = prepare_runtime_series(headers, rows, time_unit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir) if output_dir is not None else Path(source).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def decorate(ax, title):
        ax.set_title(title, pad=12)
        ax.set_xlabel("DDR frequency (MHz)")
        ax.set_ylabel("Execution time (ms)")
        ax.set_xticks(FREQUENCIES)
        ax.tick_params(axis="x", labelrotation=35)
        ax.margins(y=.1)
        ax.grid(alpha=.2, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, stem):
        for extension in ("png", "pdf"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=300)
            files.append(path)
        plt.close(fig)

    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "pdf.fonttype": 42}):
        for model, points in sorted(groups.items()):
            fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
            values = [points.get(frequency, math.nan) for frequency in FREQUENCIES]
            ax.plot(FREQUENCIES, values, color="#2673b8", marker="o", linewidth=1.8, markersize=5)
            title = Path(model.replace("\\", "/")).name
            decorate(ax, "\n".join(textwrap.wrap(title, 75)))
            if not any(math.isfinite(t) for t in values):
                ax.text(.5, .5, "No valid runtime measurements", ha="center", va="center", transform=ax.transAxes)
            stem = re.sub(r"[^\w.-]", "_", title)[:90] + "_" + sha256(model.encode()).hexdigest()[:10]
            save(fig, f"{stem}_ddr_runtime")
        fig, ax = plt.subplots(figsize=(12, max(6, len(groups) * .25)), layout="constrained")
        colors = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20" if len(groups) <= 20 else "turbo")
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*"]
        for index, (model, points) in enumerate(sorted(groups.items())):
            values = [points.get(frequency, math.nan) for frequency in FREQUENCIES]
            color = colors(index if len(groups) <= 20 else index / max(len(groups) - 1, 1))
            ax.plot(FREQUENCIES, values, color=color, marker=markers[index % len(markers)],
                    linewidth=1.6, markersize=4, label=textwrap.fill(model, 55))
        decorate(ax, "Execution time across DDR frequencies")
        ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
        save(fig, "all_models_ddr_runtime")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="默认 summary 所在目录，覆盖同名图")
    parser.add_argument("--sheet", help="XLSX 默认读取第一张工作表")
    parser.add_argument("--time-unit", choices=["s", "ms", "us"], help="输入时间未注明单位时使用；纵轴始终显示 ms")
    args = parser.parse_args()
    files = plot_summary(args.summary, args.output_dir, sheet=args.sheet, time_unit=args.time_unit)
    print(f"已生成 {len(files) // 2 - 1} 个模型的独立图和总图（PNG/PDF）：{files[0].parent.resolve()}")


if __name__ == "__main__":
    main()
