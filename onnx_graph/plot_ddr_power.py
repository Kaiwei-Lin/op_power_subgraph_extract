"""从 summary.csv 或 XLSX 生成每个模型的 DDR 频率/组件功耗折线图。"""

import argparse
import csv
from hashlib import sha256
import math
from pathlib import Path
import re

CHANNELS = ["VBUCK0_PERI", "VBUCK8_1V8", "VBUCK1_1V12", "DDRPHY"]
FREQUENCIES = (418, 749, 1066, 1536, 2102, 2746, 3187, 3744, 4253)


def read_table(path, sheet=None):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            table = list((book[sheet] if sheet else book.worksheets[0]).values)
        finally:
            book.close()
    else:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        import io
        table = list(csv.reader(io.StringIO(text)))
    if not table:
        raise ValueError("输入表为空")
    headers = [str(h).strip() if h is not None else "" for h in table[0]]
    if any(not h for h in headers) or len(set(headers)) != len(headers):
        raise ValueError("表头存在空列或重复列")
    rows = []
    for index, values in enumerate(table[1:], 2):
        if not any(v is not None and str(v).strip() for v in values):
            continue
        if len(values) != len(headers):
            raise ValueError(f"第 {index} 行列数与表头不一致")
        rows.append(dict(zip(headers, values)))
    return headers, rows


def power_watts(value, bare_unit="W"):
    if value is None or str(value).strip() == "":
        return math.nan
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(W|mW|uW|µW|μW)?\s*", str(value))
    if not match or isinstance(value, bool):
        raise ValueError(f"无效功耗值：{value!r}")
    number = float(match[1])
    if not math.isfinite(number):
        raise ValueError(f"非有限功耗值：{value!r}")
    return number * {"W": 1, "mW": 1e-3, "uW": 1e-6, "µW": 1e-6, "μW": 1e-6}[match[2] or bare_unit]


def prepare_series(headers, rows, channels, bare_unit="W"):
    model_col = next((h for h in ("OMC相对路径", "OMC名字", "OMC/模型名称（照片辨识）", "OMC文件") if h in headers), None)
    freq_col = next((h for h in ("DDR频率_MHz", "DDR频率 MHz") if h in headers), None)
    if not model_col or not freq_col:
        raise ValueError("需要模型名称列和 DDR频率_MHz 列；无频率的旧统计表不能绘图")
    columns = {}
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("组件列表不能为空或重复")
    for channel in channels:
        matches = [h for h in headers if h == channel]
        if not matches:
            matches = [h for h in headers if re.split(r"[（(]", h, maxsplit=1)[0].strip() == channel]
        if len(matches) != 1:
            raise ValueError(f"组件 {channel!r} 找不到唯一列，候选：{matches}")
        columns[channel] = matches[0]
    groups = {}
    for row in rows:
        if row[freq_col] is None or str(row[freq_col]).strip() == "":
            continue  # 混合总表中的无频率测量不参与曲线。
        freq = float(row[freq_col])
        if freq not in FREQUENCIES:
            raise ValueError(f"不支持的 DDR 频率：{freq}")
        model = str(row[model_col] or "").strip()
        if not model:
            raise ValueError("模型名称为空")
        points = groups.setdefault(model, {})
        if freq in points:
            raise ValueError(f"模型 {model} 的 {freq} MHz 重复，不能自动合并测量")
        points[freq] = [power_watts(row[columns[c]], bare_unit) for c in channels]
    if not groups:
        raise ValueError("没有可绘制的 DDR 频点记录")
    return groups


def power_ranges(groups, channel_count):
    """每个频点按组件计算模型间极差；不足两个有效模型时保持未知。"""
    result = {}
    for frequency in FREQUENCIES:
        differences = []
        for i in range(channel_count):
            values = [points[frequency][i] for points in groups.values()
                      if frequency in points and math.isfinite(points[frequency][i])]
            differences.append(max(values) - min(values) if len(values) >= 2 else math.nan)
        result[frequency] = differences
    return result


def plot_summary(source, output_dir=None, channels=None, *, sheet=None, bare_unit="W",
                 y_min=None, y_max=None, y_step=None):
    channels = CHANNELS if channels is None else channels
    headers, rows = read_table(source, sheet)
    groups = prepare_series(headers, rows, channels, bare_unit)
    # 输入解析统一为 W；仅在绘图边界转换成 mW，源数据保持不变。
    groups = {model: {freq: [value * 1000 for value in values] for freq, values in points.items()}
              for model, points in groups.items()}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    values = [v for points in groups.values() for point in points.values() for v in point if math.isfinite(v)]
    lower = min(0, min(values, default=0)) if y_min is None else y_min
    upper = max(max(values, default=0) * 1.05, lower + 0.1) if y_max is None else y_max
    if not all(math.isfinite(v) for v in (lower, upper)) or lower >= upper:
        raise ValueError("Y 轴上下限必须有限且下限小于上限")
    if y_step is not None:
        if not math.isfinite(y_step) or y_step <= 0 or (upper - lower) / y_step > 1000:
            raise ValueError("Y 轴刻度间隔必须为有限正数，且刻度数不能超过 1000")
        ticks = [lower + i * y_step for i in range(math.floor((upper - lower) / y_step + 1e-9) + 1)]
    else:
        ticks = [t for t in MaxNLocator(nbins=6).tick_values(lower, upper) if lower <= t <= upper]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    output = Path(output_dir) if output_dir is not None else Path(source).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    files = []
    def draw(ax, model, points):
        for i, channel in enumerate(channels):
            values = [points.get(f, [math.nan] * len(channels))[i] for f in FREQUENCIES]
            ax.plot(FREQUENCIES, values, marker="o", linewidth=2, markersize=5, label=channel)
        title = Path(model.replace("\\", "/")).name
        import textwrap
        ax.set_title("\n".join(textwrap.wrap(title, 70)), fontsize=12, pad=14)
        ax.set_xlabel("DDR frequency (MHz)")
        ax.set_ylabel("Power (mW)")
        ax.set_xticks(FREQUENCIES)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(lower, upper)
        ax.set_yticks(ticks)

    def save(fig, stem):
        for ext in ("png", "svg"):
            path = output / f"{stem}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            files.append(path)
        plt.close(fig)

    for model, points in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        draw(ax, model, points)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=min(4, len(channels)), frameon=False)
        title = Path(model.replace("\\", "/")).name
        stem = re.sub(r"[^\w.-]", "_", title)[:100] + "_" + sha256(model.encode()).hexdigest()[:10]
        save(fig, stem)
    columns = min(3, len(groups))
    rows_count = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(7 * columns, 4.8 * rows_count),
                             squeeze=False, sharex=True, sharey=True, layout="constrained")
    for ax, (model, points) in zip(axes.flat, sorted(groups.items())):
        draw(ax, model, points)
        ax.tick_params(labelleft=True, labelbottom=True)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("DDR power comparison — shared Y axis", fontsize=16)
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(channels)), frameon=False)
    save(fig, "all_models_ddr_power")
    differences = power_ranges(groups, len(channels))
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    draw(ax, "Power range across models (max - min)", differences)
    ax.set_ylabel("Power difference (mW)")
    finite = [v for point in differences.values() for v in point if math.isfinite(v)]
    # 极差采用独立纵轴，避免绝对功耗的统一范围掩盖差异。
    ax.set_yticks([])
    ax.set_ylim(0, max(max(finite, default=0) * 1.05, 0.01))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=min(4, len(channels)), frameon=False)
    save(fig, "all_models_power_range")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="默认 summary 所在目录；覆盖同名图，不删除其他文件")
    parser.add_argument("--channels", nargs="+", default=CHANNELS, help="要绘制的组件列名，可增删")
    parser.add_argument("--sheet", help="XLSX 工作表名称，默认第一张")
    parser.add_argument("--power-unit", choices=("W", "mW", "uW"), default="W", help="无单位功耗的单位，默认 W")
    parser.add_argument("--y-min", type=float, help="统一 Y 轴下限，单位 mW")
    parser.add_argument("--y-max", type=float, help="统一 Y 轴上限，单位 mW")
    parser.add_argument("--y-step", type=float, help="统一 Y 轴刻度间隔，单位 mW")
    args = parser.parse_args()
    files = plot_summary(args.summary, args.output_dir, args.channels, sheet=args.sheet, bare_unit=args.power_unit,
                         y_min=args.y_min, y_max=args.y_max, y_step=args.y_step)
    print(f"已生成模型独立图、总览图和功耗极差图（PNG/SVG）：{files[0].parent.resolve()}")


if __name__ == "__main__":
    main()
