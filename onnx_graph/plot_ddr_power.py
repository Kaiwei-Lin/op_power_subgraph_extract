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


def plot_summary(source, output_dir, channels=None, *, sheet=None, bare_unit="W"):
    channels = CHANNELS if channels is None else channels
    headers, rows = read_table(source, sheet)
    groups = prepare_series(headers, rows, channels, bare_unit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for model, points in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        for i, channel in enumerate(channels):
            values = [points.get(f, [math.nan] * len(channels))[i] for f in FREQUENCIES]
            ax.plot(FREQUENCIES, values, marker="o", linewidth=2, markersize=5, label=channel)
        title = Path(model.replace("\\", "/")).name
        import textwrap
        ax.set_title("\n".join(textwrap.wrap(title, 70)), fontsize=12, pad=14)
        ax.set_xlabel("DDR frequency (MHz)")
        ax.set_ylabel("Power (W)")
        ax.set_xticks(FREQUENCIES)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=min(4, len(channels)), frameon=False)
        stem = re.sub(r"[^\w.-]", "_", title)[:100] + "_" + sha256(model.encode()).hexdigest()[:10]
        for ext in ("png", "svg"):
            path = output / f"{stem}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            files.append(path)
        plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="必须为新目录")
    parser.add_argument("--channels", nargs="+", default=CHANNELS, help="要绘制的组件列名，可增删")
    parser.add_argument("--sheet", help="XLSX 工作表名称，默认第一张")
    parser.add_argument("--power-unit", choices=("W", "mW", "uW"), default="W", help="无单位功耗的单位，默认 W")
    args = parser.parse_args()
    files = plot_summary(args.summary, args.output_dir, args.channels, sheet=args.sheet, bare_unit=args.power_unit)
    print(f"已生成 {len(files) // 2} 个模型的 PNG/SVG 图：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
