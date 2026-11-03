"""固定 DDR 频率下的跨模型吞吐率/功耗比较；不平滑、不跨频率连线。"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

if __package__:
    from .plot_ddr_power import read_table, power_watts, FREQUENCIES
else:
    from plot_ddr_power import read_table, power_watts, FREQUENCIES

RAILS = ("VBUCK0_PERI", "VBUCK1_1V12", "DDRPHY", "VBUCK8_1V8")


def column(headers, names):
    for name in names:
        if name in headers:
            return name
    candidates = [h for h in headers if re.split(r"[（(]", h)[0].strip() in names]
    if len(candidates) != 1:
        raise ValueError(f"无法确定列 {names}: {candidates}")
    return candidates[0]


def prepare(headers, rows, throughput_unit=None):
    model_col = column(headers, ["OMC相对路径", "OMC名字", "OMC/模型名称（照片辨识）", "OMC文件"])
    freq_col = column(headers, ["DDR频率_MHz", "DDR频率 MHz"])
    throughput_col = column(headers, ["数据移动吞吐", "数据移动吞吐率", "数据移动吞吐率(B/s)"])
    rail_cols = {rail: column(headers, [rail]) for rail in RAILS}
    data, units = {}, set()
    for row in rows:
        if row[freq_col] is None or str(row[freq_col]).strip() == "":
            continue
        model = str(row[model_col] or "").strip()
        freq = float(row[freq_col])
        if not model or freq not in FREQUENCIES:
            raise ValueError(f"模型名或频率无效：{model}, {freq}")
        if (model, freq) in data:
            raise ValueError(f"重复模型/频点：{model}, {freq}")
        declared = "B/s" if "(B/s)" in throughput_col else row.get("数据移动吞吐单位")
        if declared and throughput_unit and declared != throughput_unit:
            raise ValueError("吞吐率参数与表内单位不一致")
        unit = declared or throughput_unit
        if unit not in {"B/s", "GB/s", "MB/s"}:
            raise ValueError("吞吐率单位不明确，请用 --throughput-unit 指定 B/s、MB/s 或 GB/s")
        units.add(unit)
        raw = row[throughput_col]
        x = math.nan if raw is None or str(raw).strip() == "" else float(raw)
        if not math.isnan(x) and (not math.isfinite(x) or x < 0):
            raise ValueError(f"异常吞吐率：{raw}")
        x *= {"B/s": 1e-9, "MB/s": 1e-3, "GB/s": 1}[unit]
        powers = {rail: power_watts(row[c]) for rail, c in rail_cols.items()}
        if any(v < 0 for v in powers.values()):
            raise ValueError(f"负功耗需要核查：{model}, {freq}")
        data[model, freq] = dict(x=x, **powers)
    if not data:
        raise ValueError("没有有效频率记录")
    return data, sorted(units)


def correlations(pairs):
    from scipy.stats import pearsonr, spearmanr
    if len(pairs) < 2:
        return None, None, "insufficient_pairs"
    x, y = zip(*pairs)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, None, "constant_input"
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic), "exploratory"


def analyze(source, output, *, sheet=None, throughput_unit=None):
    headers, rows = read_table(source, sheet)
    data, units = prepare(headers, rows, throughput_unit)
    models = sorted({m for m, _ in data})
    frequencies = sorted({f for _, f in data})
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*"]
    if len(models) > len(markers):
        raise ValueError("模型超过 12 个，请扩展 markers 配置以保持模型标记唯一")
    statistics, correlation_rows = [], []
    for freq in frequencies:
        for rail in RAILS:
            records = [data[m, freq] for m in models if (m, freq) in data]
            powers = [r[rail] for r in records if math.isfinite(r[rail])]
            mean = sum(powers) / len(powers) if powers else None
            statistics.append(dict(DDR_MHz=freq, rail=rail, n_models=len(powers), expected_models=len(models),
                min_power_W=min(powers) if powers else None, max_power_W=max(powers) if powers else None,
                mean_power_W=mean, relative_difference_pct=(max(powers)-min(powers))/mean*100 if mean else None))
            pairs = [(r["x"], r[rail]) for r in records if math.isfinite(r["x"]) and math.isfinite(r[rail])]
            pearson, spearman, status = correlations(pairs)
            correlation_rows.append(dict(DDR_MHz=freq, rail=rail, n_pairs=len(pairs),
                                         pearson_r=pearson, spearman_rho=spearman, status=status))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), layout="constrained")
    norm = Normalize(min(FREQUENCIES), max(FREQUENCIES))
    cmap = plt.get_cmap("viridis")
    for index, (ax, rail) in enumerate(zip(axes.flat, RAILS)):
        for freq in frequencies:
            points = sorted([(data[m, freq]["x"], data[m, freq][rail], m) for m in models
                             if (m, freq) in data and math.isfinite(data[m, freq]["x"])
                             and math.isfinite(data[m, freq][rail])])
            color = cmap(norm(freq))
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=1.15, alpha=.85)
            for x, y, model in points:
                ax.scatter(x, y, color=color, marker=markers[models.index(model)], s=42,
                           edgecolors="white", linewidths=.4, zorder=3)
        ax.set_title(f"({chr(97+index)}) {rail}", loc="left", fontweight="bold")
        ax.set_xlabel("Data Movement Throughput (GB/s)")
        ax.set_ylabel("Average Power (W)")
        ax.grid(alpha=.18, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=.08, y=.12)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.025, pad=.025)
    bar.set_label("DDR frequency (MHz)")
    bar.set_ticks(FREQUENCIES)
    handles = [Line2D([], [], color="0.25", marker=markers[i], linestyle="None", markersize=7, label=m)
               for i, m in enumerate(models)]
    fig.legend(handles=handles, title="Model", loc="outside lower center", ncol=min(3, len(models)), frameon=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/"power_vs_throughput.png", dpi=300)
    fig.savefig(output/"power_vs_throughput.pdf")
    plt.close(fig)
    for filename, records in [("fixed_frequency_statistics.csv", statistics), ("correlation_analysis.csv", correlation_rows)]:
        with (output/filename).open("w",encoding="utf-8-sig",newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    validation = dict(source=str(Path(source).resolve()), models=models, throughput_units=units,
        missing_model_frequencies=[dict(model=m, DDR_MHz=f) for m in models for f in FREQUENCIES if (m,f) not in data],
        missing_values=[dict(model=m, DDR_MHz=f, field=k) for (m,f),r in data.items() for k,v in r.items() if not math.isfinite(v)],
        note="No smoothing or outlier removal. Correlations are exploratory, not significance or causal evidence.")
    (output/"validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    return statistics, correlation_rows, validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--sheet")
    parser.add_argument("--throughput-unit",choices=["B/s","MB/s","GB/s"])
    args=parser.parse_args()
    output=args.output_dir or args.summary.resolve().parent/"power_throughput_analysis"
    analyze(args.summary,output,sheet=args.sheet,throughput_unit=args.throughput_unit)
    print(output.resolve())


if __name__=="__main__":
    main()
