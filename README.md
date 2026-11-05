# ONNX / OMC 子图分析与功耗工具

本仓库用于将 ONNX 转为 NetworkX、按层级提取子图、统计计算量、筛选和去重，再对已编译的 OMC 合并实测功耗并绘图。

完整流程：

```text
原始 ONNX
  → 按输出路径建立层级
  → 统计子图 / 匹配 profiling 时间
  → 按条件筛选、导出子图 ONNX
  → 按连接、形状和属性去重，保留代表子图
  → 在外部工具中编译为 OMC 并采集功耗
  → OMC 静态统计 + 编号功耗文件 → summary.csv
  → DDR 曲线 / 吞吐率与功耗关系 / 单 Channel 波形
```

已有 OMC 时，可以直接从 `merge_power.py` 开始，不需要原始 ONNX。OMC 支持解析和统计；生成可执行子图 ONNX 需要原始 ONNX。仓库不负责调用编译器或执行设备功耗采集。

## 目录与 Python 文件总览

以下路径以克隆后的仓库根目录为基准。当前有 **17 个功能 Python 文件和 19 个测试 Python 文件**；每个文件的说明见本节及后续使用示例。

```text
op_power_subgraph_extract/
├── README.md
└── onnx_graph/
    ├── __init__.py
    ├── __main__.py
    ├── main.py
    ├── core.py
    ├── stats.py
    ├── profiling.py
    ├── similarity.py
    ├── omc.py
    ├── ge_ir_pb2.py
    ├── compiled_stats.py
    ├── merge_power.py
    ├── plot_ddr_power.py
    ├── plot_ddr_runtime.py
    ├── plot_channel_throughput.py
    ├── plot_power_throughput.py
    ├── plot_power_waveform.py
    ├── verify_mobilenet.py
    ├── requirements.txt
    ├── name_aliases.example.json
    ├── tests/                     # 逐文件说明见“测试文件”
    └── models/                    # 本机模型和 profiling 数据，不上传
```

| Python 文件 | 用途 | 使用方式 / 主要输入输出 |
|---|---|---|
| `onnx_graph/main.py` | 统一执行统计、筛选、批量导出、去重和代表子图汇总 | 修改文件顶部配置后运行；输入 ONNX，输出分层统计、子图 ONNX、导出清单、代表总表 |
| `onnx_graph/core.py` | ONNX ↔ NetworkX 的核心转换、名称修复、前缀提取与边界处理 | 供其他脚本导入；输入 ONNX/图，输出 `nx.MultiDiGraph` 或 `ModelProto` |
| `onnx_graph/stats.py` | 按 slash 层级统计 ONNX/OMC 的节点、算子类型、MACs/FLOPs，可附加 profiling 时间 | CLI 或 Python API；输出 `stats.json`、`stats.txt`、每个层级一份 CSV |
| `onnx_graph/profiling.py` | 将算子运行时间 CSV 映射到图节点，并按运行算子取均值 | 由 `stats.py` 调用；提供时间匹配、已知时间、完整时间及自定义“算术强度” |
| `onnx_graph/similarity.py` | 比较子图的连接、端口、形状、类型与属性，进行精确或相似分组 | 输入 ONNX 文件夹；输出分组、代表清单、两两相似度和失败清单 |
| `onnx_graph/omc.py` | 解析 OMC/OM 的 IMOD 分区、GE 图、节点和张量描述符 | 供转换与统计调用；输出 GE NetworkX 图，估算支持的 GE 算子计算量 |
| `onnx_graph/ge_ir_pb2.py` | 随仓库提供的 DDK 6.1.1.0 GE protobuf 定义 | 解析器只提取 descriptor 字节；不需要直接运行或手动修改此生成文件 |
| `onnx_graph/compiled_stats.py` | 对目录内每个已编译 OMC 整图统计，可关联代表子图总表 | 输出 `omc_summary.csv/.json` 及各 OMC 的明细；同时提供通用编号函数 |
| `onnx_graph/merge_power.py` | OMC 编号与统计，合并旧平均功耗表或新 db 采样 XLSX，支持 9 个 DDR 频点 | 输出 `numbering.csv`、`summary.csv/.json` 和 `omc_stats/` |
| `onnx_graph/plot_ddr_power.py` | 画每个模型的 DDR 频率—功耗折线图、所有模型总览和跨模型功耗极差图 | 输入 summary CSV/XLSX；输出 PNG/SVG，默认功耗单位 mW |
| `onnx_graph/plot_ddr_runtime.py` | 画同一模型不同 DDR 频率的执行时间折线图，每个模型一张 | 输入 summary CSV/XLSX；输出 PNG/PDF，纵轴 ms，同次运行统一 Y 轴范围 |
| `onnx_graph/plot_channel_throughput.py` | 在同一坐标系比较 DDR 相关或指定 Channel 的平均功耗与吞吐率 | 输入 summary；输出一张折线图的 PNG/PDF，默认 GB/s 和 mW |
| `onnx_graph/plot_power_throughput.py` | 论文用固定 DDR 频率的跨模型吞吐率—功耗四宫格及统计分析 | 输出 300 DPI PNG、矢量 PDF、功耗差异 CSV、逐频率相关性 CSV 和校验 JSON；功耗用 W |
| `onnx_graph/plot_power_waveform.py` | 读取指定 db XLSX，画单个 Channel 的原始功耗随时间的波形 | 输入详细采样文件和 Channel；输出 PNG/PDF，默认时间 ms、功耗 mW |
| `onnx_graph/verify_mobilenet.py` | 使用指定 MobileNetV3 样例验证完整往返和子图数值结果 | 需要本机样例模型与 `onnxruntime`；输出验证用 ONNX、NetworkX 文件和报告 |
| `onnx_graph/__main__.py` | 包的命令行入口 | `python -m onnx_graph`，提供 `convert`、`restore`、`extract`、`repair-names` |
| `onnx_graph/__init__.py` | 包的公开 API 导出 | 支持 `from onnx_graph import onnx_to_networkx, extract_prefix, ...`，不是业务运行入口 |

其他文件：`requirements.txt` 为依赖清单；`name_aliases.example.json` 为旧 name 模式的别名映射示例；`.gitignore` 排除模型、原始功耗、profiling、输出报告和缓存。

## 安装与命令执行位置

使用 **Python 3.10 或以上**；本项目此前以 Python 3.10 为主要验证环境。

```powershell
git clone https://github.com/Kaiwei-Lin/op_power_subgraph_extract.git
cd op_power_subgraph_extract
python -m pip install -r onnx_graph/requirements.txt

# 读取功耗/summary XLSX 或画 db 波形时需要
python -m pip install openpyxl

# 仅运行 ONNX Runtime 数值验证及相关测试时需要
python -m pip install onnxruntime
```

`requirements.txt` 包含 ONNX、NetworkX、protobuf、Matplotlib、SciPy。`openpyxl` 和 `onnxruntime` 按上述用途额外安装。模型、设备工具、原始测量文件不包含在仓库中。

本文的 `python -m onnx_graph...` 命令均从**克隆后的仓库根目录**运行。在原工作区 `D:/WHU/huawei_power_opt` 使用源码时，先进入 `tools` 目录再运行相同命令。仅 `main.py` 路径配置相对自身 `__file__` 计算；CLI 传入的相对路径按当前工作目录解释。

例如本机数据可放为：

```text
onnx_graph/models/my_model/
├── model.onnx
├── model_op.csv
├── batch_omc/
│   ├── slash_0/a.omc
│   └── slash_1/b.omc
└── power/
    ├── 1_db.xlsx
    ├── 1.csv
    └── 2_db.xlsx
```

## 按任务选择入口

| 当前任务 | 入口 |
|---|---|
| 从原始 ONNX 一次完成统计、筛选、导出、去重 | `main.py` |
| 只想按层级统计，不导出子图 | `stats.py` |
| 只转换 NetworkX 或提取一个前缀 | `python -m onnx_graph`，或 `core.py` API |
| 对已导出的 ONNX 分组、计算相似度 | `similarity.py` |
| 已有 OMC，只做静态统计 | `compiled_stats.py`，或不传 `--power-dir` 的 `merge_power.py` |
| 已有 OMC 和编号功耗文件，生成总表 | `merge_power.py --batch-dir ... --power-dir ...` |
| 看各模型在不同 DDR 频率下的功耗 | `plot_ddr_power.py` |
| 看各模型在不同 DDR 频率下的执行时间 | `plot_ddr_runtime.py` |
| 看多个 Channel 的功耗与吞吐率分布 | `plot_channel_throughput.py` |
| 固定 DDR 频率比较模型，并计算相关性 | `plot_power_throughput.py` |
| 看一份 db 文件中某个 Channel 的瞬时功耗 | `plot_power_waveform.py` |

## main.py：ONNX 统计、筛选、导出与去重

### 配置与运行

`main.py` 是配置式入口，**没有命令行参数解析器**。先修改顶部配置，再运行：

```powershell
python -m onnx_graph.main
# 或
python onnx_graph/main.py
```

以下为一份通用配置示例，不会自动修改源文件：

```python
MODEL_PATH = PACKAGE_DIR / "models/my_model/model.onnx"
OUTPUT_DIR = MODEL_PATH.parent / "networkx" / MODEL_PATH.stem
STATS_DIR = MODEL_PATH.parent / "networkx" / (MODEL_PATH.stem + "_stats")

BATCH_FROM_STATS = True
GENERATE_STATS = True
SLASH_COUNTS = [0, 1, 2, 3]
HIERARCHY_SOURCE = "outputs"
PROFILE_CSV = None  # 有算子 profiling 时改为对应 CSV 路径
BATCH_SIZE = 1
INPUT_SHAPES = {}
FILTERS = []        # 不筛选

KEEP_REPRESENTATIVES_ONLY = True
GROUP_MODE = "exact"
SIMILARITY_THRESHOLD = 0.90
SEPARATE_SLASH_COUNTS = True
```

| 配置项 | 含义 |
|---|---|
| `MODEL_PATH` | 原始 ONNX；若是 OMC 且启用统计，则统计后停止，不导出 ONNX |
| `OUTPUT_DIR` | 批量结果写入其 `batch/` 子目录 |
| `STATS_DIR` | 分层 CSV 和完整统计 JSON 的目录 |
| `BATCH_FROM_STATS` | `True` 走完整批处理；`False` 按 `PREFIX` 提取单个子图示例 |
| `GENERATE_STATS` | `True` 先生成统计，仅消费本次生成的 CSV；`False` 复用已有 `stats_<数字>_slash.csv` |
| `SLASH_COUNTS` | 要生成的层级，例如 `[0, 1, 2, 3]` |
| `PROFILE_CSV` | 算子运行时间来源；`None` 禁用时间匹配 |
| `BATCH_SIZE` / `INPUT_SHAPES` | 统计计算量时的形状假设，不直接改写用于提取的原始模型 |
| `FILTERS` | 对统计列的过滤，支持 AND/OR 联合条件 |
| `KEEP_REPRESENTATIVES_ONLY` | 对本次选中的导出文件分组，并删除其中的非代表 ONNX；比较失败的文件保留 |
| `GROUP_MODE` | `exact` 按配置同构分组；`similar` 按相似度阈值分组 |
| `SEPARATE_SLASH_COUNTS` | 是否按 slash 层级分别分组 |
| `REPAIR_NAMES` / `NAME_ALIASES_JSON` | 仅 `HIERARCHY_SOURCE="name"` 时参与名称修复 |
| `GE_PROTO_PATH` | 可选 OMC schema 覆盖路径，通常使用内置文件即可 |

当前源码中的 `MODEL_PATH` 仍为 `models/mobilenetv3-large_224_1.0.onnx`，`PROFILE_CSV` 为同目录的 `_20261014_151905_op.csv`，实际生效的 `FILTERS` 为 `[[('已知exec_time', '>', 15)]]`。这不是“默认不过滤”的配置；第一次用于其他模型时应修改路径和筛选条件。同一个配置变量多次赋值时，最后一次赋值生效。

### 常用筛选条件

```python
FILTERS = []  # 全部保留
FILTERS = [("已知exec_time", ">", 15)]
FILTERS = [("节点数", ">", 10)]

# AND：时间 > 15 且节点数 > 10
FILTERS = [("已知exec_time", ">", 15), ("节点数", ">", 10)]

# OR：时间 > 15 或节点数 > 10
FILTERS = [[("已知exec_time", ">", 15)], [("节点数", ">", 10)]]
```

支持 `>`、`>=`、`<`、`<=`、`==`、`!=`。数值阈值使用数字；文本只用 `==`、`!=`。未知值留空，不能当作 0 通过数值条件。`exec_time` 阈值单位为 **us**。

### 输出与诊断

```text
STATS_DIR/
├── stats.json
├── stats.txt
├── stats_0_slash.csv
├── stats_1_slash.csv
└── ...

OUTPUT_DIR/batch/
├── slash_0/*.onnx
├── slash_1/*.onnx
├── batch_export.csv
├── batch_export.json
├── representatives_summary.csv
├── name_output_mismatches.json
├── output_hierarchy_issues.json
└── grouping_*/...                  # 开启去重时的分组明细
```

`representatives_summary.csv` 记录代表子图名称、路径、节点数、同组数量、每种算子数量、已知/完整 `exec_time` 和匹配覆盖情况。代表行由去重分组产生，未开启去重时不应把此文件当作所有导出项清单；完整导出记录看 `batch_export.csv/.json`。

只有清单没有 ONNX 时，查看清单的“状态”和“原因”：`filtered` 表示未通过筛选，`failed` 表示前缀不匹配或导出失败，`deduplicated` 表示已删除非代表文件。统计时与导出时必须使用同一原始模型、层级来源及名称配置。

## core.py、__init__.py 与 __main__.py：图转换和提取

### Python API

```python
from pathlib import Path
import onnx
from onnx_graph import (
    onnx_to_networkx, nodes_by_prefix, extract_prefix, networkx_to_onnx,
)

model_path = Path("onnx_graph/models/my_model/model.onnx")
graph = onnx_to_networkx(model_path, hierarchy_source="outputs")
node_ids = nodes_by_prefix(graph, "encoder/block_1")
subgraph = extract_prefix(graph, "encoder/block_1")
submodel = networkx_to_onnx(subgraph)
onnx.save(submodel, "subgraph.onnx")
```

`__init__.py` 还导出 `refresh_graph`、`repair_node_names`、`load_name_aliases`、`check_name_outputs`、`omc_to_networkx`。这些是程序调用接口，不是独立统计脚本。

`core.py` 将 ONNX 算子表示为 `networkx.MultiDiGraph` 节点，ID 初始为整数，避免空名称或重名覆盖。主要数据：

| 位置 | 内容 |
|---|---|
| `graph.nodes[id]['proto']` | 完整 ONNX `NodeProto`，包含属性、子图属性和元数据 |
| 节点 `name`、`op_type`、`inputs`、`outputs` 等 | 从 protobuf 派生的便捷字段 |
| `original_name`、`original_index` | 源模型名称与原始序号 |
| `hierarchy`、`hierarchy_path`、`scope` | 当前分层结果 |
| `graph.graph['model']` | 源 `ModelProto`，包括权重、输入输出、opset、函数、类型等信息 |
| 图的诊断字段 | 名称修复、name/output 不匹配、输出分层问题 |
| 边 | 张量依赖及 `tensor`、`output_index`、`input_index`、`implicit`，保留多输入端口和多重边 |

编辑 ONNX 语义时修改 `proto`，再调用 `refresh_graph(graph)`；只修改便捷名称或边属性不会直接改变导出的 ONNX。`extract_prefix` 返回独立拷贝；NetworkX 自带视图可能共享 protobuf 对象。

提取只选择指定前缀范围内的算子，前缀匹配有完整路径边界，`block_1` 不会误选 `block_10`。跨边界依赖成为子图输入，所需初始化权重随模型保存；外部使用的输出及子图末端输出成为边界输出。导出按依赖排序并进行 ONNX 检查。

无名算子仍保留为整数节点；模型输入输出本身是张量信息，不会额外伪装成算子。缺失的 `GraphProto.name` 会自动补齐。边界类型无法推断时需要真实 `value_info`，不会猜成任意 float/形状。控制流算子作为完整节点保留，不按前缀切开内部控制流子图；带训练绑定的模型不支持直接子图导出。

### 包命令行入口

`__main__.py` 提供以下子命令：

```powershell
# 保存完整 NetworkX 图
python -m onnx_graph convert model.onnx model.nx.pkl --hierarchy-source outputs

# 从可信的本机 pickle 恢复 ONNX
python -m onnx_graph restore model.nx.pkl restored.onnx

# 单前缀提取
python -m onnx_graph extract model.onnx subgraph.onnx --prefix encoder/block_1 --hierarchy-source outputs

# 旧 name 模式：根据张量路径修复名称
python -m onnx_graph repair-names model.onnx repaired.onnx --name-aliases onnx_graph/name_aliases.example.json
```

`convert/extract` CLI 及核心 API 的层级默认值为 `name`；如需按输出路径分层，应显式指定 `outputs`。pickle 仅加载可信来源。OMC 只支持 `convert` 到 GE NetworkX，不支持从 GE 图恢复为可执行 ONNX。

## 层级来源、名称修复与别名

### outputs 模式：推荐用于原名与路径差别很大的模型

`main.py`、独立 `stats.py` 默认使用 `HIERARCHY_SOURCE="outputs"`。例如：

```text
node.name:       SpNetFC_1_fconv1_0_Transpose
output:          /oSpatialNet_freq_cmps_1/fconv1.0/Transpose_output_0
hierarchy_path:  oSpatialNet_freq_cmps_1/fconv1.0/Transpose
```

保留原始算子名，用输出路径确定分层；profiling 仍通过原始名称匹配。去掉前导 `/` 和输出末尾 `_output_数字`、`:数字` 后，按实际 `/` 分段；点号和下划线保留。

slash 数是前缀内部 `/` 的数量，不计前导 `/`：

| slash 数 | 示例前缀 |
|---:|---|
| 0 | `oSpatialNet_freq_cmps_1` |
| 1 | `oSpatialNet_freq_cmps_1/fconv1.0` |
| 2 | `oSpatialNet_freq_cmps_1/fconv1.0/Transpose` |

不含 `/` 的输出归入 slash=0，例如 `plain_output_0` 对应 `plain`。多个平面输出使用第一个非空输出形成一级路径；多个路径输出使用共同完整层级，不重复计数。无输出或没有共同路径的节点保留，但不参加前缀分组，记录到 `output_hierarchy_issues.json`。

outputs 模式不改节点名称，`REPAIR_NAMES` 和别名映射不生效；`name_output_mismatches.json` 为 `null`。跨 slash 层级会重复包含节点，不可将不同层级的总量相加。

### name 模式：修复可对应的扁平化名称

设置 `HIERARCHY_SOURCE="name"`，并启用 `REPAIR_NAMES=True`。解析器根据输出或输入路径证据恢复层级，匹配时可统一比较 `/`、`*`、`.`、`_`；有歧义或名称冲突时保留原名。

别名 JSON 示例：

```json
{
  "SpNetFC_1": "oSpatialNet_freq_cmps_1"
}
```

`NAME_ALIASES_JSON` 或 CLI 的 `--name-aliases` 指向此文件。最长前缀优先且只替换一次；别名用于核对证据，不会不经验证强制改名。无法对齐的节点及原名、当前名、outputs、原因保存在 `name_output_mismatches.json`。

改变层级来源或名称配置后应重新生成统计，避免用旧前缀提取新模型。

## stats.py 与 profiling.py：分层计算量和算子耗时

### 独立生成统计

```powershell
python -m onnx_graph.stats model.onnx --slashes 0 1 2 3 --hierarchy-source outputs --output-dir D:/results/model_stats

# 附加算子 profiling 时间
python -m onnx_graph.stats model.onnx --slashes 0 1 2 --profile-csv D:/profile/model_op.csv --output-dir D:/results/model_profile_stats

# 指定统计时的输入形状；输入名应替换为模型实际名称
python -m onnx_graph.stats model.onnx --input-shape input=1,3,224,224 --no-batch-assumption
```

`stats.py` 输出 `stats.json`、`stats.txt`、`stats_<数字>_slash.csv` 和层级诊断。每种 slash 数一份 CSV，字段包括子图名字、节点数、各算子数量、MACs/FLOPs、已知部分、未知节点数和完整性。

支持 `--repair-names/--no-repair-names`、`--name-aliases`、`--batch-size`、`--no-profile`、`--ge-proto`。直接运行 `python onnx_graph/stats.py` 时可以使用文件顶部默认配置。供代码调用的主要函数为 `analyze_subgraphs(...)`、`write_reports(...)`。

### profiling 的格式和时间口径

`profiling.py` 由统计流程调用，读取 CSV 的 `total time(us)` 和 `op name` 或 `onnx_node_name` 列，可选 `model name`。同一文件若包含多个模型会报错，应先筛选当前模型。

- 同一运行算子多次记录先取均值，再对匹配到子图的算子均值求和。因此十次运行的逐算子记录不会直接相加为十倍时间。
- 同一图节点对应多个运行算子时，对各运行算子的均值求和。
- 支持精确名称、运行时类型后缀、唯一的分隔符归一化匹配，以及 name 模式的别名。
- 缺失和歧义不补零；完整 `exec_time` 留空，`已知exec_time` 保留可匹配部分。
- `exec_time` 为算子耗时之和，单位 us，不是考虑并行与调度后的端到端运行时间。
- 配置了 profiling 时，每个层级 CSV 都增加时间列；未配置时不生成这些列。

历史字段“算术强度”按用户定义计算 `FLOPs / exec_time`，单位 **FLOPs/us**，实质为计算吞吐率，不是传统的 FLOPs/Byte。只有 FLOPs 完整、时间完整且大于零时才计算。

## similarity.py：子图相似度和去重

```powershell
# 精确配置分组：建议用于减少性能测试数量
python -m onnx_graph.similarity D:/results/model/batch --mode exact

# 探索相似结构
python -m onnx_graph.similarity D:/results/model/batch --mode similar --threshold 0.90

# 允许跨 slash 层级比较
python -m onnx_graph.similarity D:/results/model/batch --mode exact --merge-levels
```

`exact` 比较连接方式、输入输出端口、算子类型、形状、dtype、opset 和属性。普通浮点权重数值可以不同，但形状控制常量、索引等影响语义的值仍参与比较；不是只比较算子数量。

`similar` 使用有向、端口感知的 WL 特征和计数加权 Jaccard 相似度，再按组内条件聚类。相似度高不保证性能相同。动态形状会记录提示；形状信息不完整或不支持的复杂图不会未经确认自动合并。

输出包含 `grouping.json`、`failures.csv`，以及各层级下的 `groups.csv`、`representatives.csv`、`similarity.csv`。单独运行只生成清单，不删除 ONNX；`main.py` 的 `KEEP_REPRESENTATIVES_ONLY=True` 才会清理本次选中的非代表文件。

## omc.py 与 ge_ir_pb2.py：OMC 解析支持

`omc.py` 读取 IMOD 文件头与分区表，通过 GE protobuf 解析区域、节点、张量与依赖，并可递归展开 GraphOp。图中保留 GE OpDef 序列化信息及原始模型内容；它不是 ONNX `NodeProto` 图。

```powershell
python -m onnx_graph convert model.omc model.omc.nx.pkl --hierarchy-source outputs
python -m onnx_graph.stats model.omc --slashes 0 1 2 --output-dir D:/results/omc_hierarchy_stats
```

仓库内置的 `ge_ir_pb2.py` 来自 DDK 6.1.1.0。解析器通过 AST 提取 protobuf descriptor，**不导入执行生成的 SDK 模块**。通常不需要安装 DDK；schema 路径优先级为：

1. 显式 `--ge-proto` / API 的 `ge_proto_path`。
2. 环境变量 `ONNX_GRAPH_GE_PROTO`。
3. `DDK_PATH` 下对应的 GE schema。
4. 包内 `ge_ir_pb2.py`。
5. 原工作区可发现的单个 DDK schema。

之前设置了无效环境变量时，应修正或清除，否则会优先于内置文件。当前主要验证未加密、可识别 GE 图定义的 IMOD 文件；其他版本或加密格式可能不支持。

`FusionOpNode` 可以读取节点、属性和可见输入输出，但内部原始算子数量与 FLOPs 未必可恢复，不能根据 kernel 名猜测计算量。

## compiled_stats.py：逐 OMC 整图静态统计

```powershell
python -m onnx_graph.compiled_stats --omc-dir D:/models/batch_omc --output-dir D:/results/compiled_stats

# 可选：将 OMC 结果关联到原始代表 ONNX 总表
python -m onnx_graph.compiled_stats --omc-dir D:/models/batch_omc --summary D:/results/batch/representatives_summary.csv --output-dir D:/results/compiled_mapped
```

递归扫描 `.omc/.om`，**每个文件作为一个完整编译后模型/子图统计**，不依赖文件内部是否还保留 `/`。输出：

- `omc_summary.csv/.json`：OMC 路径、SHA256、节点和各算子数量、计算量、数据移动量、状态与原因。
- `models/<名称与路径摘要>/stats.json`、`stats.txt`：逐节点明细及整图汇总，`compiled_scope=whole_omc`。

提供代表总表时，按文件名匹配，允许去掉编译时间后缀 `_fuse_YYYYMMDD_HHMMSS` 和 `_fixed`，也会参考 slash 层级。无法唯一匹配时可在源总表添加 `OMC文件` 列，指定绝对或相对于 OMC 根目录的路径。原 ONNX 统计及时间不冒充 OMC 的实测时间。

主要 API：`analyze_compiled_subgraphs(...)`、`number_models(...)`。独立脚本要求新输出目录；不接受覆盖已有目录。

## merge_power.py：编号、功耗合并与 summary 总表

### 常用命令

```powershell
# 功耗未提供：先统计 OMC，并生成编号清单
python -m onnx_graph.merge_power --batch-dir D:/models/batch_omc --output-dir D:/results/model_summary

# 合并功耗。新旧格式可放在同一个 power 文件夹
python -m onnx_graph.merge_power --batch-dir D:/models/batch_omc --power-dir D:/power --output-dir D:/results/model_summary

# 复用之前用于采集功耗的编号，校验模型内容
python -m onnx_graph.merge_power --batch-dir D:/models/batch_omc --power-dir D:/power --numbering D:/results/model_summary/numbering.csv --output-dir D:/results/model_summary
```

`--batch-dir` 模式支持直接运行 `python onnx_graph/merge_power.py` 并传相同参数。

### 文件夹与编号规则

递归收集全部 `.omc/.om` 后，按**完整文件名忽略大小写的字典序全局排序**，从 1 连续编号。目录与 slash 层级不参与优先排序；同名文件用相对路径字典序确定先后。

例如 `b/a.omc` 编号 1，`a/b.omc` 编号 2。字典序不是自然数字序，`model_10.omc` 会排在 `model_2.omc` 前面。文件可以直接放在根目录，也可放在 `slash_0/slash0`、其他子目录；slash 目录只影响“斜杠数”字段。

`numbering.csv` 保存编号、斜杠数、OMC 相对路径和 SHA256。新增、删除或改名会影响自动排序，采集后应保留编号表并用 `--numbering` 固定关系。传入编号表时，验证完整文件清单、SHA256、层级、连续且唯一的编号，**不重新排序覆盖表中编号**。

### 功耗文件命名与 DDR 频点

功耗文件直接放在 `--power-dir` 下，不递归搜索功耗子目录。

| 文件名 | 模型编号 | DDR 频点 | 读取方式 |
|---|---:|---|---|
| `1.csv` / `1.xlsx` | 1 | 未标明 | 旧平均值表 |
| `1_db.xlsx` | 1 | 未标明 | 新详细采样表 |
| `1-1.csv` / `1-1.xlsx` | 1 | 第 1 个，即 418 MHz | 旧平均值表 |
| `1-1db.xlsx` / `1-1_db.xlsx` | 1 | 第 1 个，即 418 MHz | 新详细采样表 |
| `1-9db.xlsx` | 1 | 第 9 个，即 4253 MHz | 新详细采样表 |

文件名中的 `db` 忽略大小写。模型编号完整匹配，1 不会误配到 10。频点分隔符是 **`-`**，不使用原先的 `1_1.csv`。

| 频点编号 | DDR MHz |
|---:|---:|
| 1 | 418 |
| 2 | 749 |
| 3 | 1066 |
| 4 | 1536 |
| 5 | 2102 |
| 6 | 2746 |
| 7 | 3187 |
| 8 | 3744 |
| 9 | 4253 |

发现有效的带频点命名时自动启用九频点统计；也可用 `--ddr-sweep` 强制启用。每个 OMC 只解析一次，但每个模型/频点各占总表一行；缺失的频点保留行并标 `missing`。

只有 `1.csv` 等无频点文件时不增加 DDR 字段。混合频点与无频点记录时，无频点行的 DDR 字段留空。`numbering.csv` 始终每个模型一行，不按频点重复。

### 新 db XLSX：原始采样求均值

- **只读取第一张工作表**，不按工作表名称选择，也不合并第二张。
- 首个非空行为表头，第一列为数值时间戳，单位 **us**。
- 后续只选列名中包含精确 `[W]` 的功耗列，例如 `VBUCK0_PERI [W]`；输出 Channel 名为 `[W]` 前的内容并去掉首尾空格。
- `[A]`、`[V]`、`[mW]` 等列不参与本次功耗均值计算。
- 对整张表的所有有效采样值求**算术平均**，结果为数值 W；不截取前 12 行/列，不做时间间隔加权。
- 空功耗单元格不补零，各 Channel 按自身有效样本数求均值。真实零值保留。
- 非数值、NaN、Inf、Excel 错误或某列全部为空时功耗标 `failed`；去掉 `[W]` 后名称为空或重复也报错。

详细表的时间口径：

```text
运行时长(s) = (末时间戳(us) - 首时间戳(us)) / 1,000,000
```

这是**采样窗口跨度**，不读取 H1；需要至少两个有限、严格递增的时间戳。时间戳异常时保留有效功耗均值，时长和吞吐率留空；功耗数值异常也不会丢掉独立可计算的时间戳跨度。若包含等待、多次推理或额外采样区间，不自动裁剪或推定循环次数。

### 旧平均功耗 CSV/XLSX

CSV 寻找 `Channel`、`Power` 表头，取其后前 12 条非空记录，优先选择 `Work0.db`，其次 `Work1.db` 对应数据段。支持 UTF-8/BOM、GB18030；数字文本和 `mW/W` 后缀原样保留，不重新求均值。重复 Channel、同优先级多表或无效功耗会报错。

旧 XLSX 优先读 `Average Values Work0.db`，其次 `Average Values Work1.db`；第 2 行为 `DB/Inst/Channel/Current/Voltage/Power` 表头，读取第 3～14 行。

旧格式运行时长读取 **第 1 行第 8 列 H1**，单位 s。CSV 不跳过空首行；XLSX 从选中的平均值 sheet 读 H1。零秒保留，但不计算除以零的吞吐率。

### db 与旧文件同目录时的优先级

同一模型、同一频点，优先使用带 `db` 的详细 XLSX：

```text
1_db.xlsx     优先于 1.csv / 1.xlsx
1-1db.xlsx    优先于 1-1.csv / 1-1.xlsx
```

功耗和时长均来自选中的 db 文件。没有 db 时才使用旧文件。被跳过的旧文件不删除，列在 `summary.json` 的 `ignored_power_files` 中；选中的 db 文件失败时不偷偷回退旧值。

多个 db 对应同一编号/频点时仍为 `ambiguous`。没有 db 但同时有多个旧候选时也报歧义。未知模型或非法频点文件列入 `unused_power_files`，不会自动当作其他测量。

### 输出文件和状态

```text
output_dir/
├── numbering.csv
├── summary.csv
├── summary.json
└── omc_stats/
    ├── omc_summary.csv
    ├── omc_summary.json
    └── models/.../stats.json、stats.txt
```

| 总表字段 | 含义 |
|---|---|
| `编号`、`OMC相对路径`、`OMC文件`、`OMC_SHA256` | 模型身份及测量对应关系 |
| `OMC名字`、`DDR频点编号`、`DDR频率_MHz` | 多频点模式附加的身份与频率 |
| `OMC_节点数`、`OMC_<算子>数量` | 编译后运算节点的统计 |
| `OMC_FLOPs`、`OMC_MACs` | 完整可估算的计算量；有未知算子时留空 |
| `cube_FLOPs`、`vector_FLOPs` | 将 OMC 理论 FLOPs 拆为矩阵/卷积乘加部分与其余标量操作；完整时两者之和等于 `OMC_FLOPs` |
| `已知cube_FLOPs`、`已知vector_FLOPs` | 有未知算子时仍保留可计算部分，两者之和等于 `OMC_已知FLOPs` |
| `OMC_已知FLOPs`、`OMC_未知计算量节点数` 等 | 部分已知的计算量及覆盖情况 |
| `OMC_输入及权重字节数`、`OMC_输出字节数`、`OMC_数据转移字节数` | 逐算子逻辑读写量，单位 Byte |
| `OMC_已知数据转移字节数`、`OMC_数据转移量完整` 等 | 未知描述符情况下的已知部分和完整性 |
| `运行时长`、`运行时长单位` | 旧表 H1 或新表时间戳跨度，统一为 s |
| `数据移动吞吐`、`数据移动吞吐单位` | 完整逻辑字节数 / 有效时长，单位 B/s |
| 各 Channel 列 | 平均功耗；新 db 表均值单位 W，旧表保留原值/原单位 |
| `OMC状态/原因`、`功耗状态/原因`、`时长状态/原因` | 解析和匹配诊断 |

`missing` 表示没有文件/信息，`failed` 表示解析失败，`ambiguous` 表示多个候选，`incomplete` 表示缺少其他测量中存在的 Channel。未知量不补零，一个文件失败不阻止其他模型统计。

### 兼容的旧 --summary 模式

`merge_power.py --summary ... --power-dir ... --stats-dir ...` 为历史固定编号的 56 个子图 + `0.xlsx` 整模型流程。它依赖代码中的固定 `MODEL_FILES`，不是通用动态编号模式；新模型使用 `--batch-dir`。旧模式输出单个 CSV，并拒绝覆盖已有输出文件。

## 数据移动量、FLOPs 与时间的计算口径

```text
张量字节数 = ceil(各维度乘积 × dtype 位数 / 8)
算子逻辑数据移动量 = 全部输入字节数（含权重、偏置） + 全部输出字节数
模型逻辑数据移动量 = 各运算节点逻辑数据移动量之和
数据移动吞吐 = 完整逻辑数据移动量 / 运行时长秒数
```

中间张量在生产者处计写出，在消费者处计读入；多个消费者分别计入。共享权重也按算子使用计入。`Const`、`Data`、`NetOutput` 不单独当作运算节点累加，展开后的 GraphOp 容器不重复统计。真实 Const 描述符用于替换消费者可能存在的 `[1]` 占位描述符。

这些是**逻辑字节数和估算吞吐率**，没有模拟缓存、片上复用、物理填充、内存对齐或额外 DMA，不能直接等同实测 DDR 流量/带宽。数据移动量“完整”只表示参与统计的可见描述符可计算，不表示已还原融合节点内部所有读写。

FLOPs 按一次稠密推理估算，1 MAC = 2 FLOPs，偏置及逐元素运算另计。部分移动/形状算子计 0 FLOPs，但仍可能有搬运和时间开销。未知形状、未支持算子、无法可靠还原的 FusionOpNode 计算量保留未知；不把未知算作 0。算子数量为对应输入图的节点类型数量，OMC 融合节点数量不是原始 ONNX 算子数量。

Cube/Vector 按理论算术类型拆分，保持原有总 FLOPs 不变：

```text
cube_FLOPs   = 2 × MACs
vector_FLOPs = FLOPs - 2 × MACs
```

卷积、MatMul/BatchMatMul/Gemm 等的乘加部分计入 Cube；偏置、融合 ReLU、逐元素运算、激活、归一化、池化、Softmax 等现有公式中的其余操作计入 Vector。该拆分描述逻辑计算量，不推断实际内核被安排到哪个硬件单元，也不包含物理补齐或重复执行开销。举例：Gemm 的乘加为 48 FLOPs、偏置为 8 FLOPs，则 Cube=48、Vector=8、总量=56。

只要有未知计算量节点（例如不透明 FusionOpNode），完整 `cube_FLOPs` 和 `vector_FLOPs` 均留空；`已知cube_FLOPs`、`已知vector_FLOPs` 单独统计已知节点，不将部分值冒充完整值。`stats.json` 的 total/子图汇总中对应 `cube_flops`、`vector_flops`、`known_cube_flops`、`known_vector_flops`。独立 `compiled_stats.py` 的 OMC 总表使用 `OMC_cube_FLOPs` 等带 `OMC_` 的列名以区分原 ONNX 统计；最终功耗 `summary.csv` 使用上面的无前缀列名。九频点的每一行复用所属 OMC 的静态拆分结果。

更新 `stats.py`、`compiled_stats.py` 和 `merge_power.py` 后，用原命令重新生成 summary 即可，不需要新参数。已有 CSV 不会自动迁移。

三种时间不能混用：profiling `exec_time` 是算子均值之和（us）；旧功耗 H1 是文件记录的时长（s）；新 db 时长是时间戳采样跨度（us 转 s）。均不自动推定推理循环次数。

## 绘图脚本

所有绘图均只读源数据。CSV/XLSX 均支持的脚本默认读 XLSX 第一张 sheet，可用 `--sheet` 指定；波形脚本固定读取详细 db XLSX 第一张工作表。

### plot_ddr_power.py：频率曲线、模型总览、极差图

```powershell
python -m onnx_graph.plot_ddr_power --summary D:/results/summary.csv

# 配置组件和统一 Y 轴：单位为 mW
python -m onnx_graph.plot_ddr_power --summary D:/results/summary.csv --channels VBUCK0_PERI DDRPHY VDD_NPU --y-min 0 --y-max 1000 --y-step 100
```

- 默认组件：`VBUCK0_PERI`、`VBUCK8_1V8`、`VBUCK1_1V12`、`DDRPHY`，通过 `--channels` 或顶部 `CHANNELS` 增删。
- 每个模型一张折线图，X 轴为 9 个 DDR MHz，Y 轴为 mW；同次运行的模型图共享 Y 轴范围和刻度。
- `all_models_ddr_power.png/.svg` 为全部模型总览，每个模型一个子图。
- `all_models_power_range.png/.svg` 为各频点、各组件的跨模型 `max - min`，单位 mW，使用独立从零开始的 Y 轴；这是绝对极差，不是百分比变化率。
- 缺失点断线，极差不足两个有效模型时也断线；不补零。无频率记录跳过，同一模型/频点重复时报错。
- 输出 PNG 和 SVG，默认与 summary 同目录，`--output-dir` 可修改。输入数字默认 W；`--power-unit mW` 表示无单位输入数字原本是 mW，不是设置显示单位。

### plot_ddr_runtime.py：每个模型的 DDR 频率—执行时间折线图

```powershell
python -m onnx_graph.plot_ddr_runtime --summary D:/results/summary.csv
# 可选输出目录；默认与 summary 同目录
python -m onnx_graph.plot_ddr_runtime --summary D:/results/summary.csv --output-dir D:/results/runtime_plots
```

每个模型输出一张独立折线图的 PNG（300 DPI）和矢量 PDF，文件名带 `_ddr_runtime` 后缀，不覆盖功耗图。X 轴按数值顺序固定显示 9 个 DDR 频率（MHz），Y 轴为执行时间 **ms**，同次运行的模型图使用相同 Y 轴范围。按 OMC 相对路径区分模型，同名模型位于不同目录时生成不同文件名。

默认读取 merge_power 总表的 `运行时长` 和 `运行时长单位`，将秒乘以 1000 显示为毫秒，不重新计算或推定单次推理时间。也支持 `执行时间`、带 `(s)/(ms)/(us)` 单位的时间列；输入单位未知时必须通过 `--time-unit s/ms/us` 指定。XLSX 默认第一张 sheet，可用 `--sheet` 指定。命令的 `--time-unit` 控制输入解释，图上始终显示 ms。

缺失频点或时间留空时断线，真实零值保留；无频率记录跳过，同一模型/频率重复或时间为负数、NaN、Inf 时报错。若全部时间都缺失，则停止并提示检查数据。源表只读；已有目录可用，仅覆盖同名执行时间图片。

### plot_channel_throughput.py：DDR 相关 Channel 的吞吐率—功耗折线图

```powershell
python -m onnx_graph.plot_channel_throughput --summary D:/results/summary.csv

# 可选固定频率与组件
python -m onnx_graph.plot_channel_throughput --summary D:/results/summary.csv --channels VBUCK0_PERI DDRPHY --ddr-frequency 418
```

每个 Channel 一条带数据点的折线，按吞吐率从小到大连接测量记录，不同 Channel 用不同颜色。相同吞吐率的记录保留原始顺序，不对其求均值。X 轴自动换算到 GB/s（十进制），Y 轴默认从 W 换算成 mW，可用 `--display-power-unit W` 切换。

默认仅绘制 `VBUCK0_PERI`、`VBUCK1_1V12`、`DDRPHY`，也可修改脚本顶部 `CHANNELS` 或通过 `--channels` 增删组件、调整图例顺序。兼容 summary CSV 和此前示例的 XLSX 括号说明列名。无 DDR 字段的旧总表也能画；有频率时可筛选。

输出 `channels_power_vs_throughput.png/.pdf`（PNG 300 DPI），默认与 summary 同目录。筛选频率时文件名增加频率后缀。缺失配对跳过，零值保留，不拟合或平滑。未指定频率时，将全部测量按吞吐率排序连接，因此可能跨模型和频率；连线仅辅助观察，不能当作固定频率下的因果关系。需要固定频率比较时传入 `--ddr-frequency`。

### plot_power_throughput.py：固定频率论文四宫格

```powershell
python -m onnx_graph.plot_power_throughput --summary D:/results/summary.xlsx --output-dir D:/results/fixed_frequency
```

四个子图依次为 `(a) VBUCK0_PERI`、`(b) VBUCK1_1V12`、`(c) DDRPHY`、`(d) VBUCK8_1V8`。X 轴 GB/s，Y 轴按论文功能要求为 **W**，各子图独立设置范围。

每条线是**同一 DDR 频率下的不同模型**，按吞吐率从小到大连接。颜色/公共 colorbar 表示 DDR MHz，marker 和公共图例表示模型，不跨频率连线。当前提供 12 种模型 marker，超过 12 个模型需扩展配置。

| 输出 | 内容 |
|---|---|
| `power_vs_throughput.png` | 300 DPI 四宫格 |
| `power_vs_throughput.pdf` | 矢量 PDF |
| `fixed_frequency_statistics.csv` | 每个频率/电源轨的有效模型数、最大/最小/均值功耗，以及 `(max-min)/mean × 100%` |
| `correlation_analysis.csv` | 每个频率/电源轨的有效配对数、Pearson、Spearman |
| `validation.json` | 源文件、模型、吞吐单位、缺失频点和缺失值 |

默认输出到 summary 旁的 `power_throughput_analysis/`。吞吐单位从列名或 `数据移动吞吐单位` 读取，不明确时必须通过 `--throughput-unit B/s/MB/s/GB/s` 指定。样本不足、常量输入或均值为零时，相应统计留空。

相关性按每个频率独立计算，不用全部混合样本代替。每频率只有三个模型时，结果仅供探索，不能解释为统计显著或因果关系；连接线是离散模型间的视觉辅助，不是拟合曲线。不会为增强相关性剔除或平滑数据。

### plot_power_waveform.py：指定 db 的单 Channel 波形

```powershell
python -m onnx_graph.plot_power_waveform --db D:/power/1_db.xlsx --channel DDRPHY
python -m onnx_graph.plot_power_waveform --db D:/power/1-1db.xlsx --channel VBUCK0_PERI --output-dir D:/results/waveforms

# 可选显示单位：原始时间戳仍按 us、原始 [W] 列仍按 W 解析
python -m onnx_graph.plot_power_waveform --db D:/power/1_db.xlsx --channel DDRPHY --time-unit us --power-unit W
```

只读取第一张 sheet 的 `[W]` 功耗列，支持 Channel 名或完整列名（例如 `--channel "DDRPHY[W]"`）。无法唯一匹配时列出可选 Channel。

横轴从首个时间戳开始，默认 **ms**；纵轴默认 **mW**。逐点绘制原始采样，不取均值、不平滑、不重新排序。空功耗保留为断线；非有限/无效功耗或缺失、重复、倒序时间戳报错。

输出 `<db文件名>_<channel>_waveform.png`（300 DPI）和 `.pdf`，默认在输入文件旁。`--time-unit` 可选 `us/ms/s`，`--power-unit` 可选 `W/mW`；此脚本的 `--power-unit` 指**显示单位**，与其他脚本的同名输入单位参数含义不同。

## 输出目录的覆盖行为

| 入口 | 目录/文件已存在时 |
|---|---|
| `merge_power.py --batch-dir` | **删除整个输出目录后重建**，其中旧图及其他文件也会被删除；先检查输入路径，防止删除模型、功耗输入、schema 或当前工作目录 |
| `merge_power.py --summary` | 拒绝覆盖旧输出 CSV |
| 独立 `compiled_stats.py` | 拒绝使用已有输出目录，需指定新目录 |
| `stats.py` | 更新同名报告；不自动清除其他层级遗留 CSV |
| `main.py` 批量导出 | 复用可确认一致的已有导出；不一致报错；开启去重时清理本次非代表 ONNX |
| 绘图脚本 | 创建目录或覆盖同名图，不清空其他文件 |
| `python -m onnx_graph` 转换/单次提取 | 源与目标不能相同，但目标可以被写入覆盖 |

若 `--numbering` 指向 merge 输出目录中的旧编号表，会先读取校验，再删除旧输出并重新写出编号表。绘图默认在 summary 旁，因此重新运行同一 merge 输出目录时，旧图也会删除，需要重新绘图。

## verify_mobilenet.py：样例数值验证

```powershell
python -m pip install onnxruntime
python -m onnx_graph.verify_mobilenet
```

固定读取 `onnx_graph/models/mobilenetv3/mobilenetv3-large_224_1.0.onnx`（与 `main.py` 的默认路径不同），验证完整 ONNX ↔ NetworkX 往返、`MobilenetV3/expanded_conv_1` 提取、真实激活边界及 ONNX Runtime 数值一致性，并写到模型旁的 `networkx/`。

它是特定 MobileNetV3 的复现脚本，输入形状和前缀写在文件内，不是任意模型的通用验证器。该模型不随仓库发布；缺少本机模型或额外依赖时不能直接运行。

## 测试文件：逐个说明

测试文件都在 `onnx_graph/tests/`，不是生产输入脚本。它们使用临时目录和构造数据验证功能，不需要公司实际模型或测量文件。

| Python 测试文件 | 覆盖内容 |
|---|---|
| `test_graph.py` | 完整 ONNX/NetworkX 往返、空名称/空图名、前缀边界、多重边、编辑后刷新、外部权重、控制流、函数和元数据 |
| `test_output_hierarchy.py` | outputs 分层、slash=0 平面输出、多输出共同层级、保留原名、profiling 对齐、旧统计防混用 |
| `test_name_aliases.py` | JSON 别名校验、最长前缀、证据和歧义检查、诊断记录、profiling 共用别名与 CLI |
| `test_stats.py` | ONNX 分层计数、各类计算量公式、动态形状和 batch 假设、未知算子、CSV/JSON 输出 |
| `test_profiling.py` | 多轮均值、多个运行算子映射、部分匹配、未知/零值、时间 CSV 校验及可选开关 |
| `test_pipeline.py` | main 统计→筛选→导出→去重→代表总表完整流程，复用统计、失败停止、OMC 仅统计分支 |
| `test_batch.py` | AND/OR 筛选、数值/文本配置校验、批量前缀导出、已有文件、模型与统计一致性、逐条失败原因 |
| `test_batch_dedup.py` | 仅保留代表的联动、重复运行、比较/清理失败保护、越界与文件变化检查、跨层级设置 |
| `test_similarity.py` | 连接和端口、权重与控制常量、形状/属性差异、同构校验、WL 碰撞、完整链接聚类、失败报告 |
| `test_omc.py` | OMC 区域、嵌套图、分区表、原始信息、GE schema 安全读取、层级、转换 CLI 和未知物理形状 |
| `test_ge_costs.py` | GE 矩阵乘法/转置/广播、Const 描述符替换、归一化、PRelu、移动算子和不透明融合算子 |
| `test_compiled_stats.py` | OMC 整图总表、代表映射、缺失/歧义、未知值、嵌套目录及输出覆盖保护 |
| `test_batch_power.py` | 动态编号、九频点、新 db 首表/全部采样均值、us 时长、db 优先级、旧格式兼容、吞吐和目录覆盖保护 |
| `test_merge_power.py` | 旧固定编号 56 子图 + 整模型 Excel 合并，以及数据移动量、缺失值、冲突和文件保护 |
| `test_plot_ddr_power.py` | 单位、缺失和重复点、统一 Y 轴、总览及按频率/组件计算极差 |
| `test_ddr_runtime.py` | 按模型分组、频率排序、秒/毫秒/微秒换算、重复和异常时间、缺失断线、零值和默认输出目录 |
| `test_channel_throughput.py` | 默认 DDR Channel 与自定义组件、频率筛选、缺失与零值、单位换算、排序折线、图例和默认输出目录 |
| `test_power_throughput.py` | 论文图输入的单位、重复模型/频点校验，相关性正负方向及常量/不足样本 |
| `test_power_waveform.py` | db 第一张 sheet、Channel 匹配、原始采样和断线、时间戳检查、ms/mW 换算和源文件只读 |

运行全部测试：

```powershell
python -m unittest discover -s onnx_graph/tests -v
```

运行指定测试文件（去掉 `.py` 后缀）：

```powershell
python -m unittest onnx_graph.tests.test_batch_power -v
python -m unittest onnx_graph.tests.test_power_waveform -v
```

Excel/绘图集成测试需要 openpyxl/Matplotlib，部分数值测试需要 onnxruntime。缺少可选依赖时部分测试会跳过，应检查测试输出中的 `skipped`，不要将跳过当作已验证。

## 常见问题

**只有 batch_export.csv/json，没有 ONNX？** 查看清单中 `failed` 行的原因，而不是只看 `filtered` 行。检查过滤条件、原始模型、层级来源与统计前缀是否一致。单看 `已知exec_time` 只代表已匹配部分，筛选完整时间时使用 `exec_time`。

**某个 slash CSV 没有 exec_time？** 确认本次配置了 `PROFILE_CSV/--profile-csv`，重新生成该层级 CSV；不要混入上次未带 profiling 的旧报告。

**FusionOpNode 很多或 OMC_FLOPs 为空？** 融合节点可以解析，但内部计算量可能未知。查看 `OMC_已知FLOPs`、未知节点数和明细，不能把空值解释为零。

**ge_ir_pb2.py 找不到？** 更新整个包以包含内置 schema；修正旧环境变量，或用 `--ge-proto` 显式指定其他兼容文件。

**summary 中功耗有值但吞吐为空？** 检查时长是否有效且大于零，以及完整数据移动量是否已知；新 db 还需检查 us 时间戳是否严格递增。

**同目录有 1.csv 和 1_db.xlsx？** 当前优先 db，旧文件原样保留；被跳过文件见 `ignored_power_files`。多个 db 对应同一测量仍需人工保留一个。

**绘图提示缺少 Channel？** 检查实际列名，使用 `--channels` 选择存在的列；波形则用 `--channel`。不要将无数据点补成零以消除报错。

**查看实际 CLI 参数？** 除配置式 `main.py` 和特定样例 `verify_mobilenet.py` 外，各 CLI 可以追加 `--help`。`core.py`、`profiling.py`、`omc.py` 和 `__init__.py` 是支持模块；`ge_ir_pb2.py` 是生成的定义文件，不直接运行。
