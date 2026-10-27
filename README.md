# ONNX ↔ NetworkX

将 ONNX 转为 `networkx.MultiDiGraph`，保留完整 protobuf 信息和 `/` 路径层级，按前缀提取子图，再生成可独立执行的 ONNX。建议 Python 3.10（最低 3.10）；下方命令从克隆后的仓库根目录运行。

## 安装与模型位置

```powershell
git clone https://github.com/Kaiwei-Lin/op_power_subgraph_extract.git
cd op_power_subgraph_extract
python -m pip install -r onnx_graph/requirements.txt
```

将本机模型和 profiling CSV 放入以下位置（目录不存在时自行创建）：

```text
op_power_subgraph_extract/
  onnx_graph/
    main.py
    models/
      handfree_ainr/
        handfree_ainr.onnx
        handfree_ainr_fuse_20261020_102316_op.csv
```

仓库仅包含代码、测试、依赖清单和说明；模型、profiling CSV、生成的报告、压缩包及 Python 缓存由 `.gitignore` 排除。本文的 MobileNetV3 示例和验证结果来自此前本机测试，相关模型与产物不随仓库提供。

在原工作区使用源码时，包位于 `tools/onnx_graph/`；先进入 `tools` 目录即可使用下方相同命令，原来的 `tools.onnx_graph` 导入方式仍然可用。

## 直接运行 main.py

已经提供 `onnx_graph/main.py`，可以放在 `onnx_graph` 包内使用。默认一次完成 **生成统计 → 筛选 → 批量导出 → 去重 → 汇总代表时间**，只保留本次分组的代表 ONNX；也保留原来的单前缀示例。无需先单独运行 `stats.py`。默认读取上面的 handfree_ainr 模型和 CSV，路径基于 `__file__` 计算，不依赖启动目录。更新时请同步整个代码包。

在仓库根目录直接运行：

```powershell
python onnx_graph/main.py
```

或在项目根目录运行：

```powershell
python -m onnx_graph.main
```

### 统一配置统计、筛选和批量导出

在 `main.py` 顶部统一配置：

```python
PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_DIR / "models/handfree_ainr/handfree_ainr.onnx"
HIERARCHY_SOURCE = "outputs"  # 默认按输出路径分层，保留原始 name
REPAIR_NAMES = True  # 仅 HIERARCHY_SOURCE="name" 时生效
NAME_ALIASES_JSON = None  # 仅 name 模式生效
OUTPUT_DIR = MODEL_PATH.parent / "networkx/main_outputs"
BATCH_FROM_STATS = True
GENERATE_STATS = True  # 默认先生成本次统计，再筛选、导出
STATS_DIR = MODEL_PATH.parent / "networkx" / (MODEL_PATH.stem + "_stats")
SLASH_COUNTS = [1, 2]
PROFILE_CSV = MODEL_PATH.parent / "handfree_ainr_fuse_20261020_102316_op.csv"
# PROFILE_CSV = None  # 不需要时间统计时禁用
BATCH_SIZE = 1      # 仅用于统计估算；None 禁用未知第 0 维的 batch 假设
INPUT_SHAPES = {}   # 仅用于统计估算；例如 {"input": [1, 3, 224, 224]}
GE_PROTO_PATH = None  # 仅 OMC 统计可能需要配置
FILTERS = []  # 默认不过滤，所有 CSV 中的子图均参与导出与分组
KEEP_REPRESENTATIVES_ONLY = True  # 分组完成后删除本次选中的非代表 ONNX
GROUP_MODE = "exact"             # 或 "similar"
SIMILARITY_THRESHOLD = 0.90      # 仅影响 similar 模式
SEPARATE_SLASH_COUNTS = True     # 不同 slash 层级分别分组
```

### 按 outputs 分层、按原始 name 匹配时间（默认）

`main.py` 和独立 `stats.py` 默认使用 `HIERARCHY_SOURCE = "outputs"`。例如：

```text
原始 node.name: SpNetFC_1_fconv1_0_Transpose
output:         /oSpatialNet_freq_cmps_1/fconv1.0/Transpose_output_0
hierarchy_path: oSpatialNet_freq_cmps_1/fconv1.0/Transpose
```

一个 `/` 的分组为 `oSpatialNet_freq_cmps_1/fconv1.0`，两个 `/` 的分组为 `oSpatialNet_freq_cmps_1/fconv1.0/Transpose`。不要求 name 与 output 能互相转换，也不需要 JSON 别名。点号和真实下划线原样保留。

- `name`、`original_name` 和 NodeProto 保留原始算子名，导出的 ONNX 也保留原名；`hierarchy`、`hierarchy_path` 专门保存输出层级。连接关系仍由张量依赖确定。
- profiling 以原始 node.name 为索引，支持原有的运行时类型后缀处理及唯一的分隔符归一化匹配；不把 output 路径加入名称索引。空名称、重名或无法匹配的时间保持未知，不补零。
- 输出末尾 `_output_数字` 或 `:数字` 会去除。多个输出去掉后缀后指向同一路径时，只计一个节点；路径不同时仅分到它们共有的完整父层级，不重复计数、不任选某个分支。
- 不含 `/` 的平面输出直接作为 slash=0 的一级分组名，去除 `_output_数字` 或 `:数字` 后缀；若去除后为空，保留完整输出名。多个平面输出取第一个非空输出作为组名，一个节点只计一次，全部输出仍保留。它们不参加 slash>=1 的分组，记录为层级较浅，也不再报 `no_output_path`。例如 `plain_output_0` 归入 `plain`；设置 `SLASH_COUNTS = [0, 1, 2]` 即可生成 `stats_0_slash.csv`。
- 没有输出或带 `/` 的输出之间没有共同层级时，保留节点但不参加前缀分组。不会回退到 name 或 input 猜层级。无名节点只要有可用输出层级，也可参加分组。
- 控制台完整打印多输出共同层级和无法分层节点的 name、outputs、分配路径及原因；记录保存到 `output_hierarchy_issues.json`（统计目录、批量导出目录或单前缀输出目录）和主 JSON 报告。`common_output_scope` 表示已归入共同层级；其他原因表示未分层。成功检查且无此类节点时为 `[]`，没有选中子图而未加载模型时为 `null`。
- outputs 模式不运行 name/output 对齐检查，`name_output_mismatches.json` 写为 `null`，避免保留上次名称模式的诊断。`REPAIR_NAMES` 和 `NAME_ALIASES_JSON` 在该模式下不生效。

切换后设置 `GENERATE_STATS = True` 重新生成统计，并使用新的 `OUTPUT_DIR`。默认输出目录已改为 `networkx/main_outputs`。新 CSV 带有 `层级来源=outputs`；缺少该列或来源不同的旧 CSV 不可用于 outputs 导出。已有 ONNX 内容与本次提取不一致时会保留文件并报告失败，要求改用新目录，以免把旧子图与新时间统计混用。

函数 API 为兼容已有脚本，默认仍使用 name；请显式传入 `hierarchy_source="outputs"`，统计和导出保持一致：

```python
graph = onnx_to_networkx("model.onnx", hierarchy_source="outputs")
report = analyze_subgraphs("model.onnx", [1, 2], profile_csv="model_op.csv", hierarchy_source="outputs")
export_from_stats("model.onnx", "stats", "new_subgraphs", hierarchy_source="outputs")
```

```powershell
python -m onnx_graph.stats model.onnx --hierarchy-source outputs --profile-csv model_op.csv
python -m onnx_graph extract model.onnx subgraph.onnx --hierarchy-source outputs --prefix oSpatialNet_freq_cmps_1/fconv1.0
```

`convert/extract` CLI 的默认值仍为 name；如需按 outputs 分层请传上面的参数。OMC 使用真实 `output_desc.name` 分层，缺失时不会采用合成的 `算子名:端口` 作为层级证据。旧流程可设置 `HIERARCHY_SOURCE = "name"`，继续使用下面的名称修复和 JSON 别名功能。

`GENERATE_STATS = True` 时，在 `STATS_DIR` 更新 `stats.json`、`stats.txt` 和 `SLASH_COUNTS` 对应的 CSV，后续**只读取本次生成的 CSV**。以前运行遗留的其他层级或 `stats_01_slash.csv` 等文件不会混入；这些旧文件不删除。统计或 profiling 读取失败时立即停止，不会退回使用旧表继续导出。`stats.py` 顶部的配置不影响这一自动流程，以 `main.py` 为准。

如需复用手工修改过的统计，设置 `GENERATE_STATS = False`。此时跳过统计，读取 `STATS_DIR` **当前目录**里的全部 `stats_<数字>_slash.csv`，不递归；`SLASH_COUNTS`、`PROFILE_CSV`、`BATCH_SIZE` 等统计设置不生效。前导、末尾和重复 `/` 会归一化；`expanded_conv_1` 不会匹配 `expanded_conv_10`。CSV 必须包含 `子图名字`、`节点数` 和筛选用到的列。

`MODEL_PATH` 为 OMC/OM 且启用自动统计时，只生成统计报告并说明后续步骤未运行；编译图不能直接导出为可执行 ONNX。默认 OMC 统计目录使用 `_omc_stats` 后缀，与 ONNX 分开。

每个条件是 `(列名, 比较符, 阈值)`。**同一内层列表中的条件为 AND，不同内层列表之间为 OR**：

```python
# 完整 exec_time > 100 us；空时间不会通过筛选
FILTERS = [[("exec_time", ">", 100)]]

# 节点数 > 5
FILTERS = [[("节点数", ">", 5)]]

# 完整 exec_time > 100 us 且 节点数 > 5
FILTERS = [[("exec_time", ">", 100), ("节点数", ">", 5)]]

# 完整 exec_time > 100 us 或 节点数 > 5
FILTERS = [[("exec_time", ">", 100)], [("节点数", ">", 5)]]

# (exec_time > 100 us 且 节点数 > 5) 或 (已知exec_time > 500 us)
FILTERS = [
    [("exec_time", ">", 100), ("节点数", ">", 5)],
    [("已知exec_time", ">", 500)],
]

# 任意现有数值列都可使用；同一列可同时设置下限和上限
FILTERS = [[("Conv数量", ">=", 1), ("FLOPs", ">=", 1e6), ("FLOPs", "<", 5e7)]]

# 文本列可做相等/不等判断
FILTERS = [[("计算量完整", "==", "是"), ("节点数", ">", 5)]]
```

支持 `>`、`>=`、`<`、`<=`、`==`、`!=`；数值阈值写成数字，文本只支持 `==`、`!=`。数值比较使用十进制数，`>` 不包含阈值本身。空白数值不当作 0，也不会通过 `!= 0` 等数值条件。缺少筛选列会明确报错；未添加 profiling 的 CSV 不能筛选 `exec_time`。

`exec_time` 是完整子图的单次平均算子耗时合计；`已知exec_time` 是已匹配部分的合计，适合允许时间不完整的筛选。两列单位都是 **us**，无需再除以 profiling 的运行次数。未知 FLOPs 留空的规则相同。

导出结果保存在：

```text
OUTPUT_DIR/
  batch/
    slash_1/              # 一个 / 的代表子图 ONNX
    slash_2/              # 两个 / 的代表子图 ONNX
    grouping_exact/      # similar 模式对应 grouping_similar
      slash_1/           # groups.csv、representatives.csv、similarity.csv
      slash_2/
      grouping.json
      failures.csv
    batch_export.csv      # 每行的前缀、状态、导出路径、输入输出及失败原因
    batch_export.json     # 同时保留筛选条件和各行原始统计值
    output_hierarchy_issues.json # 输出层级需关注的节点（含共同父层级或未分层）
    name_output_mismatches.json  # 旧 name 模式诊断；outputs 模式为 null
    representatives_summary.csv  # 所有层级代表子图的时间总表
```

文件名由清理后的前缀和稳定摘要组成，避免 Windows 非法字符、过长前缀和重名覆盖。相同层级的重复前缀只处理一次；已有 ONNX 不覆盖，但若参与本次去重且被判为非代表，会删除该子图文件。需重新导出模型内容时请换一个输出目录。CSV/JSON 清单每次更新为本次运行结果，未被本次筛选选中的旧模型及目录内其他文件不会参与分组或删除；若希望目录严格只包含本次运行的模型，请使用新的 `OUTPUT_DIR`。

默认流程为：**生成统计 → 筛选统计行 → 导出子图 → 按连接关系与计算配置分组 → 写出代表清单 → 删除非代表 ONNX → 汇总代表时间**。在各层级的 `representatives.csv` 中读取 `ONNX文件` 即可得到后续测试文件列表。原始完整模型不参与清理。将 `KEEP_REPRESENTATIVES_ONLY = False` 可恢复导出全部模型、跳过去重的行为；单前缀模式不受影响。导入函数 `export_from_stats` 时默认不去重，需要显式传入 `deduplicate=True`，并可传 `group_mode`、`similarity_threshold`、`separate_slash_counts`；`csv_paths` 可限定读取统计目录中的指定文件，不传则读取全部匹配文件。

**代表子图时间总表** `OUTPUT_DIR/batch/representatives_summary.csv` 将所有层级合在一张表，每个代表一行，包含：

```text
斜杠数,分组,子图名字,ONNX文件,节点数,同组子图数,Add数量,Conv数量,Relu数量,...,已知exec_time,exec_time,exec_time_unit,时间匹配节点数,时间未匹配节点数
```

算子按类型各占一列，列名沿用统计表的 `<op类型>数量`，按名称排序；自定义算子保留 domain，例如 `custom::Relu数量`。数量来自该代表子图在 ONNX 中的节点计数，不乘以同组子图数；没有的算子填 `0`。即使复用的旧统计 CSV 没有算子数量列，也会从 ONNX 补齐。没有 profiling 时仍输出算子数量。

`已知exec_time` 直接取该代表子图统计行的值，不把同组其他子图的时间累加；仍是多次 profiling 先按算子取平均、再将已匹配算子的平均时间相加，单位 `us`。`exec_time` 在任一节点缺少时间时留空，而 `已知exec_time` 保留已匹配部分；匹配/未匹配节点数显示覆盖情况。没有提供 profiling 时，时间和覆盖情况列留空；测得的真实零值仍写 `0`。不同 slash 层级可能覆盖相同运算，因此总表不添加跨层级时间合计。未启用去重、没有代表结果或整体分组失败时，总表只有表头，避免将上次运行的代表误当作本次结果。

本机已用 MobileNetV3 与 `_20261016_110720_op.csv` 跑通完整流程：统计 97 个子图、保留 90 个代表，总表 90 行；全部时间值逐行与统计表一致，其中 45 行有完整 `exec_time`，45 行仅有已知部分。示例位于 `onnx_graph/models/mobilenetv3/networkx/main_pipeline_with_profile_20261020/`，总表为其中的 `batch/representatives_summary.csv`，验证记录为 `pipeline_validation.json`。原始 ONNX 和 profiling CSV 的内容摘要均未改变。

清单状态包括 `exported`（已导出并保留）、`filtered`（未通过筛选）、`duplicate`（同前缀重复行）、`exists`（已有文件仍保留）、`deduplicated`（已删除非代表文件）和 `failed`（导出失败）。`batch_export.csv` 新增 `去重状态`、`分组`、`代表ONNX`、`与代表相似度`、`去重说明`；被删除行的 `ONNX文件` 是历史路径，实际测试应使用 `代表ONNX`。JSON 的 `deduplication` 记录本次分组配置及保留/删除数量。

不能解析或不支持比较的 ONNX 会保留，标记 `kept_unchecked` 并写出原因；这些文件不在代表清单里，需单独测试。缺少形状信息的可解析模型保留为独立代表。删除失败则标记 `cleanup_failed`，保留文件。整体分组失败或检测到分组期间文件变化时，不进入删除阶段。一个子图导出失败后继续处理其他子图；零匹配时仍保存空分组和导出清单，不导出整图。批量模式不额外保存完整模型的 pickle。

只看到 `batch_export.csv/json` 不代表导出成功：先看清单中的“状态”和“原因”。例如 `{'filtered': 6267, 'failed': 16}` 表示 6267 行未通过筛选、16 行处理失败，没有任何成功导出；应查看 **failed 行**的原因。`filtered` 行也会记录筛选条件及实际值，空值显示为 `<空>`。控制台直接显示输出目录及各类最多三条原因，完整信息保留在清单中。统计与导出必须使用相同的 `HIERARCHY_SOURCE`；旧 name 模式下 `REPAIR_NAMES` 和别名配置也须保持一致。

每个导出模型都使用现有依赖提取逻辑，保留所需权重、推断边界输入输出并运行 ONNX checker。同时检查所选前缀的实际节点数和 CSV 中的算子数量，发现不一致会记录失败。请使用**同一原始 ONNX、相同名称修复设置**生成的统计；这种数量校验不能证明两份同结构但不同权重的模型相同。统计时的 batch/输入尺寸假设不会自动改写导出模型的形状。OMC 编译图不支持直接导出 ONNX，也不能假定其融合后的 CSV 与原始 ONNX 一一对应。

### 原来的单前缀示例

将 `BATCH_FROM_STATS = False`，设置 `MODEL_PATH`、`PREFIX`、`HIERARCHY_SOURCE` 即可继续使用。示例会打印所选节点及其输入输出，保存完整图为 `full_graph.nx.pkl`，导出子图为 `subgraph.onnx`。默认保存到模型所在目录下的 `networkx/main_outputs/`；重复运行会更新这些单前缀示例产物。启用名称修复时还会保存 `rename_report.json`。

## 快速使用

```python
from pathlib import Path
import onnx
from onnx_graph import (
    onnx_to_networkx, nodes_by_prefix, extract_prefix, networkx_to_onnx,
)

source = Path("onnx_graph/models/mobilenetv3/mobilenetv3-large_224_1.0.onnx")
graph = onnx_to_networkx(source)

node_ids = nodes_by_prefix(graph, "MobilenetV3/expanded_conv_1")
for node_id in node_ids:
    print(node_id, graph.nodes[node_id]["name"], graph.nodes[node_id]["op_type"])

subgraph = extract_prefix(graph, "MobilenetV3/expanded_conv_1")
model = networkx_to_onnx(subgraph)
onnx.save(model, source.parent / "expanded_conv_1.onnx")

# 不做编辑时，完整模型可以无损往返。
restored = networkx_to_onnx(graph)
assert restored.SerializeToString() == onnx.load(source).SerializeToString()
```

依赖安装：`python -m pip install -r onnx_graph/requirements.txt`。ONNX Runtime 仅数值验证脚本需要：`python -m pip install onnxruntime`。

## 图里保存了什么

| 位置 | 内容 |
| --- | --- |
| `graph.nodes[id]["proto"]` | 完整、可编辑的 `NodeProto`，包含所有 protobuf 字段、属性、子图属性、文档及元数据 |
| `name / op_type / domain / inputs / outputs / attributes / doc_string` | 节点的便捷查询字段；由 `proto` 派生 |
| `original_name / original_index` | 导入前的名称和原始序号；改 NetworkX 节点 ID 后仍用于对应源模型 |
| `hierarchy / scope` | 完整路径分段 tuple，以及父层级，例如 `MobilenetV3/expanded_conv/depthwise` |
| `graph.graph["hierarchy"]` | `层级路径 -> 该层级下所有节点 ID`，键不带首尾 `/` |
| `graph.graph["model"]` | 完整源 `ModelProto`，保留权重、稀疏权重、模型输入输出、value_info、opset、IR、函数、元数据等 |
| `graph.graph["rename_report"]` | 开启名称修复时，每个顶层节点的修复结果和候选路径 |
| 边属性 | `tensor`、`output_index`、`input_index`、`implicit` |

节点 ID 初始为从 0 开始的整数，避免空名称或重名节点互相覆盖。权重和模型输入是张量，不额外伪装成算子节点。同一对算子可能有多条边，例如 `Mul(a, a)` 的两个输入端口都会保留；这符合 [NetworkX MultiDiGraph](https://networkx.org/documentation/stable/reference/classes/multidigraph.html) 的多重有向边模型。

`NodeProto` 是编辑和导出的唯一依据。便捷字段和边会在查询前缀、导出时自动刷新；直接改边或 `data["name"]` 不会改 ONNX：

```python
from onnx_graph import refresh_graph

node_id = node_ids[0]
graph.nodes[node_id]["proto"].name = "MyModel/block/Conv"
# 修改连接时，更新 NodeProto.input；修改算子属性时，更新 NodeProto.attribute。
# graph.nodes[node_id]["proto"].input[0] = "已有张量名"
refresh_graph(graph)
```

也可以用 `graph.subgraph(node_ids)`、`remove_node`、`nx.relabel_nodes` 等 NetworkX 操作后导出。添加节点时提供完整 `proto`，不要给新节点复用已有节点的 `original_index`。改变算子的形状语义后，应同步修改相应类型记录；本工具不是自动图优化器。

`extract_prefix` 返回独立深拷贝；NetworkX 自带的视图或浅拷贝仍共享 protobuf 对象，编辑它们时遵循 NetworkX 的共享规则。

## 前缀与子图边界

- `MobilenetV3/expanded_conv_1` 匹配该节点名称本身及其 `/` 子层级，不匹配 `expanded_conv_10`。
- 前后带 `/` 的查询也可用；空前缀选择全部节点。没有匹配时 `extract_prefix` 报错。
- 只选择匹配前缀的算子，内部连接来自张量依赖。跨边界的普通张量成为子图输入，使用到的初始化权重随模型保存。前缀之外的 `Constant` 算子输出也作为边界输入，不自动扩大算子集合。
- 原模型输出、被未选中节点使用的输出、子图内没有消费者的输出会成为子图输出。中间结果同时供子图内外使用时也保留为输出。
- 初始化权重若原本同时是模型输入，会保留其可覆盖输入语义。子图保留需要的量化注释和参数。
- `If/Loop/Scan` 作为完整算子保留，内部 `GraphProto` 信息不丢失；闭包引用的外部张量也计入依赖。前缀选择与名称修复作用于顶层图，不切开控制流子图。
- 导出按依赖排序并默认运行 ONNX checker；环和重复张量生产者报错。

中间张量缺少类型时，先对完整源图做 [ONNX 形状推断](https://onnx.ai/onnx/api/shape_inference.html)。未知的类型/形状不会被默认为 float 或任意形状；自定义算子无法推断时，需要传入真实的边界信息：

```python
from onnx import helper, TensorProto

model = networkx_to_onnx(
    subgraph,
    value_info={
        "实际边界张量名": helper.make_tensor_value_info(
            "实际边界张量名", TensorProto.FLOAT, ["batch", 24, 56, 56]
        ),
    },
)
```

完整模型保留训练信息；带 `training_info` 的模型暂不支持截取，以免留下指向已移除节点的训练绑定。

## 按输入输出路径修复名称（旧 name 模式）

以下配置仅在 `HIERARCHY_SOURCE = "name"` 时生效。默认 outputs 分层保留原始名称，无需执行修复。

```python
graph = onnx_to_networkx("company.onnx", repair_names=True)
for record in graph.graph["rename_report"]:
    print(record)
onnx.save(networkx_to_onnx(graph), "company.fixed.onnx")
```

也可单独调用 `fixed_model, report = repair_node_names(model)`；原传入模型不受修改。

对你的示例：

```text
name:    mono_model_cfb_d3_LN2_Reshape
output:  /mono_model/cfb_d3/LN2/Reshape_output_0
结果:    /mono_model/cfb_d3/LN2/Reshape
```

`mono_model*cfb_d3*LN2*Reshape` 也能得到相同结果，`mono_model`、`cfb_d3` 中真实的下划线会保留。

也支持节点名中的 `_` 对应输出路径中的 `.`：

```text
name:    layers_0_Reshape_1
output:  /layers.0/Reshape_1_output_0
结果:    /layers.0/Reshape_1
```

匹配时将 `/`、`*`、`.` 统一按 `_` 比较，确定唯一候选后使用张量路径的原始拼写。因此 `layers.0` 的点号和 `Reshape_1` 的下划线都会保留。点号不增加 slash 层级：`layers.0` 对应 0 个 `/`，`layers.0/Reshape_1` 对应 1 个 `/`。多个路径扁平化后相同仍标记为 `ambiguous`，不会猜选。更新后保持 `REPAIR_NAMES = True`，并用 `GENERATE_STATS = True` 重新生成前缀统计。

### JSON 前缀别名与未匹配检查

如果名称前缀本身不同，例如 `spNetFC_1` 对应 `oSpatialNet_freq_cmps_1`，可以复制 `onnx_graph/name_aliases.example.json` 为 `onnx_graph/name_aliases.json`，填写：

~~~json
{
  "spNetFC_1": "oSpatialNet_freq_cmps_1"
}
~~~

在 `main.py` 中启用：

~~~python
HIERARCHY_SOURCE = "name"
REPAIR_NAMES = True
NAME_ALIASES_JSON = PACKAGE_DIR / "name_aliases.json"
GENERATE_STATS = True
~~~

这份配置由统计、导出和 profiling 名称匹配共用，运行期间只读取一次；默认 `None` 不启用别名。个人配置 `name_aliases.json` 已加入忽略规则，示例文件随代码提供。修改映射后需要重新生成统计，使 CSV 前缀与导出一致。

例如 `spNetFC_1_fconv1_1_conv_Conv` 与 `/oSpatialNet_freq_cmps_1/fconv1.1/conv/Conv_output_0` 会对应到节点名 `/oSpatialNet_freq_cmps_1/fconv1.1/conv/Conv`。别名只用于比较，仍需输入输出路径的证据，不会单凭配置强制改名。

规则是从名称开头匹配，最长前缀优先，一次替换；`spNetFC_1` 不匹配 `spNetFC_10`，不做任意位置的替换，也不递归应用别名链。原有名称能直接对应输出时优先使用原有对应关系。JSON 必须是字符串到字符串的对象；文件不存在、格式错误、空前缀、重复键或扁平化后冲突的键会报错。

应用别名和修复后，会逐一检查当前节点名与 outputs（去除 `_output_数字` 或 `:数字` 后）的对应关系。控制台打印**全部**未确认节点的原始 name、当前 name、outputs 和原因，包含已有 `/` 但输出不匹配的节点，以及通过 input 恢复名称后仍不能对应 output 的节点；空名称继续跳过。

诊断保存在：

- 自动统计：`STATS_DIR/name_output_mismatches.json`，同时写入 `stats.json` 的同名字段及 `stats.txt`。
- 批量导出：`OUTPUT_DIR/batch/name_output_mismatches.json`，同时写入 `batch_export.json`。
- 单前缀示例：`OUTPUT_DIR/name_output_mismatches.json`。

每条记录包含 `node_id`、`op_type`、`original_name`、`name`、`outputs`、`reason`、`rename_status`。原因包括 `no_output_match`（没有对应输出）、`ambiguous_outputs`（多个候选路径）、`ambiguous_name`（名称恢复有歧义）、`name_collision`（改名冲突）和 `missing_outputs`（没有输出）。未发现问题时文件内容为 `[]`；仅复用统计且没有选中任何子图时，导出阶段不会加载模型，诊断文件为 `null`，控制台会说明未检查。

节点名和张量名本来就不要求相同，因此这些记录表示无法确认命名对应关系，不代表 ONNX 连接有错误。相关节点仍保留，前缀分组以当前名称为准。

独立命令和 Python API 也支持：

~~~powershell
python -m onnx_graph.stats model.onnx --hierarchy-source name --name-aliases onnx_graph/name_aliases.json
python -m onnx_graph repair-names model.onnx fixed.onnx --name-aliases onnx_graph/name_aliases.json
~~~

~~~python
graph = onnx_to_networkx("model.onnx", repair_names=True, name_aliases="name_aliases.json")
issues = graph.graph["name_output_mismatches"]
~~~

`name_aliases` 也可直接传入字典；`repair_node_names`、`analyze_subgraphs`、`export_from_stats`、`omc_to_networkx` 使用同一个参数名。OMC 只修复显示名称并检查描述符名称，原始依赖和二进制信息保留。名称修复 CLI 在输出 JSON 中列出 `name_output_mismatches`；其改名报告也包含每个节点的 outputs。

### 通用修复规则

1. 空名称（包括字段缺省、空字符串、纯空白）直接跳过名称修复，报告标记为 `skipped_unnamed`。
2. 对已有名称，优先使用 OUTPUTS，去除末尾 `_output_数字` 或 `:数字`，再验证路径扁平化后是否对应原名。
3. OUTPUTS 无法直接匹配时，从 INPUTS/OUTPUTS 的父路径恢复层级；仅当剩余尾部是当前 `op_type` 或带数字后缀的算子名时采用。不会把上游算子的名称直接用作当前节点名。
4. 已有 `/` 层级的名称保留，避免改变融合算子本来正确的名称；多个候选、名称冲突或缺少证据时保留原名。
5. 只改 `NodeProto.name`，不改输入输出张量名及连接。报告包含 `renamed / already_hierarchical / unresolved / ambiguous / collision / skipped_unnamed`。

输入输出附近的无名算子会保留在 NetworkX 中，以整数 ID 标识；按具体层级前缀选择时忽略它们，空前缀仍代表完整图。即使设置 `repair_names=True`，也不会为这些节点猜测或生成名称。截取中间层时，与无名算子相连的张量仍用于确定子图输入输出。

这里跳过的是名称处理，计算节点及其连接仍完整保存，因此模型可以正常往返。ONNX 的 `NodeProto.name` 是可选字段；`model.graph.input/output` 则是独立的张量信息，依赖其张量名称建立连接，详见 [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html#nodes)。

整张图的 `GraphProto.name` 是另一个字段，ONNX checker 要求它非空。导出及 `repair_node_names` 会把缺省、空字符串或纯空白的主图名补为 `onnx_graph`；主图中控制流子图的空图名也会按位置生成。已有图名和空算子名保持原状。普通加载仍保留源模型的原始字段，修复发生在独立的输出副本中；因此源模型缺图名时，导出结果会多出必要的图名，不能要求与原始无效模型逐字节相同。`main.py` 已使用模型校验处理这类情况。

公司内部模型尚未提供，因此这些规则用你给出的样例及合成用例验证；实际命名规则超出上述范围时，可根据报告扩展匹配规则。

## 命令行

```powershell
# ONNX -> NetworkX pickle
python -m onnx_graph convert onnx_graph/models/mobilenetv3/mobilenetv3-large_224_1.0.onnx onnx_graph/models/mobilenetv3/networkx/mobilenetv3.nx.pkl

# NetworkX -> ONNX
python -m onnx_graph restore onnx_graph/models/mobilenetv3/networkx/mobilenetv3.nx.pkl onnx_graph/models/mobilenetv3/networkx/mobilenetv3.roundtrip.onnx

# 按前缀提取
python -m onnx_graph extract onnx_graph/models/mobilenetv3/mobilenetv3-large_224_1.0.onnx onnx_graph/models/mobilenetv3/networkx/expanded_conv_1.onnx --prefix MobilenetV3/expanded_conv_1

# 公司模型名称修复，自动生成 company.fixed.rename.json
python -m onnx_graph repair-names company.onnx company.fixed.onnx

# 名称修复与前缀提取组合使用
python -m onnx_graph extract company.onnx company.block.onnx --repair-names --prefix /mono_model/cfb_d3/LN2
```

`.nx.pkl` 保存完整 NetworkX 对象及 protobuf，可用 `pickle.load` 读回。只加载可信 pickle；此格式可能执行 Python 代码。GraphML/普通 JSON 不能直接无损保存这些 protobuf 对象。

子图的 NetworkX 对象仍带完整源模型以确定边界和推断类型，因此子图 pickle 仍包含源模型权重；导出的子图 ONNX 只带需要的权重。加载路径时会读取外部权重文件，并在导出时默认内嵌；超过 protobuf 大小上限的超大模型需另做外部数据保存适配。

## 按层级统计子图、算子和计算量

新增 `stats.py`，可直接运行，也可用 `python -m onnx_graph.stats`。只需现有的 ONNX、NetworkX 依赖。直接运行时修改脚本开头的配置：

```python
MODEL_PATH = Path(r"D:/models/company.onnx")
HIERARCHY_SOURCE = "outputs"
REPAIR_NAMES = True  # 仅 name 模式生效
SLASH_COUNTS = [1, 2]  # 只看一个层级可改成 [1] 或 [2]；支持其他非负整数
BATCH_SIZE = 1
INPUT_SHAPES = {}  # 例如 {"input:0": [1, 224, 224, 3]}
PROFILE_CSV = None  # 可选：Path(r"D:/models/_20261016_110720_op.csv")
OUTPUT_DIR = None  # 默认在模型旁的 networkx/<模型名>_stats/ 保存报告
```

`SLASH_COUNTS` 指**分组前缀**里的 `/` 数量，去掉前导、末尾和重复 `/` 后计数。例如：

| 设置 | 分组前缀示例 | 分组包含的节点 |
| --- | --- | --- |
| `[1]` | `MobilenetV3/expanded_conv_1` | 该模块下的所有节点 |
| `[2]` | `MobilenetV3/expanded_conv_1/depthwise` | depthwise 下的所有节点 |
| `[1, 2]` | 同时列出以上两种层级 | 各层级独立统计，不把两个层级的计算量相加 |

每个节点在同一层级只归入一个前缀；`expanded_conv_1` 不会匹配 `expanded_conv_10`。前缀恰好是完整层级路径时，它也构成一个子图。无可用层级或路径不足该深度的节点不参与该层级分组，报告通过 unscoped_node_ids 和 shallower_node_ids 单独列出其 ID；它们仍参与主图总量统计。每组对应原 NetworkX 图中这些节点诱导出的子图，计算量为其节点计算量之和。

在仓库根目录运行：

```powershell
# 按 stats.py 的配置运行（默认 handfree_ainr，两个层级；独立统计默认不加载 profiling）
python onnx_graph/stats.py

# 公司模型：按输出路径分层，只统计前缀有两个 / 的子图
python -m onnx_graph.stats D:/models/company.onnx --hierarchy-source outputs --slashes 2

# 指定真实输入尺寸，避免猜测动态维度；多输入可以重复 --input-shape
python -m onnx_graph.stats D:/models/company.onnx --slashes 1 2 --input-shape "input:0=1,224,224,3"

# 按输出分层并关闭 batch 假设，保留未知维度
python -m onnx_graph.stats D:/models/company.onnx --hierarchy-source outputs --no-batch-assumption
```

输出包括控制台和 `stats.txt` 中的**全部子图前缀、节点数、算子类型及数量、MACs、FLOPs**；`stats.json` 还包含每组节点 ID、逐节点形状/计算公式/未知原因、名称修复记录和输入假设。原始 ONNX 不会被修改。

每次运行还会按 `SLASH_COUNTS` **分别生成 CSV**，例如 `[1, 2]` 生成 `stats_1_slash.csv` 和 `stats_2_slash.csv`，保存在同一个输出目录。CSV 每行一个子图，第一行就是字段名，不混入说明或合计行：

```text
子图名字,节点数,Add数量,AveragePool数量,Clip数量,Conv数量,...,MACs,FLOPs,MFLOPs,计算量完整,未知计算量节点数,已知MACs,已知FLOPs
```

算子列根据模型实际包含的类型自动生成，各层级列名及顺序一致；某子图没有的算子填 `0`。`MACs/FLOPs` 是未缩放的数值，`MFLOPs = FLOPs / 1,000,000`，可直接求和或排序。计算量不完整时，完整总量三列留空，`计算量完整` 为“否”，并保留已知部分及未知节点数量。名称修复、输入尺寸和计算口径记录仍在配套 TXT/JSON 中。无可用层级或层级较浅的未分组节点不出现在 CSV 中，因此合计对应当前层级覆盖的节点。

CSV 使用 UTF-8 BOM 编码，方便 Excel 识别中文；逗号、引号、换行由标准 CSV 引号规则处理。无需新增 Python 依赖。只修改 `SLASH_COUNTS` 或使用 `--slashes 2` 即可控制本次导出的层级。

### 可选的运行时间及“算术强度”

设置 `PROFILE_CSV` 为当前模型的 `_op.csv` 路径，或使用 `--profile-csv`，即可为每个层级的 CSV 增加以下列：

| 字段 | 含义 |
| --- | --- |
| `exec_time` | 子图所有节点的平均算子时间之和，单位微秒；缺少任一节点的时间则留空 |
| `exec_time_unit` | `us` |
| `算术强度` | 按用户定义计算 `FLOPs / exec_time`；任一值未知或时间为 0 时留空 |
| `算术强度单位` | `FLOPs/us`；数值除以 1000 即 GFLOP/s |
| `已知exec_time` | 已匹配节点的时间合计，便于查看不完整子图的已知部分 |
| `时间匹配节点数` / `时间未匹配节点数` | 当前子图的时间覆盖情况 |

这里按要求保留“算术强度”这个字段名，其口径表示**计算吞吐率**，通常的算术强度按 FLOPs/Byte 定义，参见 [NVIDIA 的性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understand-perf)。该时间是算子耗时之和，不等于包含并行、调度及传输的子图端到端延迟。

CSV 应为 UTF-8（允许 BOM），包含 `op name` 和 `total time(us)` 列；表头及字段前后空白自动去除，时间必须是有限非负数。也支持已有 `onnx_node_name` 列的标注文件，此列非空时优先用于匹配；空时间视为缺失。多个模型混在同一文件时会报错，需要先筛选出当前模型。

匹配先使用 ONNX 修复前/后的完整节点名称，再尝试去掉运行时名称末尾的 `(Convolution)`、`(Activation)` 等类型标记，最后使用 `/`、`*`、`.`、`_` 扁平化后的唯一候选。名称冲突、多个候选或没有对应节点时不分配时间，不按行号或 taskid 猜测。编译器插入的 `TransData`、无法确定归属的融合算子以及没有独立记录的 ONNX 节点，会在 JSON 的 `profile.entries` / `profile.unmatched_node_ids` 中列出。

**多轮采样先取平均，再对子图求和**，避免把 10 轮推理误计为一轮。同一 ONNX 节点对应多个不同名称的运行算子时，将各运行算子的均值相加。默认保留包括首轮在内的全部有效记录，各算子的采样次数写入 JSON。原始 CSV 不含输入尺寸校验信息，使用者应保证 `INPUT_SHAPES`、batch 与实际 profiling 一致。

```powershell
# 使用当前 MobileNetV3 的 10 轮 profiling，两个层级仍分别输出 CSV
python -m onnx_graph.stats onnx_graph/models/mobilenetv3/mobilenetv3-large_224_1.0.onnx --profile-csv onnx_graph/models/mobilenetv3/omc/profiling/_20261016_110720_op.csv --slashes 1 2

# 忽略脚本中配置的 PROFILE_CSV，输出原有不带时间/算术强度列的报告
python -m onnx_graph.stats --no-profile
```

在代码中调用时使用 `analyze_subgraphs(..., profile_csv="实际模型的_op.csv")`。`PROFILE_CSV = None` 或不传入 `profile_csv` 即禁用；不增加额外依赖。开启后 JSON 中的子图记录还包含 `exec_time`、`known_exec_time`、`arithmetic_intensity`、缺少时间的节点 ID 等，逐节点记录可追溯到 `profile.entries`。

当前样例验证：1960 行记录包含 196 个运行算子，每个 10 次；192 个能匹配 ONNX，4 个编译器插入算子未匹配，49 个 ONNX 节点没有独立时间。已匹配时间合计 **4094.9 us**。1 个 `/` 的分组中 5/21 个、2 个 `/` 的分组中 43/76 个具有完整时间，其余保留已知部分。带时间的演示报告保存在 `onnx_graph/models/mobilenetv3/networkx/mobilenetv3-large_224_1.0_stats/with_profile_20261016_110720/`。

### 计算量估算口径

- Conv：`MACs = 输出元素数 × (输入通道数/group) × 卷积核体积`，支持分组和 depthwise 卷积。权重的第二维已包含 `/group`，不重复相除。FLOPs 为 `2 × MACs + 偏置加法`，依据 [ONNX Conv 的权重布局](https://onnx.ai/onnx/operators/onnx__Conv.html)。
- MatMul/Gemm：`MACs = 输出元素数 × K`，支持批量广播、向量乘法、转置；Gemm 另计 alpha/beta 缩放及偏置。语义见 [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) 和 [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html)。
- 支持常见逐元素算子、激活、池化、Softmax、归约及 BN/LN/InstanceNorm。比较、exp/log/sqrt 等按一次标量操作计入近似 FLOPs，归一化采用未融合公式；逐节点公式写入 JSON。池化按完整窗口估算，padding 边界可能高估。
- Reshape、Transpose、Shape、Gather 等形状/索引/数据搬移算子计 0 FLOPs，但仍可能有显著内存开销。该报告不能直接换算成实际耗时或功耗。
- 只对主图统计；自定义、控制流、量化或未支持的算子，以及缺少必要形状的节点，都标为未知。该组完整 `macs/flops` 为 `null`，`known_macs/known_flops` 是已知部分；文本显示 `+ ?`，不会伪装成完整总量。

脚本默认把普通输入的**未知第 0 维**视为 batch=1，并记录这一假设；已有的固定维度不会被改动。如果你的输入第一维不是 batch，请设置 `BATCH_SIZE = None` 或加 `--no-batch-assumption`，再通过 `INPUT_SHAPES`/`--input-shape` 指定形状。其他未知维度不会自动取 1。改变输入尺寸后会清除缓存的中间形状并重新推断，未能推断的仍标为未知。导入调用 `analyze_subgraphs` 时默认不假设 batch：

```python
from onnx_graph.stats import analyze_subgraphs

report = analyze_subgraphs(
    "company.onnx", slash_counts=[1, 2], repair_names=True,
    input_shapes={"实际输入张量名": [1, 3, 224, 224]},
)
for level in report["levels"]:
    for subgraph in level["subgraphs"]:
        print(subgraph["prefix"], subgraph["operator_counts"], subgraph["flops"])
```

给定 MobileNetV3 在 batch=1、输入 `[1,224,224,3]` 下：1 个 `/` 的前缀共 **21** 组，2 个 `/` 共 **76** 组，两种分组都覆盖全部 **241** 个节点；估算 **216,591,040 MACs / 449,148,684 FLOPs**，无未知计算量。`MobilenetV3/expanded_conv_1` 包含 `Conv×3, Relu×2`，为 **19,468,288 MACs / 41,018,880 FLOPs**。完整报告位于 `onnx_graph/models/mobilenetv3/networkx/mobilenetv3-large_224_1.0_stats/`。

## 已验证的结果与复现

### OMC 输入（实验支持）

`stats.py` 已支持将 `MODEL_PATH` 改为 `.omc`，继续按层级生成 CSV、JSON 和 TXT，并可选匹配 profiling 时间。当前验证的是 DDK 6.1.1.0 生成的未加密 IMOD 文件（256 字节文件头、版本 `0x10000000`）。其他版本、加密文件或没有可识别图定义的文件会报错，不会按字符串猜测计算图。

```python
# 以下为按 name 分层的历史 OMC 验证配置
HIERARCHY_SOURCE = "name"
MODEL_PATH = PACKAGE_DIR / "models/mobilenetv3/omc/mobilenetv3-large_224_1.0.omc"
PROFILE_CSV = PACKAGE_DIR / "models/mobilenetv3/omc/profiling/_20261016_110720_op.csv"
REPAIR_NAMES = True
SLASH_COUNTS = [1, 2]
GE_PROTO_PATH = None
```

也可以不修改默认配置，在仓库根目录运行：

```powershell
python -m onnx_graph.stats onnx_graph/models/mobilenetv3/omc/mobilenetv3-large_224_1.0.omc --hierarchy-source name --slashes 1 2 --profile-csv onnx_graph/models/mobilenetv3/omc/profiling/_20261016_110720_op.csv

# 单独保存完整 NetworkX 图；也支持 --ge-proto 和 --repair-names
python -m onnx_graph convert onnx_graph/models/mobilenetv3/omc/mobilenetv3-large_224_1.0.omc onnx_graph/models/mobilenetv3/omc/networkx/mobilenetv3.omc.nx.pkl
```

OMC 解析需要 DDK 的 `tools/tools_ascendc/package/python/opc_tool/ge_ir_pb2.py`。当前仓库会自动找到已有文件；在其他电脑上设置 `GE_PROTO_PATH`、环境变量 `ONNX_GRAPH_GE_PROTO`，或传 `--ge-proto D:/DDK/tools/tools_ascendc/package/python/opc_tool/ge_ir_pb2.py`。也可以设置指向 DDK 根目录的 `DDK_PATH`。安装本目录的 `requirements.txt` 即可满足 Python 依赖；Windows 上不需要运行 Linux 编译器。解析器仅读取这个文件中的 protobuf descriptor 字节，不执行 SDK Python 代码，也不随本仓库复制 SDK schema。

OMC 使用编译时的输入形状，`BATCH_SIZE` 不改变它；`INPUT_SHAPES`/`--input-shape` 非空时会明确报错，避免用不对应的形状估算。默认报告目录为模型旁的 `networkx/<模型名>_omc_stats/`，与 ONNX 报告区分。

**NetworkX 与信息保留：**

```python
from onnx_graph import omc_to_networkx, nodes_by_prefix, extract_prefix

graph = omc_to_networkx("model.omc", repair_names=True, ge_proto_path="path/to/ge_ir_pb2.py")
ids = nodes_by_prefix(graph, "MobilenetV3/expanded_conv_1")
subgraph = extract_prefix(graph, "MobilenetV3/expanded_conv_1")
for node_id, data in subgraph.nodes(data=True):
    print(node_id, data["name"], data["op_type"], data["input_tensors"], data["output_tensors"])
```

- 递归解析 `GraphOp.attr["subgraph"]` 中的 GraphDef。NetworkX 保留外层、内层的全部节点，包括常量和输入输出节点；节点 ID 跨图唯一。
- 每条边保留输入/输出端口、张量引用和控制依赖。不同图的同名节点不会串联。**各图保留自己的依赖边**，`graph.graph["regions"]` 的 `parent_node_id`、`inputs`、`outputs` 描述 GraphOp 包含关系及边界；这不是把所有 GraphOp 内联后的单层计算图。
- 节点 `proto` 是完整 GE OpDef 的序列化字节，`attributes` 提供可读属性，`input_tensors/output_tensors` 提供形状、类型和布局；它不是 ONNX NodeProto。`constant_tensor` 记录常量内部的逻辑张量描述，避免误用编译后的 `[1]` 占位输出形状。
- 图级 `source_bytes` 保留完整原始 OMC，包括权重、编译任务和未知字段；`model_proto`、`schema_bytes`、`regions[*].proto` 保留模型与子图定义。可以使用标准 `pickle.dump/load` 保存完整 NetworkX 对象。提取子图后仍带完整源文件作为元数据，pickle 不会因此缩减为独立可执行模型。
- 在 `HIERARCHY_SOURCE="name"` 的旧模式下，`REPAIR_NAMES` 只根据可确认的输入输出描述名称修复显示名称；原始 OpDef 和依赖引用不变。没有名称的节点保留并跳过名称修复、层级分组。统计时另外排除 `Const/Data/NetOutput` 和已展开的 `GraphOp`，避免把常量和容器计为运行算子。

**计算量与验证结果：**

目前估算 GE `Convolution`（含分组、已验证的 depthwise 布局和融合 ReLU）、`Add/Sub/Mul/Div/RealDiv`、常见 `Activation` 模式、`Eltwise`、`PoolingD`、`Softmax`；数据搬移/形状变换算子按 0 FLOPs 计。使用逻辑尺寸，未计入物理通道补齐或内核开销。`GemmV3`、`BatchMatMulV2`、`LayerNorm` 等尚未实现的 GE 算子会保留统计和时间，计算量标为未知，不能套用同名 ONNX 公式。

当前 MobileNetV3 OMC 的实际结果：

| 项目 | 结果 |
| --- | --- |
| 外层 + GraphOp 内部节点 | 4 + 384 = 388（包含常量、容器、输入输出） |
| 纳入计算统计的节点 | 197；其中 196 个匹配 profiling，外层 `graphop_data_cast_0` 没有独立记录 |
| 1 个 `/` 的分组 | 21 组，覆盖 194 个运算节点 |
| 2 个 `/` 的分组 | 65 组，覆盖 194 个运算节点 |
| 分组完整性 | 21/21、65/65 组均有完整 FLOPs 和时间 |
| 逻辑计算量 | 216,591,040 MACs / 448,725,324 FLOPs |
| 已匹配时间合计 | 4179.7 us；整图完整 `exec_time` 和“算术强度”因一个节点缺时间而留空 |
| `MobilenetV3/expanded_conv_1` | `Convolution×3`（两个 ReLU 已融合），41,018,880 FLOPs，115 us，356,685.913043 FLOPs/us |

64 个卷积的 MACs 已与原始 ONNX 逐节点核对一致。编译后算子融合、消除和插入会改变算子数、层级分组数及 FLOPs，因此 OMC 不必与 ONNX 的 76 个二级分组相同。没有足够层级名称的 3 个运行节点仍保留在总量及 JSON 中。另验证了本仓库 `crossband/fullband/mhsa/spatialnet_layer/tconvffn` 五个 OMC 均能解析和统计；其中未支持的计算公式保持未知。

**能力边界：** OMC→NetworkX、层级查询、诱导子图、统计和时间关联已支持；`networkx_to_onnx` 会拒绝 OMC 图。当前不支持把 OMC 子图导出为可执行 ONNX/OMC，也不支持编辑 GE OpDef 后调用 `refresh_graph`。这类可执行子模型应从原始 ONNX 提取并重新编译。华为将 OMC 定义为硬件强相关的编译模型，参见[官方 OMC 转换说明](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-model-conversion-example)；文件分区结构可参考 [Netron 的 IMOD 读取实现](https://github.com/lutzroeder/netron/blob/main/source/om.js)。官方也提供 `omg --mode 1 --om model.omc --json model.json` 的离线模型查看路径，参见 [OMG 参数](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-overall-parameter)；本实现不依赖该命令。

## 导出子图的相似度与去重

使用 `similarity.py` 读取已导出的 ONNX，生成分组和代表子图清单，用于减少性能测试数量。**连接方式参与比较**，节点名字、层级前缀和普通元数据不参与。原始 ONNX 不修改、不删除；后续测试可读取 `representatives.csv` 中的 `ONNX文件` 列。

修改 `similarity.py` 顶部配置后直接运行：

```python
INPUT_DIR = Path(r"D:\你的目录\batch")  # 包含 slash_1、slash_2 等子目录
OUTPUT_DIR = None                     # 自动生成 grouping_exact 或 grouping_similar
GROUP_MODE = "exact"                  # 默认：配置相同才合并
SIMILARITY_THRESHOLD = 0.90           # 仅影响 similar 模式
SEPARATE_SLASH_COUNTS = True          # 各 slash 层级分别分组
```

```powershell
# 在仓库根目录执行；也可直接 python onnx_graph/similarity.py
python -m onnx_graph.similarity "D:\你的目录\batch"

# 可选：按相似度合并；阈值越低，分组越宽松
python -m onnx_graph.similarity "D:\你的目录\batch" --mode similar --threshold 0.9

# 可选：允许跨 slash 层级比较，使用独立输出目录
python -m onnx_graph.similarity "D:\你的目录\batch" --merge-levels --output-dir "D:\你的目录\grouping_all"
```

不需要新增依赖。`REPAIR_NAMES` 不影响这一步，比较依据来自 ONNX 的张量依赖。若输入目录保留 `batch_export.json`，报告会恢复原来的子图前缀；通过层级和文件名关联，清单中的旧电脑绝对路径不影响读取。没有清单时使用 ONNX 文件名。没有 `slash_<数字>` 目录的文件放入 `unclassified` 组。

比较内容：

- **连接关系**：构建“算子—张量”有向多重图，保留输入/输出端口、分支汇合、重复使用同一张量、共享权重及子图输入输出边界。因此 `Conv→Relu→Add` 与 `Conv→Add→Relu`、串联与并联不会因算子数量相同而被判为配置相同。独立分支在 ONNX 文件中的排列顺序不影响结果。
- **计算配置**：算子类型/domain/opset、属性、每个张量的 dtype 和形状。属性顺序不影响比较，但显式属性与省略默认属性等不同写法可能保守地分开。
- **权重值**：忽略直接作为 Conv/ConvTranspose、Gemm/MatMul、BN/InstanceNorm/LayerNorm、PRelu、Add/Sub/Mul/Div 数据参数使用的浮点/复数常量值，保留其形状和类型。其他常量保留值，特别是形状、索引、Resize scales、Clip 阈值等；还会保留为这些控制参数提供数据的上游常量。整数/量化权重、经过转换才送入算子的权重等保守地保留值。这一规则优先避免误合并。

每对图提供两个 0～1 分数：**结构相似度**只使用算子类型、边界角色及连接关系；**配置相似度**进一步纳入形状、类型、属性和需保留的常量值。实现采用带方向和端口的 [WL 图特征](https://www.jmlr.org/beta/papers/v12/shervashidze11a.html)，默认聚合 0、1、2 轮邻域，随后计算特征计数的加权 Jaccard：

```text
相似度 = Σ min(特征A计数, 特征B计数) / Σ max(特征A计数, 特征B计数)
```

这是相似度指标，不是统计学的相关系数。相似分组使用配置相似度。没有把 FLOPs 或 exec_time 混入分数，它们不能证明连接和配置相同。

- `exact`：先用 WL 特征筛候选，再以 [VF2 图同构](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isomorphism.is_isomorphic.html)核对全部配置标签、方向、端口和多重边；WL 分数为 1 本身不能证明同构。默认推荐这一模式选择性能测试代表。
- `similar`：使用 [complete-link 聚类](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)的规则，合并两组时取所有跨组图对的最低相似度。因此每组**任意两个图**均达到阈值，避免 A≈B、B≈C 就将 A 与差异较大的 C 强行合并。实现使用现有依赖及标准库，不要求安装 SciPy。每组选择与组内其他图平均相似度最高的真实 ONNX 作代表，平分时按文件顺序选择。

输出示例：

```text
batch/grouping_exact/
  grouping.json
  failures.csv
  slash_1/
    groups.csv
    representatives.csv
    similarity.csv
  slash_2/
    groups.csv
    representatives.csv
    similarity.csv
```

`groups.csv` 每行一个子图，包含分组编号、代表 ONNX、与代表相似度、组内最低相似度、精确配置组、节点数及算子数量等；编号只在当前层级内唯一。`representatives.csv` 每组一行。`similarity.csv` 列出同层级所有两两组合（不重复 A/B 和 B/A，不列自身），同时标明是否配置相同。`grouping.json` 记录本次配置、全部有效分组及失败文件；CSV 为 UTF-8 BOM，方便 Excel 打开。重复运行会更新报告，改变输入集合或分组设置时建议另设输出目录，避免读取其他运行遗留的层级文件。

动态维度保留原符号并提示，使用代表模型时必须保持相同实际尺寸；缺少类型或形状信息的模型不自动合并。控制流子图、局部函数、稀疏权重/属性、量化注解目前会明确记录到 `failures.csv`，这些模型仍须单独测试，不能从测试清单中直接丢弃。自定义算子若缺少中间张量形状，也会保留为独立组。两两比较的计算量及分数矩阵内存随子图数量平方增长，宜先用统计条件筛选，再比较导出的子图。

本机 MobileNetV3 已导出的 97 个 ONNX：`exact` 模式得到 `slash_1: 21→20`、`slash_2: 76→70`，共 90 个代表；例如 `expanded_conv_8` 与 `expanded_conv_9` 配置相同。`similar` 阈值 0.9 的结果相同，未额外合并。全部成功解析，模型中的动态 batch 会在报告中提示。这是结构和配置验证结果，尚未在设备上验证代表模型是否覆盖同组其他模型的实际延迟；编译优化、权重稀疏性和数据相关算子仍可能影响性能。

### 测试

```powershell
python -m unittest discover -s onnx_graph/tests -v

# 可选数值验证：先安装 onnxruntime，并将 MobileNetV3 模型放到
# onnx_graph/models/mobilenetv3/mobilenetv3-large_224_1.0.onnx
python -m onnx_graph.verify_mobilenet
```

135 个测试通过。新增 9 个 outputs 分层测试覆盖 slash=0 平面输出分组/多输出/时间均值及导出、原名保留、逐字节往返、子图数值验证、编辑后层级刷新、多输出共同父层级、缺失/冲突输出、原名时间匹配、空名及重名、完整主流程和诊断、旧 CSV/旧 ONNX 防混用及 CLI；新增 1 个 OMC 测试覆盖输出描述符分层及原名/依赖保留。新增 10 个别名测试覆盖 JSON 校验、最长前缀及边界、原始模型保留、输出与 input 证据、歧义及重名、缺少输出、profiling 别名匹配、完整流程及全部未匹配节点的打印和保存、CLI 参数；其中 10 个完整流程测试覆盖统计/时间/联合筛选/导出去重及总表衔接、修复名称共用、排除遗留层级 CSV、复用已有统计、统计失败停止、形状假设和可选 profiling、总表零值/部分覆盖、OMC 仅统计分支、自定义 domain 算子计数及指定 CSV 校验；8 个导出去重联动测试覆盖默认开启/关闭、代表清单与最终文件一致、重复运行、只处理本次筛选结果、空选择、解析失败保留、分组/删除失败、越界及文件变化检查、跨层级相似分组；14 个相似度测试覆盖连接顺序、串并联、端口、共享张量、独立分支排列、忽略名称和普通浮点权重、类型/形状/属性差异、控制常量及其上游值、外部权重、WL 特征相同但不同构、未知/动态形状、聚类链式误合并、分层报告与清单迁移、相似模式及失败记录；14 个批量导出测试覆盖严格阈值、AND/OR 联合条件、未知值、分层批量导出与数值结果、前缀边界、名称修复、重复/已有文件保留、文件名冲突、空筛选结果、空 CSV、控制台及清单中的过滤/失败原因、CSV 校验、统计与模型不一致、缺少边界类型时继续处理其他子图；32 个转换测试覆盖完整字段往返、空图名补齐、空图名模型运行 main.py、控制流子图空图名、重复与空名称、无名输入输出算子的跳过及边界保留、多输入/输出、多重边、前缀边界、名称修复歧义、控制流外部依赖、外部与稀疏权重、局部函数、量化元数据等；14 个统计测试覆盖层级分组、修复开关、分组/depthwise 卷积、矩阵乘法与广播、池化/归一化、不同 opset 的 Softmax、动态形状、未知算子、分层 CSV 的算子列/未知值/中文及特殊字符、空层级和报告输出；11 个运行时间测试覆盖多轮均值、多个运行算子映射、修复前后名称、歧义、不完整时间、零值及吞吐率、CSV 数据校验、可选开关和导出；13 个 OMC 测试覆盖嵌套图、区域隔离、多重/控制边、原始信息及 pickle 保留、CLI 转换、空名称和修复、32/64 位分区表、畸形文件、schema 不执行、分层 CSV 和 depthwise/融合计算量。OMC 集成测试需要 DDK GE schema，未提供时会明确跳过；ONNX 测试不依赖 DDK。

给定模型结果：241 个节点、279 条依赖边、144 个初始化张量。经过 pickle 保存/加载和 ONNX 恢复后，与原始文件**逐字节一致**。

`MobilenetV3/expanded_conv_1`：5 个节点、4 条内部边、6 个初始化张量。输入为 `MobilenetV3/expanded_conv/add:0`，形状 `[batch, 16, 112, 112]`；输出为 `MobilenetV3/expanded_conv_1/project/BatchNorm/FusedBatchNorm:0`，形状 `[batch, 24, 56, 56]`。

使用固定种子生成的一组输入，在 ONNX Runtime CPU 上关闭图优化，对比完整模型的中间激活与子图结果，最大绝对误差为 **0.0**。这是该测试输入的验证结果。

生成的模型、NetworkX pickle、名称修复样例与 `validation.json` 位于 `onnx_graph/models/mobilenetv3/networkx/`；原始 ONNX 保持不变。
