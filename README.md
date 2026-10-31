# SPGA

SPGA 是一个面向版图拓扑生成的实验项目，使用离散扩散模型（D3PM）和 Diffusion Transformer（DiT）学习版图结构，并生成新的二值版图拓扑。

项目同时提供数据生成、数据格式转换、版图约束生成和 ICCAD16 数据处理工具。

## 主要功能

- 使用 Sparse DiT 建模版图拓扑
- 使用离散扩散过程生成二值版图
- 使用数据驱动的空间先验提升生成结构的合理性
- 将版图折叠为 `16 × 32 × 32` 的模型输入张量
- 支持 ICCAD16 数据和虚拟数据
- 生成版图的几何约束与合法性信息

## 项目结构

```text
SPGA/
├── data/
│   ├── fake/              # 虚拟数据
│   ├── iccad16/           # ICCAD16 数据
│   ├── data.csv           # 原始 CSV 数据
│   └── new_data.csv       # 处理后的 CSV 数据
├── src/
│   ├── cdnsgen.py         # DiT、D3PM、训练和生成逻辑
│   ├── test.py            # 加载模型并生成版图
│   ├── iccad.py           # ICCAD 数据加载和张量折叠
│   ├── fakegen.py         # 生成虚拟版图数据
│   ├── csv2msgpack.py     # CSV 转 MsgPack
│   ├── filter.py          # 生成版图约束
│   ├── euv.py             # EUV 数据处理工具
│   └── util.py            # 数据和版图辅助函数
└── README.md
```

## 环境安装

建议使用 Python 3.10 或更高版本，并创建独立虚拟环境：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux/macOS：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install numpy pandas scipy matplotlib pillow msgpack progress scikit-learn torch
```

如果使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch。

## 数据准备

### 生成虚拟数据

```bash
python src/fakegen.py data/fake/train.msgpack --count 10000 --min-size 3 --max-size 10 --seed 42
```

### 将 CSV 转换为 MsgPack

```bash
python src/csv2msgpack.py \
    data/data.csv \
    data/iccad16/train.msgpack \
    data/iccad16/test.msgpack \
    --train-count 8001
```

Windows 命令行也可以写成一行：

```bash
python src/csv2msgpack.py data/data.csv data/iccad16/train.msgpack data/iccad16/test.msgpack --train-count 8001
```

### 生成版图约束

```bash
python src/filter.py \
    data/iccad16/train.msgpack \
    data/iccad16/train_with_constraints.msgpack \
    --workers 4
```

## 模型训练

训练入口位于 `src/cdnsgen.py` 中的 `main()` 函数。运行前请检查并修改以下配置：

- 训练数据路径
- 模型输出路径
- `batch_size`
- `max_iter`
- GPU 编号

当前代码中部分路径是作者本机路径，例如：

```text
D:\wyz\deepattern-master\...
```

使用前需要替换为本项目所在目录下的实际路径，并确保训练数据目录名称与代码保持一致。

完成配置后运行：

```bash
python src/test.py
```

> 注意：当前 `src/test.py` 用于大规模生成，训练逻辑位于 `src/test.py` 引用的 `DiffusionModel` 中。建议根据实验需求在 `src/cdnsgen.py` 中调用 `detector.run(data, resume=False)` 开始训练。

## 版图生成

训练完成并保存模型后，可以运行：

```bash
python src/test.py
```

生成结果默认保存在模型目录下的 `test/` 子目录中，文件格式为 `.pkl`。

生成过程会：

1. 加载模型权重；
2. 加载空间稳态先验；
3. 通过离散扩散反向过程采样版图；
4. 将生成结果还原为版图拓扑；
5. 保存拓扑和合法性相关信息。

## 数据格式

每条版图记录通常包含以下字段：

- `cX`：版图宽度
- `cY`：版图高度
- `topoSig`：按行展开的二值拓扑字符串
- `xDelta`：横向几何尺寸信息
- `yDelta`：纵向几何尺寸信息
- `valid`：版图是否合法

## 实验结果

本项目代码和数据主要用于版图拓扑生成实验。具体的生成质量、合法率、训练损失和对比实验结果，请根据实际实验补充。

建议记录以下指标：

- 训练损失
- 生成版图数量
- 合法版图比例
- 与测试集的拓扑相似度
- 不同先验强度下的生成效果

## 注意事项

1. 当前项目没有统一的 `requirements.txt`，依赖需要手动安装。
2. 部分脚本仍包含作者本机的绝对路径，运行前必须修改。
3. 训练和批量生成过程计算量较大，建议使用 CUDA GPU。
4. 数据目录名称需要与代码中的路径保持一致，例如 `iccad16_fake`、`iccad16` 和 `fake` 不能混用。
5. 训练产生的模型权重、采样图片和生成结果建议单独保存，不要全部提交到 GitHub。

## 引用

如果本项目用于论文、报告或其他研究工作，请引用对应的论文或项目来源。

## License

当前仓库尚未指定开源许可证。如需公开发布，建议根据项目用途选择合适的许可证，例如 MIT License 或 Apache License 2.0。
