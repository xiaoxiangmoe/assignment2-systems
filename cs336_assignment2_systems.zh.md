# CS336 作业 2（系统）：系统与并行

版本 26.1.3
CS336 课程组
2026 年春季

## 1 作业概览

在本次作业中，你将获得提升单 GPU 训练速度、并将训练扩展到多块 GPU 的实践经验。

**你将实现的内容**

1. 基准测试（benchmarking）与性能分析（profiling）框架
2. 激活检查点（activation checkpointing）
3. FlashAttention-2 Triton 内核
4. 分布式数据并行（distributed data parallel）训练
5. 优化器状态分片（optimizer state sharding）
6. 完全分片数据并行（fully sharded data parallel）训练

**代码结构**

作业代码与本讲义均可在 GitHub 上获取：

github.com/stanford-cs336/assignment2-systems

请使用 Git 克隆（clone）该仓库。如有任何更新，我们会通知你，你可以通过 `git pull` 获取最新内容。

1. `cs336-basics/`：在本次作业中，你将对作业 1 中构建的一些组件进行性能分析。该文件夹包含作业 1 的官方参考实现代码，因此你可以在其中找到 `cs336-basics/pyproject.toml` 和 `cs336-basics/cs336_basics` 包。如果你想使用自己的模型实现，可以修改根目录下的 `pyproject.toml` 文件，使其指向你自己的包。
2. `/`：cs336-systems 的根目录。我们创建了一个名为 `cs336_systems` 的空模块。注意，其中没有任何代码，因此你完全可以从零开始自由实现。
3. `tests/*.py`：该目录包含你必须通过的所有测试。这些测试调用 `tests/adapters.py` 中定义的钩子（hook）。你需要实现这些适配器（adapter），把你的代码与测试连接起来。编写更多测试和/或修改测试代码可能有助于调试你的代码，但你的实现必须通过原始提供的测试套件。
4. `README.md`：该文件包含关于预期目录结构的更多细节，以及搭建环境的一些基本说明。

**如何提交**

你将向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题。请对你的回答进行排版。
- `code.zip`：包含你编写的所有代码。

运行 `test_and_make_submission.sh` 脚本即可创建 `code.zip` 文件。

## 2 性能分析与基准测试

在本作业的第一部分，我们将探索如何优化 Transformer 模型的性能，以最高效地利用 GPU。我们将对模型进行性能分析，弄清它在前向传播（forward pass）和反向传播（backward pass）中把时间和内存花在了哪里，然后用自定义 GPU 内核优化自注意力（self-attention）操作，使其比常规 PyTorch 所能达到的更快。在本作业的后续部分，我们将利用多块 GPU，并理解如何在集群上训练模型。

### 2.1 性能分析

在实现任何优化之前，最好先对程序做一次性能分析（profile），弄清它的资源（如时间和内存）花在了哪里。否则，我们可能会去优化那些并不占用大量时间或内存的模型部分，因而看不到可衡量的端到端改进。

我们将实现三条性能评估路径：

1. 使用 Python 标准库进行简单的端到端基准测试，为前向传播和反向传播计时；
2. 使用 NVIDIA Nsight Systems 工具进行计算性能分析，弄清这些时间如何分布在 CPU 和 GPU 两侧的各个操作上；
3. 内存性能分析。

#### 2.1.1 准备工作——导入你的 basics Transformer 模型

首先确保你能加载上一份作业中的模型。在上一次作业中，我们把模型组织成了一个 Python 包，以便之后轻松导入。我们已在 `./cs336-basics` 文件夹中加入了模型的参考实现，并在 `pyproject.toml` 文件中指向了它。像往常一样调用 `uv run [command]` 时，uv 会自动定位到这个本地的 `cs336-basics` 包。如果你想使用自己的模型实现，可以修改 `pyproject.toml` 文件，使其指向你自己的包。

你可以用以下命令测试能否导入你的模型：

```bash
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

现在应该可以使用作业 1 中的相关模块了（例如，对于 `model.py`，可以用 `import cs336_basics.model` 导入）。

#### 2.1.2 模型规模

在本次作业中，我们将对模型进行基准测试与性能分析，以更好地理解其性能。为了感受规模变化带来的影响，我们将使用并引用以下模型配置。除排行榜（leaderboard）之外，所有模型都使用 10,000 的词表大小和 4 的批量大小（batch size），上下文长度（context length）则各不相同。本次作业（以及之后的作业）会要求用表格和图来呈现大量结果。我们强烈建议你用代码自动生成本报告的表格，因为在 LaTeX 或 Typst 中手工排版表格非常繁琐。可以参考 `pandas.DataFrame.to_latex()` 和 `pandas.DataFrame.to_typst()`，或者自己写一个函数，从你偏好的表格表示生成它们。

| 规模 | d_model | d_ff | num_layers | num_heads |
|------|---------|------|------------|-----------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

表 1：不同模型规模的配置。这些配置大多基于 GPT-2 的配置。除非另有说明，均使用上下文长度 512。

#### 2.1.3 端到端基准测试

我们现在来实现一个简单的性能评估脚本。我们将测试模型的许多变体（改变精度、替换层等），因此让脚本支持通过命令行参数启用这些变体会很有价值，之后运行起来会方便得多。

首先，我们对模型做尽可能简单的性能分析：为前向传播、反向传播和优化器步骤计时。由于我们只测量速度和内存，使用随机权重和数据即可。

测量性能是件微妙的事——一些常见的陷阱会导致我们测不到想测的东西。对 GPU 代码做基准测试时，一个注意事项是 CUDA 调用是异步的。当你调用一个 CUDA 内核时——比如调用 `torch.matmul` 时——PyTorch 函数调用会立即把控制权交还给你的代码，而不会等待矩阵乘法完成。这样，CPU 可以在 GPU 完成矩阵乘法的同时继续向前运行并调度新操作，这是一个重要的性能优势。但另一方面，这意味着天真地测量 `torch.matmul` 调用返回所需的时间，并不能告诉我们 GPU 实际运行矩阵乘法花了多长时间。在 PyTorch 中，我们可以调用 `torch.cuda.synchronize()` 等待所有已调度的 GPU 内核完成，从而更准确地测量 CUDA 内核的运行时间。这里的“同步”指的是让 CPU 运行时与 GPU 运行时同步。理解了这一点，我们来编写基础的性能分析基础设施。

**问题 (benchmarking_script)：基准测试脚本（4 分）**

(a) 编写一个脚本，对你的模型的前向传播、反向传播和优化器步骤进行基本的端到端基准测试。具体来说，你的脚本应支持以下功能：

- 给定超参数（例如层数），初始化一个模型。
- 生成一个随机数据批次。
- 先运行 $w$ 个预热（warm-up）步骤（在开始计时之前），然后对 $n$ 个步骤的执行计时（根据参数决定：仅前向传播；前向加反向传播；或前向、反向加优化器步骤）。计时可以使用 Python 的 `timeit` 模块（可以使用 `timeit` 函数，也可以使用 `timeit.default_timer()`，后者提供系统最高分辨率的时钟，因此比 `time.time()` 更适合作为基准测试的默认选择）。
- 在每个步骤之后调用 `torch.cuda.synchronize()`。

**提交要求：** 一个脚本，用给定的超参数初始化一个 basics Transformer 模型，创建一个随机数据批次，并对仅前向传播、前向加反向传播，以及包含优化器步骤的完整训练步骤分别计时。

(b) 对 2.1.2 节中描述的各模型规模，测量前向传播、反向传播和优化器步骤的耗时。使用 5 个预热步骤，并在 10 个测量步骤上计算计时结果的平均值和标准差。一次前向传播需要多长时间？一次反向传播呢？各次测量之间的波动大吗，还是标准差很小？

**提交要求：** 用 1-2 句话给出你的计时结果。

(c) 基准测试的一个隐患是没有执行预热步骤。不做预热步骤，重复你的分析。这会对结果产生什么影响？你认为为什么会这样？再试着用 1 或 2 个预热步骤运行脚本。为什么结果仍可能不同？

**提交要求：** 用 2-3 句话回答。

#### 2.1.4 Nsight Systems 性能分析器

端到端基准测试无法告诉我们模型在前向和反向传播中把时间和内存花在了哪里，因此也无法揭示具体的优化机会。要想知道程序在每个组件（例如函数）上花了多少时间，我们可以使用性能分析器（profiler）。执行性能分析器通过在函数开始和结束运行时插入守卫（guard）来对代码进行插桩（instrument），从而给出函数级的详细执行统计（如调用次数、平均耗时、在该函数上累计花费的时间等）。

标准的 Python 性能分析器（如 CProfile）无法分析 CUDA 内核，因为这些内核在 GPU 上异步执行。幸运的是，NVIDIA 提供了一个可以通过命令行工具 `nsys` 使用的性能分析器。我们建议你从包管理器或其下载页面的安装程序获取最新版本。在本作业的这一部分，你将使用 `nsys` 分析你的 Transformer 模型的运行情况。

使用 `nsys` 很直接：在上一节的 Python 脚本前加上 `nsys profile` 运行即可。例如，可以对脚本 `benchmark.py` 做一次基本的性能分析：

```bash
$ uv run nsys profile -- python benchmark.py
```

随后你可以在本地机器上用 NVIDIA Nsight Systems 桌面应用查看分析结果。在分析视图的 CUDA API 行中选中某个 CUDA API 调用（CPU 侧），会在 CUDA HW 行中高亮所有对应的内核执行（GPU 侧）。

一次更全面的性能分析命令可能长这样：

```bash
$ uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py
```

图 1：一次详细的 Nsight Systems 跟踪（trace）。（图见英文原版 PDF。）

在这个例子中，`--trace` 指定记录哪些 API，`--pytorch` 在模块调用和 autograd 期间插入 nvtx 标签，`--cudabacktrace` 和 `--python-backtrace` 提供更好的回溯信息，帮助定位某个内核是从代码的哪个位置发起的，`--gpu-metrics-devices` 指定测量哪块 GPU 的利用率。

为一次运行添加性能分析并不是免费的，整体会拖慢运行。通常值得只启用当次运行所需要的功能。具体来说，在不需要回溯信息时，你可能想去掉 `--cudabacktrace=all` 和 `--python-backtrace=cuda`，因为它们的额外开销特别大。

我们鼓励你尝试 `nsys profile` 的各种命令行选项，感受它的能力。你还可以用 NVTX 范围（range）标注代码，它们会在分析结果的 NVTX 行中显示为区块，涵盖其中的所有 CUDA API 调用及关联的内核执行。特别地，你应当用 NVTX 范围来忽略基准测试脚本中的预热步骤（通过对该 nvtx 标签应用 `--nvtx-capture` 过滤器）。你还可以隔离出负责模型前向与反向传播的内核，甚至可以按如下方式标注实现，隔离出自注意力层不同部分所对应的内核：

```python
...
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    ... # Q, K, V, mask
):
    ...
    with nvtx.range("computing attention scores"):
        ... # compute attention scores between Q and K
    with nvtx.range("computing softmax"):
        ... # compute softmax of attention scores
    with nvtx.range("final matmul"):
        ... # compute output projection
    return ...
```

然后可以在基准测试脚本中用标注版替换原始实现：

```python
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
```

最后值得注意的是，`torch.compile` 会使时间和资源难以归因到代码的具体部分。你可能需要用 `torch.compile` 和 nvtx 标注对代码的不同部分进行包裹与拆解，才能把时间和资源使用正确地归因到源码的各个部分。

**问题 (nsys_profile)：Nsight Systems 性能分析（5 分）**

使用 `nsys` 对你的前向传播、反向传播和优化器步骤进行性能分析：从表 1 中自选两种模型规模，并配合三个大于 128 的 2 的幂次上下文长度，其中最大可用的模型规模应搭配你能放进内存的最长上下文长度。挑选你认为最值得观察的组合。对每次分析回答以下问题：

(a) 前向传播总共花了多少时间？它与之前用 Python 标准库测得的结果一致吗？

**提交要求：** 用 1-2 句话回答。

(b) 前向传播期间，哪个 CUDA 内核占用的累计 GPU 时间最多？在模型的一次前向传播中，这个内核被调用了多少次？当你同时做前向和反向传播时，占用运行时间最多的还是同一个内核吗？（提示：查看 “Stats System View” 下的 “CUDA GPU Kernel Summary”，并用 NVTX 范围过滤，以识别模型的哪些部分对应哪些内核。）

**提交要求：** 用 1-2 句话回答。

(c) 尽管绝大多数 FLOP 发生在矩阵乘法中，你会注意到仍有若干其他内核占据了不容小觑的整体运行时间。在前向传播中，除矩阵乘法之外，你还看到哪些内核占据了不容忽视的 CUDA 运行时间？

**提交要求：** 用 1-2 句话回答。

(d) 用你实现的 AdamW 分析一个完整训练步骤的运行（即前向传播、计算损失并运行反向传播，最后是优化器步骤，就像训练时那样）。与仅做推理（只有前向传播）相比，花在矩阵乘法上的时间占比如何变化？其他内核呢？

**提交要求：** 用 1-2 句话回答。

(e) 在前向传播中，比较模型自注意力层内 softmax 操作与矩阵乘法操作的运行时间。运行时间的差异与 FLOP 的差异相比如何？

**提交要求：** 用 1-2 句话回答。

#### 2.1.5 混合精度

到目前为止，在本作业中我们一直以 FP32 精度运行——所有模型参数和激活值都是 `torch.float32` 数据类型。然而，现代 NVIDIA GPU 包含专门用于加速低精度矩阵乘法的 GPU 核心（Tensor Core）。例如，NVIDIA B200 的规格表标明其 FP32 最大吞吐为 80 TFLOPS，而 FP16（半精度浮点）或 BF16（bfloat16）的最大吞吐则明显更高，达到惊人的 2500 TFLOPS。因此，使用低精度数据类型应当能帮助我们加速训练与推理。

然而，天真地把模型转换为低精度格式可能会降低模型精度。例如，实践中许多梯度值太小，无法用 FP16 表示，因此在天真地用 FP16 精度训练时会变成零。为应对这一问题，用 FP16 训练时通常会使用损失缩放（loss scaling）——直接把损失乘以一个缩放因子，增大梯度幅度，使其不至于下溢为零。此外，FP16 的动态范围比 FP32 小，可能导致溢出，表现为 NaN 损失。完整的 bfloat16 训练通常更稳定（因为 BF16 与 FP32 的动态范围相同），但与 FP32 相比仍可能影响最终的模型性能。

为了利用低精度数据类型带来的加速，通常使用混合精度（mixed-precision）训练。在 PyTorch 中，这通过 `torch.autocast` 上下文管理器实现。此时，某些操作（如矩阵乘法）以低精度数据类型执行，而其他需要 FP32 完整动态范围的操作（如累加与归约）则保持原样。例如，以下代码会在前向传播中自动识别哪些操作应以低精度执行，并把这些操作转换为指定的数据类型：

```python
model: torch.nn.Module = ...  # e.g. your Transformer model
dtype: torch.dtype = ...  # e.g. torch.bfloat16
x: torch.Tensor = ...  # input data
with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x)
```

如上所述，即使被累加的张量本身已被降精度转换，让累加保持在高精度通常也是个好主意。下面的练习将帮助你建立直觉，理解为什么如此。

**问题 (mixed_precision_accumulation)：混合精度累加（1 分）**

运行以下代码，并评论结果的准确性。

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```

**提交要求：** 用 2-3 句话回答。

接下来，我们先把混合精度应用到一个玩具模型上建立直觉，然后再应用到基准测试脚本。

**问题 (benchmarking_mixed_precision)：混合精度基准测试（2 分）**

(a) 考虑以下模型：

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

假设我们在 GPU 上训练该模型，且模型参数原本是 FP32。我们想使用 FP16 的 autocast 混合精度。以下各项的数据类型分别是什么：

- autocast 上下文中的模型参数？
- 第一个前馈层（`ToyModel.fc1`）的输出？
- 层归一化（`ToyModel.ln`）的输出？
- 模型预测的 logits？
- 损失？
- 模型的梯度？

**提交要求：** 列出上述每个组件的数据类型。

(b) 你应当已经看到，FP16 混合精度 autocast 对待层归一化层与对待前馈层不同。层归一化的哪些部分对混合精度敏感？如果改用 BF16 而不是 FP16，我们还需要区别对待层归一化吗？为什么？

**提交要求：** 用 2-3 句话回答。

(c) 修改你的基准测试脚本，使其可以选择用 BF16 混合精度运行模型。对 2.1.2 节描述的每种语言模型规模，分别测量有无混合精度时前向与反向传播的耗时。比较全精度与混合精度的结果，并评论随模型规模变化的趋势。你可能会发现 `nullcontext` 这个无操作（no-op）上下文管理器很有用。

**提交要求：** 用 2-3 句话给出你的计时结果与评论。

#### 2.1.6 内存性能分析

到目前为止，我们一直在关注计算性能。现在把注意力转向内存——语言模型训练与推理中的另一大资源。PyTorch 也自带一个强大的内存性能分析器，可以跟踪随时间变化的内存分配。

要使用内存性能分析器，可以按如下方式修改你的基准测试脚本：

```python
...  # warm-up phase in your benchmarking script
# Start recording memory history.
torch.cuda.memory._record_memory_history(max_entries=1000000)
...  # what you want to profile in your benchmarking script
# Save a pickle file to be loaded by PyTorch's online tool.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)
```

这会输出一个 `memory_snapshot.pickle` 文件，你可以把它加载到以下在线工具中：pytorch.org/memory_viz。该工具能让你看到整体内存使用的时间线，以及每一笔具体的分配——包括其大小和指向其来源代码的调用栈。使用该工具时，先在浏览器中打开上面的链接，然后把 pickle 文件拖放到页面上。

现在，你将使用 PyTorch 性能分析器分析模型的内存使用。

**问题 (memory_profiling)：内存性能分析（4 分）**

对表 1 中的 xl 模型，在上下文长度 128 和 2048 下，分析包含前向传播、反向传播和优化器步骤的完整训练步骤。

(a) 给你的性能分析脚本添加一个选项，使模型经由内存性能分析器运行。复用之前的一些基础设施可能会有帮助（例如启用混合精度、加载特定模型规模等）。然后运行脚本，分别获取 xl 模型在仅推理（只有前向传播）和完整训练步骤下的内存分析结果。你的内存时间线长什么样？你能根据看到的峰值判断正在运行哪个阶段吗？

**提交要求：** 两张来自 memory_viz 工具的 xl 模型 “Active memory timeline” 图片：一张对应前向传播，一张对应运行完整训练步骤（前向与反向传播，然后优化器步骤），外加 2-3 句话的回答。

(b) 仅做前向传播时，每种上下文长度的峰值内存用量是多少？做完整训练步骤时呢？

**提交要求：** 一张表格，每种上下文长度两个数字。

(c) 找出 xl 模型使用混合精度时的峰值内存用量，仅前向传播与完整训练步骤两种情况都要。混合精度会显著影响内存用量吗？

**提交要求：** 用 2-3 句话回答。

(d) 考虑 xl 模型。给定我们的参考超参数，Transformer 残差流（residual stream）中一个激活张量在单精度下有多大？用 MiB 表示（即把字节数除以 $1024^2$）。

**提交要求：** 用 1-2 句话给出你的推导。

(e) 现在仔细观察 pytorch.org/memory_viz 中 xl 模型前向传播内存快照的 “Active Memory Timeline”。当你降低 “Detail” 级别时，工具会按相应级别隐藏最小的分配（例如，把 “Detail” 设为 10% 时只显示最大的 10% 的分配）。显示出来的最大分配是多大？查看调用栈，你能说出这些分配来自哪里吗？

**提交要求：** 用 1-2 句话回答。

(f) Nsight Systems 也有用于内存性能分析的标志。你可以把它们与之前的 Nsight 标志结合起来，弄清在模型生命周期的不同步骤中发生了哪些分配。利用 PyTorch 提供的 NVTX 标签，确定模型中单个 TransformerBlock 为反向传播保存了多少内存（这些张量常被称为残差，residual）。指出贡献最大的 5 个操作，以及它们在总内存中各占的百分比。

在反向传播期间，这些张量都会被释放，但同时会产生新的梯度张量。根据你的分析结果——前向传播期间分配了多少内存，以及反向传播中每经过一个 TransformerBlock 内存用量如何变化——计算一个 TransformerBlock 产生的梯度张量占用多少内存。结果符合你的预期吗？

**提交要求：** Nsight Systems 的截图，外加 1-2 段话的回答。

## 3 单 GPU 内存

本作业的后面部分将探索把张量分片到多块 GPU 上的技巧，但也有一些技巧即使对单 GPU 训练也适用。其中最常见的是梯度检查点（gradient checkpointing，亦称激活检查点，activation checkpointing）。

### 3.1 Autograd 残差

回忆一下：为了对模型做反向传播，我们需要保存前向传播中产生的激活。虽然这对某些操作来说显然必要，但默认情况下被保存的会比你想的多得多。这些为反向传播保存的张量称为“残差”（residual），或干脆叫“保存的张量”（saved tensor）。

我们来认识一下网络里到底保存了什么。从我们不起眼的 RMSNorm 函数开始（为简单起见使用纯 FP32），我们给 autograd 保存或取回张量的时刻加上一些钩子：

```python
import torch
from torch import nn

x = torch.randn((4, 512, 2560), requires_grad=True)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


ln = RMSNorm(x.shape[-1])
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

输出显示被写出的张量数量令人担忧，其中好几个都是完整激活大小！

```
$ uv run scripts/autograd_experiment.py
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
```

#### 3.1.1 算子融合

在这个例子中，显然所用操作的粒度过高了。我们希望有单个算子：输入 RMSNorm 权重和激活，吐出输出，并且在反向传播中也是单一整体。这正是内核融合（kernel fusion）的动机之一。由于 RMSNorm 的行为相当规整，我们甚至可以用 `torch.compile` 自动融合它：

```python
...
ln = torch.compile(RMSNorm(x.shape[-1]))
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

新输出明显好多了：

```
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
```

反向传播只需保存一个完整大小的激活张量——即 RMSNorm 函数的输入。还要注意，加载的顺序不再是保存的逆序，每个残差也不再有 grad_fn 依赖——PyTorch 把我们整个 RMSNorm 当成了单个函数。

### 3.2 激活检查点

融合无疑有用，但能省的内存有限。举个例子，我们把 xl 规模的单个 TransformerBlock 融合起来看看：

```python
import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))
# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)
...
# Now logs the number of bytes saved
total_size_bytes = 0


def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):  # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


...
# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x)
print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")
```

脚本告诉我们为反向传播保存了多少内存：

```
...
Total size of saved tensors in single TransformerBlock: 3651.31 MiB
```

每层 3.6 GiB。如果对每一层都这样做，仅是为反向传播保存的激活就达到 114 GiB！注意力操作的残差中存在不小的浪费，我们将在第 4 节修复；但即使修复之后，内存占用仍会随批量大小、序列长度和嵌入维度线性增长。

#### 3.2.1 重计算

与其守住生成的每个张量不放，我们也可以只周期性地保存结果的检查点，并重计算中间的值。PyTorch 提供了一个可以简单调用的接口来处理这件事。`torch.utils.checkpoint.checkpoint` 接收一个函数及该函数的参数，然后按如下方式修改所传入函数的行为：

1. 在前向传播中：
   1. 保存函数的输入值；
   2. 抑制前向传播中张量的保存。
2. 在反向传播中：
   1. 前置一个重计算步骤：从之前保存的输入重新计算前向传播，并为反向传播保存所需的值；
   2. 运行反向传播，随后所有张量都可以释放。

在跑 4 个 transformer block 的简单例子中，内存如预期般累加：

```python
...


def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks(x)
print(f"Total size of saved tensors in four TransformerBlocks: {total_size_bytes / (1024**2):.2f} MiB")
```

```
Total size of saved tensors in four TransformerBlocks: 14605.25 MiB
```

但我们可以这样使用梯度检查点：

```python
from torch.utils.checkpoint import checkpoint


def two_blocks(x):
    x = block(x)
    x = block(x)
    return x


def four_blocks_checkpoint(x):
    # checkpoint throws out all the saved tensors until the backward pass
    # when getting to the checkpointed block in the backward pass,
    # it reruns a forward pass to produce the saved tensors,
    # then completes normal backward pass.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
print(f"Total size of saved tensors in four TransformerBlocks with checkpointing: {total_size_bytes / (1024**2):.2f} MiB")
```

```
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=<torch.autograd.function.CompiledFunctionBackward object at 0x7aa0657a19d0>
Total size of saved tensors in four TransformerBlocks with checkpointing: 160.00 MiB
```

请记住，这并没有消除内存使用，而是把内存使用分解成了两类：为准备重计算而在每个检查点调用入口处保存的长期存储（检查点本身），以及在被检查点包裹的块内重计算时产生的、用于支撑反向传播的短期内存。由于我们主要关心峰值内存，我们希望平衡保存检查点的内存成本与物化整个块的残差的内存成本。使用的检查点越多，单个块内需要物化的内存越少，但检查点本身需要的内存越多。

在上面的例子中，显然重计算残差的内存占主导（部分原因是我们的检查点很小），因此把天平向检查点内存一侧多挪一些是有利的。也就是说，我们应当缩小被检查点包裹的块的范围。

使用更大或更小的检查点块并不影响重计算带来的计算成本，但我们还可以用递归检查点（recursive checkpointing）——把 checkpoint 调用嵌套在其他 checkpoint 调用内部——以计算为代价继续降低所需内存。

**问题 (gradient_checkpointing)：内存最优的梯度检查点（4 分）**

考虑一个由 $N$ 个相同 block 顺序堆叠而成的 Transformer。不做任何检查点时，所有 $N$ 个块的残差都同时存活，峰值激活内存为 $O(N)$。我们可以自由地把前向传播的任意子集包裹进 `checkpoint`，包括把 checkpoint 调用互相嵌套。

(a) 忽略计算成本，什么检查点策略能最小化峰值激活内存？描述你会如何安排 checkpoint 调用（给出代码示意即可），并给出你的策略的渐近峰值激活内存和计算量（表示为 $N$ 的函数）。假设单个块保存的残差远大于每个检查点的簿记开销。

**提交要求：** 用 3-5 句话描述该策略及其渐近峰值内存，外加一小段代码示意。

(b) 考虑如上所述的 xl 模型配置，批量大小为 4，序列长度为 2048。如果你只有运行一步重计算的时间/计算预算（即不允许嵌套 checkpoint 调用），降低峰值内存的最佳检查点策略是什么？对你的运行做峰值内存分析以验证你的假设。同时比较次小和次大的检查点块规模的峰值内存，以确保结论可靠。

**提交要求：** 用 3-5 句话描述你的推理过程，并给出你的策略实测的峰值内存。

## 4 GPU 内核

### 4.1 用 FlashAttention-2 优化注意力

#### 4.1.1 PyTorch 注意力基准测试

你的性能分析很可能表明，注意力层在内存和计算两方面都有优化机会。从高层看，注意力操作由一次矩阵乘法、一次 softmax、再接一次矩阵乘法组成：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\mathrm{mask}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\right) V \tag{1}$$

朴素的注意力实现需要为每个 batch/head 元素保存形状为 $\text{seq\_len} \times \text{seq\_len}$ 的注意力分数矩阵，它随序列长度增长会非常大，对任何输入或输出较长的任务都会导致内存不足（out-of-memory）错误。我们将按照 FlashAttention-2 论文实现一个注意力内核：它按分块（tile）计算注意力，从不显式物化 $\text{seq\_len} \times \text{seq\_len}$ 的注意力分数矩阵，从而能扩展到更长的序列。

**问题 (pytorch_attention)：PyTorch 注意力基准测试（2 分）**

(a) 在不同规模下对你的注意力实现做基准测试。编写一个脚本，使其：

(i) 把批量大小固定为 8，且不使用多头注意力（即去掉 head 维度）。

(ii) 遍历 head 嵌入维度 $d_{\text{model}}$ 取值 [16, 32, 64, 128] 与序列长度取值 [256, 1024, 4096, 8192, 16384] 的笛卡尔积。

(iii) 为相应的尺寸创建随机输入 $Q$、$K$、$V$。

(iv) 用这些输入对注意力做 100 次前向传播并计时。

(v) 测量反向传播开始前的内存占用，并对 100 次反向传播计时。

(vi) 确保预热，并在每次前向/反向传播后调用 `torch.cuda.synchronize()`。

取决于你的 GPU，其中一些配置预计会内存不足。报告你在这些配置下得到的计时（或内存不足错误）。在多大的规模下会出现内存不足错误？找一个内存不足的最小配置，核算一下注意力的内存用量（可以使用作业 1 中 Transformer 内存用量的公式）。为反向传播保存的内存随序列长度如何变化？你会怎样消除这部分内存成本？

**提交要求：** 一张包含你的计时结果的表格、你对内存用量的计算，以及 1-2 段话的回答。

### 4.2 JIT 编译注意力的基准测试

自 2.0 版本起，PyTorch 还附带一个强大的即时（just-in-time）编译器，它会自动尝试对 PyTorch 函数应用多种优化：简介见 https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html 。特别地，它会通过动态分析计算图，尝试自动生成融合的 Triton 内核。使用 PyTorch 编译器的接口非常简单。例如，要把它应用到模型的单层上，可以写：

```python
layer = SomePyTorchModule(...)
compiled_layer = torch.compile(layer)
```

现在，`compiled_layer` 在功能上与 `layer` 表现一致（包括其前向与反向传播）。我们也可以用 `torch.compile(model)` 编译整个 PyTorch 模型，甚至编译一个调用 PyTorch 操作的 Python 函数。

**问题 (torch_compile)：Torch Compile（2 分）**

(a) 扩展你的注意力基准测试脚本，加入 PyTorch 注意力实现的编译版本，并在上面 pytorch_attention 问题中的相同配置下，比较它与未编译版本的性能。

**提交要求：** 一张表格，比较编译后的注意力模块与上面 pytorch_attention 问题中未编译版本的前向和反向传播计时。

(b) 现在，在你的端到端基准测试脚本中编译整个 Transformer 模型。前向传播的性能如何变化？前向加反向传播以及优化器步骤合在一起呢？

**提交要求：** 一张比较原始版与编译版 Transformer 模型的表格。

鉴于我们已经看到的随序列长度变化的扩展行为，要处理长序列必须有显著的改进。即使有 `torch.compile`，当前的实现在长序列下仍有很差的内存访问模式。为此，我们将编写一个 FlashAttention-2 的 Triton 实现——在那里我们能显著更好地控制内存如何被访问、何时计算什么。

#### 4.2.1 示例——加权和

为了介绍你需要了解的 Triton 知识以及它如何与 PyTorch 互操作，我们以一个“加权和”（weighted sum）操作的示例内核来展开。关于上手 Triton 的更多资源，见 Triton 教程。需要指出，这些教程没有使用新的、更方便的块指针（block pointer）抽象，我们会在下文中讲解。

给定输入矩阵 $X$，我们把它的元素乘以逐列的权重向量 $w$，并对每行求和，得到 $X$ 与 $w$ 的矩阵-向量乘积。我们先讲该操作的前向传播，然后编写反向传播的 Triton 内核。

**前向传播**

我们内核的前向传播就是下面这个广播内积：

```python
def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)
```

编写 Triton 内核时，我们让每个程序实例（可能并行运行）计算 $x$ 的一个行分块的加权和，并把对应的标量输出写入输出张量。在 Triton 中，程序实例是一个都运行同一程序的线程块，这些线程块可以在 GPU 上并行运行。我们不直接传入张量作为参数，而是传入指向其首元素的指针，以及每个张量的步长（stride），步长告诉我们如何沿各轴移动。

借助步长，并用程序 ID 划分工作（即实例 $i$ 处理 $x$ 的第 $i$ 个行分块），我们就能加载与当前运行实例要求和的 $x$ 行分块对应的张量。在这个简单例子中，Triton 版与 PyTorch 版前向传播的主要区别在于需要做指针算术以及显式的加载/存储。我们将使用 `tl.make_block_ptr` 的块指针抽象来大大简化指针算术，不过这意味着需要先做一些准备工作来构造块指针。

图 2：加权和内核示例（4.2.1 节）中的分块与块指针推进示意图。（图见英文原版 PDF。）

分块方式以及块指针如何推进，请参考图 2。上面的加权和函数写成 Triton 内核如下：

```python
import triton
import triton.language as tl


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row,
    x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)
    # Block pointers give us a way to select from an ND region of memory
    #  and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to load/store at a time
    # - The order of the dimensions in memory from major to minor
    #    axes (= np.argsort(strides)) for optimizations, needed for
    #    TMA support on >=Hopper
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide NUM_ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)
        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # Move by D_TILE_SIZE
    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide NUM_ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))
```

现在把这个内核包装进一个 PyTorch Autograd 函数，使其能与 PyTorch 互操作（即接收 Tensor 作为输入、输出 Tensor，并且之后还能在反向传播中与 autograd 引擎协同工作）：

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]
        # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape
        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)
        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])
```

注意，当我们用 `weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)]` 调用 Triton 内核时，我们通过传入元组 `(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)` 定义了一个所谓的线程块“启动网格”（launch grid）。随后，在内核中可以用 `tl.program_id(0)` 访问线程块索引。

**反向传播**

由于我们定义了自己的内核，因此也需要编写自己的反向函数。

在前向传播中，我们拿到层的输入，需要计算其输出。回忆一下：在反向传播中，我们拿到的是目标函数关于输出的梯度，需要计算关于每个输入的梯度。在我们的例子中，操作的输入是一个矩阵 $x: \mathbb{R}^{n \times h}$ 和一个权重向量 $w: \mathbb{R}^{h}$。为简洁起见，把这个操作记作 $f(x, w)$，其值域为 $\mathbb{R}^{n}$。于是，假设给定 $\nabla_{f(x,w)}\mathcal{L}$——损失 $\mathcal{L}$ 关于本层输出的梯度——我们可以应用多元链式法则，得到关于 $x$ 和 $w$ 的梯度表达式：

$$(\nabla_x \mathcal{L})_{ij} = \sum_{k=1}^{n} \frac{\partial f(x, w)_k}{\partial x_{ij}} (\nabla_{f(x,w)}\mathcal{L})_k = w_j \cdot (\nabla_{f(x,w)}\mathcal{L})_i \tag{2}$$

$$(\nabla_w \mathcal{L})_j = \sum_{i=1}^{n} \frac{\partial f(x, w)_i}{\partial w_j} (\nabla_{f(x,w)}\mathcal{L})_i = \sum_{i=1}^{n} x_{ij} \cdot (\nabla_{f(x,w)}\mathcal{L})_i \tag{3}$$

这给出了计算反向传播的简单公式。要求关于 $x$ 的反向步骤，应用式（2），取 $w$ 与 $\nabla_{f(x,w)}\mathcal{L}$ 的外积。要计算关于 $w$ 的反向步骤（即 $(\nabla_w \mathcal{L})_j$），必须把输入梯度与对应的输出行相乘。

我们的反向传播内核会先定义所有块指针，然后计算 $\nabla_x \mathcal{L}$：

```python
@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr,
    partial_grad_weight_ptr,  # Grad outputs
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)
    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(
            n_row_tiles,
            D,
        ),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)
        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # Never out of bounds for dim 0
        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
```

计算梯度 $\nabla_x \mathcal{L}$ 很直接，我们把结果写入输出张量对应的分块。然而，计算 $\nabla_w \mathcal{L}$ 更有挑战。每个内核实例负责 $x$ 的一个行分块，但现在需要对 $x$ 的行求和。我们不在反向传播中直接做这个求和，而是假设 `partial_grad_weight_ptr` 包含一个 $\text{n\_row\_tiles} \times H$ 的矩阵，其第一个维度只在来自 $x$ 的行分块内部归约。我们在写入该张量之前，先在当前行分块内做归约。在内核之外，我们用 `torch.sum` 把各个行分块的结果加起来，得到 $\nabla_w \mathcal{L}$。¹

于是，`autograd.Function` 的最后一部分相对简单：

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # ... (defined earlier)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # These don't have to be the same
        n_rows, D = x.shape
        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight
```

最后，我们得到了一个用起来很像 `torch.nn.functional` 中那些函数的函数：

```python
f_weightedsum = WeightedSumFunc.apply
```

现在，对两个 PyTorch 张量 $x$ 和 $w$ 调用 `f_weightedsum`，会得到类似如下的张量：

```
tensor([ 90.8563, -93.6815, -80.8884,  ..., 103.4840, -21.4634, -24.0192],
       device='cuda:0', grad_fn=<WeightedSumFuncBackward>)
```

注意张量上附带的 `grad_fn`——它表明 PyTorch 知道当这个张量出现在计算图中时，反向传播该调用什么。至此，我们完成了加权和操作的 Triton 实现。

---

¹ 当然，我们也可以自己为这一步写一个内核。

---

#### 4.2.2 FlashAttention-2 前向传播

你将按照 FlashAttention-2 [T. Dao, 2023] 用一个大幅改进的 Triton 实现替换你的 PyTorch 注意力实现。FlashAttention-2 运用了一些技巧按分块计算前向传播，从而获得高效的内存访问模式，并避免在全局内存上物化完整的注意力矩阵。

在进入本节之前，我们强烈建议至少阅读 FlashAttention 原始论文 [T. Dao et al., 2022]，它会给你关于 FlashAttention 实现高效注意力的核心技术的直觉：以在线（online）方式跨分块计算 softmax（这项技术由 [M. Milakov et al., 2018] 提出）。我们还推荐看看 H. He [4]，进一步了解 GPU 实际如何执行 PyTorch 代码。

**理解朴素注意力的低效之处**

回忆一下，注意力的前向传播（暂不考虑掩码）可以写成：

$$S = QK^\top/\sqrt{d} \tag{4}$$

$$P_{ij} = \mathrm{softmax}_j(S)_{ij} \tag{5}$$

$$O = PV \tag{6}$$

标准的反向传播是：

$$dV = P^\top dO \tag{7}$$

$$dP = dO\, V^\top \tag{8}$$

$$dS_i = \mathrm{dsoftmax}(dP_i) = (\mathrm{diag}(P_i) - P_i P_i^\top)\, dP_i \tag{9}$$

$$dQ = dS\, K/\sqrt{d} \tag{10}$$

$$dK = dS^\top Q/\sqrt{d} \tag{11}$$

可以看到，反向传播依赖前向传播中一些非常大的激活。例如，计算式（7）中的 $dV$ 需要 $P$，即形状为 (batch_size, n_heads, seq_len, seq_len) 的注意力分数——这个激活矩阵的大小随序列长度二次增长，这解释了之前在长序列下对注意力做基准测试时遇到的内存问题。在朴素注意力的前向与反向传播中，我们都要付出可观的内存 IO 成本，在片上 SRAM 与 GPU HBM 之间搬运 $P$ 和其他大激活。标准实现中有多次这样的传输：例如，一个标准的反向传播实现会在式（7）和式（9）的计算中各从 HBM 读一次 $P$。

FlashAttention 的主要目标是避免向 HBM 读写注意力矩阵，以降低 IO 与峰值内存成本。我们通过三种技术实现这一点：分块（tiling）、重计算（recomputation）和算子融合（operator fusion）。

**分块**

为避免向 HBM 读写注意力矩阵，我们在无法访问完整输入的情况下计算 softmax 归约。具体来说，我们把注意力计算重构为：把输入切成分块，并对输入分块做多趟处理，从而增量式地完成 softmax 归约。

**重计算**

我们避免把形状为 (batch_size, n_heads, seq_len, seq_len) 的大型中间注意力矩阵存入 HBM。取而代之的是，我们在 HBM 中保存某些“激活检查点”，然后在反向传播中重计算前向传播的一部分，以获得计算梯度所需的其他激活。FlashAttention-2 还保存注意力分数的 logsumexp，即 $L$，它将用于简化反向传播的计算。$L$ 的表达式为：

$$L_i = \log\left(\sum_j \exp(S_{ij})\right) \tag{12}$$

在最终的内核中，我们会以在线方式计算它，但最终结果应当相同。分块与重计算结合在一起，我们的内存 IO 与峰值用量就不再依赖于 $\text{sequence\_length}^2$，因此可以使用更长的序列长度。

**算子融合**

最后，我们通过在单个内核中执行所有操作，避免对注意力矩阵和其他中间激活的重复内存 IO——这称为算子融合或内核融合。我们将为前向传播编写单个 Triton 内核，在 HBM 与 SRAM 之间数据传输有限的情况下完成注意力涉及的所有操作。算子融合部分地由重计算支撑，因为这样可以避免通常需要为把每个中间激活存入 HBM 而付出的内存 IO。

关于这些技术的更多直觉，请查阅 FlashAttention 论文 [T. Dao et al., 2022; T. Dao, 2023]。

**基于重计算的反向传播**

利用 $L$，我们可以做适当的重计算并高效地计算反向传播。在开始反向传播之前，我们在全局内存中预先计算 $D = \mathrm{rowsum}(O \circ dO)$（其中 $\circ$ 是逐元素乘法），它等于 $\mathrm{rowsum}(P \circ dP)$，因为 $P\, dP^\top = P (dO\, V^\top)^\top = (PV)\, dO^\top = O\, dO^\top$（且对任意矩阵 $A$、$B$ 有 $\mathrm{rowsum}(A \circ B) = \mathrm{diag}(AB^\top)$）。有了 $L$ 和 $D$ 向量，反向传播就可以在不使用 softmax 的情况下计算。反向传播的完整计算现在是：

$$S = QK^\top/\sqrt{d} \tag{13}$$

$$P_{ij} = \exp(S_{ij} - L_i) \tag{14}$$

$$dV = P^\top dO \tag{15}$$

$$dP = dO\, V^\top \tag{16}$$

$$dS_{ij} = P_{ij}(dP_{ij} - D_i) \tag{17}$$

$$dQ = dS\, K/\sqrt{d} \tag{18}$$

$$dK = dS^\top Q/\sqrt{d} \tag{19}$$

可以看到，这一系列操作不要求我们在前向传播中把注意力分数 $P$ 存入 HBM——我们在式（13）和式（14）中从激活 $Q$、$K$ 和 $L$ 重计算它们。

**FlashAttention 前向传播的细节**

现在我们已经对 FlashAttention-2 所用的技术有了高层认识，接下来深入你将实现的 FA2 前向传播内核的细节。为了避免向 HBM 读写注意力矩阵，我们希望使用分块，即彼此独立地计算输出的每个分块。这要求我们能够计算 $P$ 的分块，理想情况下能在两个维度上（查询维度和键维度）都分块。

然而，对 $S$ 施加 softmax 时，需要对 $S$ 的整行做归约才能算出 softmax 的分母，这意味着我们无法直接按分块计算 $P$。FlashAttention-2 用在线 softmax 解决了这个问题。下文中，我们用下标 $i$ 表示当前的查询分块，用上标 $(j)$ 表示当前的键分块。查询维度上的分块大小为 $B_q$，键维度上的分块大小为 $B_k$。我们不在隐藏维度 $d$ 上分块。

我们还维护一些按行的运行值：$m_i^{(j)} \in \mathbb{R}^{B_q}$ 和 $l_i^{(j)} \in \mathbb{R}^{B_q}$。按行的 $m_i^{(j)}$ 是运行最大值，跟踪它是为了能以数值稳定的方式计算 softmax（回想作业 1 中我们 softmax 实现用过的这个技巧）。每当处理 $S$ 的新的按行分块时（即 $j$ 增大时），我们都会更新 $m_i^{(j)}$。利用运行最大值，可以把未归一化的 softmax 值（分子）计算为 $\tilde{P}_i^{(j)} = \exp(S_{ij} - m_i^{(j)})$。$l_i^{(j)}$ 是 softmax 分母的运行代理值，将用未归一化的 softmax 值按 $l_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)}) \cdot l_i^{(j-1)} + \mathrm{rowsum}(\tilde{P}_i^{(j)})$ 更新。最终写出输出时，我们需要用 $l_i^{(T_k)}$ 完成归一化，它是处理完所有键分块之后 $l_i^{(j)}$ 的最终值。算法 1 给出了 GPU 上的前向传播过程。

**算法 1：FlashAttention-2 前向传播**

1. **要求：** $Q \in \mathbb{R}^{N_q \times d}$，$K, V \in \mathbb{R}^{N_k \times d}$，分块大小 $B_q$、$B_k$
2. 把 $Q$ 切分为 $T_q = \lceil N_q / B_q \rceil$ 个大小为 $B_q \times d$ 的分块 $Q_1, \dots, Q_{T_q}$
3. 把 $K$、$V$ 切分为 $T_k = \lceil N_k / B_k \rceil$ 个大小为 $B_k \times d$ 的分块 $K^{(1)}, \dots, K^{(T_k)}$ 和 $V^{(1)}, \dots, V^{(T_k)}$
4. 对 $i = 1, \dots, T_q$ 循环：
5. 从全局内存加载 $Q_i$
6. 初始化 $O_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q \times d}$，$l_i^{(0)} = 0 \in \mathbb{R}^{B_q}$，$m_i^{(0)} = -\infty \in \mathbb{R}^{B_q}$
7. 对 $j = 1, \dots, T_k$ 循环：
8. 从全局内存加载 $K^{(j)}$、$V^{(j)}$
9. 计算 softmax 前的注意力分数分块 $S_i^{(j)} = \dfrac{Q_i (K^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$
10. 计算 $m_i^{(j)} = \max\left(m_i^{(j-1)},\ \mathrm{rowmax}\left(S_i^{(j)}\right)\right) \in \mathbb{R}^{B_q}$
11. 计算 $\tilde{P}_i^{(j)} = \exp\left(S_i^{(j)} - m_i^{(j)}\right) \in \mathbb{R}^{B_q \times B_k}$
12. 计算 $l_i^{(j)} = \exp\left(m_i^{(j-1)} - m_i^{(j)}\right)\, l_i^{(j-1)} + \mathrm{rowsum}\left(\tilde{P}_i^{(j)}\right) \in \mathbb{R}^{B_q}$
13. 计算 $O_i^{(j)} = \mathrm{diag}\left(\exp\left(m_i^{(j-1)} - m_i^{(j)}\right)\right) O_i^{(j-1)} + \tilde{P}_i^{(j)} V^{(j)}$
14. 结束循环
15. 计算 $O_i = \mathrm{diag}\left(l_i^{(T_k)}\right)^{-1} O_i^{(T_k)}$
16. 计算 $L_i = m_i^{(T_k)} + \log\left(l_i^{(T_k)}\right)$
17. 把 $O_i$ 作为 $O$ 的第 $i$ 个分块写入全局内存
18. 把 $L_i$ 作为 $L$ 的第 $i$ 个分块写入全局内存
19. 结束循环
20. 返回输出 $O$ 和 logsumexp $L$

在着手用 Triton 实现前向传播之前，我们先汇总一些编写 Triton 内核的通用技巧。

**Triton 技巧与窍门**

- 你可以在 Triton 中用 `tl.device_print` 打印语句来调试：https://triton-lang.org/main/python-api/generated/triton.language.device_print.html 。还有一个设置 `TRITON_INTERPRET=1` 可以在 CPU 上运行 Triton 解释器，不过我们发现它有 bug。
- 定义块指针时，确保偏移量正确，并且块偏移量要乘以相应的分块大小。
- 线程块的启动网格在 `torch.autograd.Function` 子类的方法中设置（正如我们在加权和示例中看到的）：

```python
kernel_fn[(launch_grid_d1, launch_grid_d2, ...)](...arguments...)
```

- 用 `tl.dot` 执行矩阵乘法。
- 要推进块指针，使用 `*_block_ptr = *_block_ptr.advance(...)`。

**问题 (flash_forward)：FlashAttention-2 前向传播（15 分）**

(a) 编写一个纯 PyTorch（不使用 Triton）的 `autograd.Function`，实现 FlashAttention-2 前向传播。它会比常规的 PyTorch 实现慢很多，但能帮助你调试 Triton 内核。

你的实现应接收输入 $Q$、$K$、$V$ 以及标志 `is_causal`，并产生输出 $O$ 和 logsumexp 值 $L$。本任务可以忽略 `is_causal` 标志。随后，`autograd.Function` 的 forward 应保存 $L$、$Q$、$K$、$V$、$O$ 供反向传播使用，并返回 $O$。记住，`autograd.Function` 的 forward 方法的第一个参数始终是上下文（ctx）。任何 `autograd.Function` 类都需要实现 backward 方法，但目前你可以让它直接抛出 `NotImplementedError`。如果需要参照物，可以用 PyTorch 实现式（4）至式（6）以及式（12），并比较输出。

接口为 `def forward(ctx, Q, K, V, is_causal=False)`。分块大小由你自行决定，但请确保至少为 $16 \times 16$。我们会始终用 2 的幂且不小于 16 的维度测试你的代码，因此你无需担心越界访问。

**提交要求：** 一个在前向传播中实现 FlashAttention-2 的 `torch.autograd.Function` 子类。为测试你的代码，实现 `[adapters.get_flashattention_autograd_function_pytorch]`。然后运行 `uv run pytest -k test_flash_forward_pass_pytorch`，确保你的实现通过该测试。

(b) 按照算法 1 为 FlashAttention-2 的前向传播编写一个 Triton 内核。然后，编写另一个 `torch.autograd.Function` 子类，在前向传播中调用这个（融合后的）内核，而不是用 PyTorch 计算结果。以下是一些针对本题的提示：

- 调试时，建议把你执行的每个 Triton 操作的结果与 (a) 中编写的分块 PyTorch 实现进行比较。
- 你的启动网格应设为 `(T_q, batch_size)`，即每个 Triton 程序实例只从单个 batch 索引加载元素，只对 $Q$、$O$、$L$ 的单个查询分块读写。
- 内核应只包含一个循环，遍历键分块 $1 \le j \le T_k$。
- 在循环末尾推进块指针。
- 使用下面的函数声明（利用我们给你的块指针，你应该能推断出其余指针的设置）：

```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    ...
```

其中 `scale` 为 $1/\sqrt{d}$，`Q_TILE_SIZE` 和 `K_TILE_SIZE` 分别为 $B_q$ 和 $B_k$。这些你之后可以调优。

以下补充准则可能帮你避免精度问题：

- 片上缓冲区（$O_i$、$l$、$m$）的 dtype 应为 `tl.float32`。如果要累加到输出缓冲区，使用 acc 参数（`acc = tl.dot(..., acc=acc)`）。
- 在把 $\tilde{P}_i^{(j)}$ 与 $V^{(j)}$ 相乘之前，先把 $\tilde{P}_i^{(j)}$ 转换为 $V^{(j)}$ 的 dtype；在把 $O_i$ 写入全局内存之前，也把它转换为合适的 dtype。类型转换用 `tensor.to` 完成。可以用 `tensor.dtype` 获取张量的 dtype，用 `*_block_ptr.type.element_ty` 获取块指针/指针的 dtype。

**提交要求：** 一个用你的 Triton 内核在前向传播中实现 FlashAttention-2 的 `torch.autograd.Function` 子类。实现 `[adapters.get_flash_autograd_function_triton]`。然后运行 `uv run pytest -k test_flash_forward_pass_triton`，确保你的实现通过该测试。

(c) 给你的 `autograd.Function` 实现添加一个作为最后一个参数的标志，用于因果掩码（causal masking）。它应是一个布尔标志，设为 True 时启用用于因果掩码的索引比较。你的 Triton 内核应有对应的附加参数 `is_causal: tl.constexpr`（此类型注解是必需的）。在 Triton 中，为查询和键构造合适的索引向量，并比较它们以形成大小为 $B_q \times B_k$ 的方形掩码。对被掩掉的元素，把常量 -1e6 加到注意力分数矩阵 $S_i^{(j)}$ 的对应元素上。确保用 `ctx.is_causal = is_causal` 保存掩码标志供反向传播使用。

**提交要求：** 为你的 `torch.autograd.Function` 子类添加一个附加标志，用你的 Triton 内核实现带因果掩码的 FlashAttention-2 前向传播。确保该标志可选且默认为 False，使之前的测试仍能通过。

**用重计算实现反向传播**

注意，与式（7）至式（11）的标准反向传播不同，我们可以利用重计算来避免式（13）至式（19）所示反向传播中的 softmax 操作。这意味着我们可以用一个平凡的内核计算反向传播，不需要任何在线技巧。因此，在这一部分，你可以通过对一个普通 PyTorch 函数（而非 Triton）调用 `torch.compile` 来实现反向传播。

**问题 (flash_backward)：FlashAttention-2 反向传播（5 分）**

用 PyTorch（而非 Triton）和 `torch.compile` 为你的 FlashAttention-2 `autograd.Function` 实现反向传播。你的实现应接收 $Q$、$K$、$V$、$O$、$dO$ 和 $L$ 张量作为输入，并返回 $dQ$、$dK$ 和 $dV$。记得计算并使用 $D$ 向量。你可以按照式（13）至式（19）的计算进行。

**提交要求：** 运行 `uv run pytest -k test_flash_backward` 测试你的实现。

现在来比较你的（部分基于）Triton 的 FlashAttention-2 实现与你的常规注意力 PyTorch 实现的性能。

**问题 (flash_benchmarking)：FlashAttention-2 基准测试（5 分）**

(a) 用 `triton.testing.do_bench` 写一个基准测试脚本，比较你的（部分基于）Triton 的 FlashAttention-2 前向与反向传播与常规 PyTorch 实现（即不使用 FlashAttention）的性能。

具体来说，你要报告一张表格，包含 Triton 与 PyTorch 两种实现的前向、反向以及端到端前向-反向传播的延迟。在开始基准测试前随机生成所需的输入，并在单块 B200 上运行基准测试。始终使用批量大小 1 和因果掩码。遍历以下笛卡尔积：序列长度取从 128 到 65536 的各个 2 的幂，嵌入维度取从 16 到 128 的各个 2 的幂，精度取 `torch.bfloat16` 和 `torch.float32`。你可能需要根据输入尺寸调整分块大小。

**提交要求：** 一张结果表格，按上述设置比较你的 FlashAttention-2 实现与 PyTorch 实现，报告前向、反向和端到端延迟。

#### 4.2.3 可选：Triton 反向传播

如果你有兴趣获得更多 Triton 练习，和/或想要一个快速的排行榜提交，我们在下面给出分块化的 FlashAttention-2 反向传播，你可以用 Triton 实现它。算法 2 展示了应在 Triton 中实现的 FlashAttention-2 反向传播。这里的一个关键技巧是计算两次 $P$：一次用于 $dQ$，另一次用于 $dK$ 和 $dV$。这让我们可以跳过线程块之间的同步，意味着可以避免缓慢的原子操作（atomics）。

**算法 2：分块化的 FlashAttention-2 反向传播**

1. **要求：** $Q, O, dO \in \mathbb{R}^{N_q \times d}$，$K, V \in \mathbb{R}^{N_k \times d}$，$L \in \mathbb{R}^{N_q}$，分块大小 $B_q$、$B_k$
2. 计算 $D = \mathrm{rowsum}(O \circ dO) \in \mathbb{R}^{N_q}$
3. 把 $Q$、$O$、$dO$ 切分为 $T_q = \lceil N_q / B_q \rceil$ 个分块 $Q_1, \dots, Q_{T_q}$，$O_1, \dots, O_{T_q}$，$dO_1, \dots, dO_{T_q}$，每个大小为 $B_q \times d$
4. 把 $K$、$V$ 切分为 $T_k = \lceil N_k / B_k \rceil$ 个分块 $K^{(1)}, \dots, K^{(T_k)}$ 和 $V^{(1)}, \dots, V^{(T_k)}$，每个大小为 $B_k \times d$
5. 把 $L$、$D$ 切分为 $T_q$ 个分块 $L_1, \dots, L_{T_q}$ 和 $D_1, \dots, D_{T_q}$，每个大小为 $B_q$
6. 对 $j = 1, \dots, T_k$ 循环：
7. 从全局内存加载 $K^{(j)}$、$V^{(j)}$
8. 初始化 $dK_0^{(j)} = dV_0^{(j)} = \mathbf{0} \in \mathbb{R}^{B_k \times d}$
9. 对 $i = 1, \dots, T_q$ 循环：
10. 从全局内存加载 $Q_i$、$dO_i$
11. 计算注意力分数分块 $S_i^{(j)} = \dfrac{Q_i (K^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$
12. 计算注意力概率 $P_i^{(j)} = \exp\left(S_i^{(j)} - L_i\right) \in \mathbb{R}^{B_q \times B_k}$
13. 计算 $dV_i^{(j)} = dV_{i-1}^{(j)} + \left(P_i^{(j)}\right)^\top dO_i \in \mathbb{R}^{B_k \times d}$
14. 计算 $dP_i^{(j)} = dO_i \left(V^{(j)}\right)^\top \in \mathbb{R}^{B_q \times B_k}$
15. 计算 $dS_i^{(j)} = P_i^{(j)} \circ \left(dP_i^{(j)} - D_i\right) \in \mathbb{R}^{B_q \times B_k}$
16. 计算 $dK_i^{(j)} = dK_{i-1}^{(j)} + \left(dS_i^{(j)}\right)^\top Q_i / \sqrt{d} \in \mathbb{R}^{B_k \times d}$
17. 结束循环
18. 把 $dK_{T_q}^{(j)}$ 和 $dV_{T_q}^{(j)}$ 作为 $dK$ 和 $dV$ 的第 $j$ 个分块写入全局内存
19. 结束循环
20. 对 $i = 1, \dots, T_q$ 循环：
21. 从全局内存加载 $Q_i$、$dO_i$
22. 初始化 $dQ_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q \times d}$
23. 对 $j = 1, \dots, T_k$ 循环：
24. 从全局内存加载 $K^{(j)}$、$V^{(j)}$
25. 计算注意力分数分块 $S_i^{(j)} = \dfrac{Q_i (K^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$
26. 计算注意力概率 $P_i^{(j)} = \exp\left(S_i^{(j)} - L_i\right) \in \mathbb{R}^{B_q \times B_k}$
27. 计算 $dP_i^{(j)} = dO_i \left(V^{(j)}\right)^\top \in \mathbb{R}^{B_q \times B_k}$
28. 计算 $dS_i^{(j)} = P_i^{(j)} \circ \left(dP_i^{(j)} - D_i\right) \in \mathbb{R}^{B_q \times B_k}$
29. 计算 $dQ_i^{(j)} = dQ_i^{(j-1)} + dS_i^{(j)} K^{(j)} / \sqrt{d} \in \mathbb{R}^{B_q \times d}$
30. 结束循环
31. 把 $dQ_i^{(T_k)}$ 作为 $dQ$ 的第 $i$ 个分块写入全局内存
32. 结束循环
33. 返回 $dQ$、$dK$、$dV$

## 5 分布式数据并行训练

在本作业的下一部分，我们将探索使用多块 GPU 训练语言模型的方法，重点关注数据并行（data parallelism）。我们先简介 PyTorch 中的分布式通信，然后研究分布式数据并行训练的一种朴素实现，再实现各种通信效率的改进并做基准测试。

### 5.1 PyTorch 中的单节点分布式通信

我们先看一个简单的 PyTorch 分布式应用：目标是生成四个随机整数张量并计算它们的和。

在下面的分布式例子中，我们将派生（spawn）四个 worker 进程，每个生成一个随机整数张量。为了跨 worker 进程对这些张量求和，我们将调用 all-reduce 集合通信操作，它会把每个进程上原来的数据张量替换为 all-reduce 后的结果（即总和）。

现在看代码：

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")


if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)
```

运行上面的脚本后，得到如下输出。如预期，每个 worker 进程最初持有不同的数据张量。all-reduce 操作把所有 worker 进程上的张量求和之后，每个 worker 进程上的 `data` 被原地修改为 all-reduce 的结果。²

```
$ uv run python distributed_hello_world.py
rank 3 data (before all-reduce): tensor([3, 7, 8])
rank 0 data (before all-reduce): tensor([4, 4, 7])
rank 2 data (before all-reduce): tensor([6, 0, 7])
rank 1 data (before all-reduce): tensor([9, 5, 3])
rank 1 data (after all-reduce): tensor([22, 16, 25])
rank 0 data (after all-reduce): tensor([22, 16, 25])
rank 3 data (after all-reduce): tensor([22, 16, 25])
rank 2 data (after all-reduce): tensor([22, 16, 25])
```

---

² 如果你多次运行这个脚本，会发现打印输出的顺序是不确定的。由于这个应用运行在分布式环境中，我们无法控制各命令执行的确切顺序——唯一的保证是：all-reduce 操作完成后，各进程持有的结果张量逐比特一致。

---

现在回头仔细看看上面的脚本。命令 `mp.spawn` 会派生 `nprocs` 个进程，以给定的 `args` 运行 `fn`。此外，`fn` 以 `fn(rank, *args)` 的方式被调用，其中 `rank` 是 worker 进程的索引（介于 0 和 `nprocs-1` 之间的值）。因此，我们的 `distributed_demo` 函数必须把这个整数 `rank` 作为第一个位置参数。此外，我们还传入了 `world_size`，它指 worker 进程的总数。

每个 worker 进程都属于一个进程组（process group），通过 `dist.init_process_group` 初始化。进程组表示一组将通过共享的 master 进行协调与通信的 worker 进程。master 由其 IP 地址和端口定义，rank 为 0 的进程担任 master。像 all-reduce 这样的集合通信操作作用于进程组中的每个进程。

在这个例子中，我们用 `"gloo"` 后端初始化进程组，但也有其他后端可用。特别地，`"nccl"` 后端会使用 NVIDIA NCCL 集合通信库，对 CUDA 张量通常性能更好。然而，NCCL 只能用于有 GPU 的机器，而 Gloo 可以在纯 CPU 机器上运行。做分布式 GPU 训练时应始终使用 NCCL，只在本地开发、没有 GPU 可用时使用 Gloo。本例使用 Gloo 是因为它能让我们在纯 CPU 机器上本地运行和开发。

运行多 GPU 作业时，确保不同的 rank 使用不同的 GPU。一种方法是在 setup 函数中调用 `torch.cuda.set_device(rank)`，这样 `tensor.to("cuda")` 会自动把张量移到指定设备。或者，你也可以显式构造每个 rank 的设备字符串（例如 `device = f"cuda:{rank}"`），然后把它作为任何数据移动的目标设备（例如 `tensor.to(f"cuda:{rank}")`）。

**术语**

在本作业剩余部分（以及你可能在网上看到的各种资料）中，你可能会在 PyTorch 分布式通信的语境下遇到以下术语。虽然本作业聚焦单节点、多进程的分布式训练，但这些术语对理解一般意义上的分布式训练很有用。可视化示意见图 3。

- **节点（node）**：网络中的一台机器。
- **worker**：参与分布式训练的程序实例。在本作业中，每个 worker 只有一个进程，因此我们会交替使用 worker、进程（process）和 worker 进程。然而，一个 worker 也可能使用多个进程（例如为训练加载数据），因此这些术语在实践中并不总是等价。
- **world size（全局规模）**：进程组中 worker 的总数。
- **全局 rank（global rank）**：在进程组中唯一标识一个 worker 的整数 ID（介于 0 和 world_size-1 之间）。例如，world size 为 2 时，一个进程的全局 rank 为 0（master 进程），另一个进程的 rank 为 1。
- **本地 world size（local world size）**：跨不同节点运行应用时，本地 world size 是在给定节点上本地运行的 worker 数量。例如，如果应用在 2 个节点上各派生 4 个 worker，则 world size 为 8，本地 world size 为 4。注意，在单节点上运行时，一个 worker 的本地 world size 等于（全局）world size。
- **本地 rank（local rank）**：在机器上唯一标识本地 worker 索引的整数 ID（介于 0 和 local_world_size-1 之间）。例如，如果应用在 2 个节点上各派生 4 个进程，则每个节点上都有本地 rank 为 0、1、2、3 的 worker。注意，运行单节点多进程分布式应用时，一个进程的本地 rank 等于其全局 rank。

图 3：一个运行在 2 个节点上、world size 为 8 的分布式应用示意图。每个 worker 进程由全局 rank（0 到 7）和本地 rank（0 到 3）标识。图取自 https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html 。（图见英文原版 PDF。）

#### 5.1.1 分布式应用基准测试的最佳实践

在本部分作业中，你将对分布式应用做基准测试，以更好地理解通信带来的开销。以下是一些最佳实践：

- 尽可能在同一台机器上运行基准测试，以便进行受控比较。
- 在对目标操作计时之前，先执行若干预热步骤。这对 NCCL 通信调用尤其重要。一般 5 次预热迭代就够了。
- 在 GPU 上做基准测试时，调用 `torch.cuda.synchronize()` 等待 CUDA 操作完成。注意，即使以 `async_op=False` 调用通信操作，这也是必要的——它在操作被排入 GPU 队列时返回（而不是在通信实际完成时）。³
- 不同 rank 上的计时可能略有差异，因此常见的做法是跨 rank 聚合测量结果以获得更准的估计。你可能会发现 all-gather 集合操作（具体是 `dist.all_gather_object` 函数）对收集所有 rank 的结果很有用。
- 一般来说，先在本地用 Gloo 在 CPU 上调试，然后按题目要求用 NCCL 在 GPU 上做基准测试。

在后端之间切换应当很简单，只需修改 `init_process_group` 调用和张量的设备转换。

---

³ 详见 https://github.com/pytorch/pytorch/issues/68112#issuecomment-965932386 。

---

**问题 (distributed_communication_single_node)：分布式通信（单节点）（5 分）**

编写一个脚本，在单节点多进程设置下对 all-reduce 操作的运行时间做基准测试。上面的示例代码可以作为一个合理的起点。尝试变化以下设置：

- **all-reduce 数据大小**：float32 数据张量，取 1MB、10MB、100MB、1GB。
- **GPU/进程数量**：2、4 或 6。

资源要求：至多 6 块 GPU。每次基准测试运行应少于 5 分钟。

**提交要求：** 比较各种设置的图和/或表格，外加 2-3 句话的评论，谈谈你的结果以及你对各因素如何相互作用的思考。

### 5.2 分布式数据并行训练的朴素实现

我们已经了解了编写 PyTorch 分布式应用的基础，现在来构建一个最小的分布式数据并行（distributed data parallel，DDP）训练实现。

数据并行把批次（batch）切分到多个设备（如 GPU）上，使得单设备放不下的更大的批量大小也能训练。例如，给定四个各自最多能处理批量大小 32 的设备，数据并行训练可以实现 128 的有效批量大小。

以下是朴素地做分布式数据并行训练的步骤。最初，每个设备各自构建一个（随机初始化的）模型。我们用 broadcast 集合通信操作把模型参数从 rank 0 发送到所有其他 rank。训练开始时，每个设备都持有一份相同的模型参数和优化器状态（例如 Adam 中累积的梯度统计量）副本。

1. 给定一个有 $n$ 个样本的批次，把批次分片，每个设备收到 $n/d$ 个互不重叠的样本（其中 $d$ 是用于数据并行训练的设备数）。$d$ 应整除 $n$，否则一些 rank 会比其他 rank 做更多工作，整个步骤会被最慢的拖累。
2. 每个设备用本地的模型参数副本，对自己的 $n/d$ 个样本运行前向传播和反向传播，计算梯度。注意，此时每个设备持有的是基于它收到的 $n/d$ 个样本算出的梯度。
3. 然后用 all-reduce 集合通信操作对不同设备的梯度取平均，使每个设备都持有按全部 $n$ 个样本平均的梯度。
4. 接下来，每个设备运行一次优化器步骤，更新自己的参数副本——从优化器的视角看，它只是在优化一个本地模型。由于所有设备都从相同的初始模型和优化器状态出发，并且每次迭代使用相同的平均梯度，参数和优化器状态在所有设备上会保持同步。至此，我们完成了一次训练迭代，可以重复这个过程。

**问题 (naive_ddp)：朴素 DDP（5 分）**

**提交要求：** 实现一种朴素形式的分布式数据并行训练：在反向传播之后对各个参数梯度逐一做 all-reduce。为测试你的实现，实现 `[adapters.get_ddp]` 和（可选的）`[adapters.ddp_on_after_backward]`，然后运行 `uv run pytest tests/test_ddp.py`。

**问题 (naive_ddp_benchmarking)：朴素 DDP 基准测试（3 分）**

在这个朴素 DDP 实现中，参数梯度在每次反向传播后跨 rank 逐一做 all-reduce。为更好地理解数据并行训练的开销，编写一个脚本，对用该朴素 DDP 实现训练的、你之前实现的语言模型做基准测试。测量每个训练步骤的总时间，以及花在梯度通信上的时间占比。在单节点设置（1 个节点、2 块 GPU）下，对 2.1.2 节描述的 xl 模型规模收集测量数据。

**提交要求：** 描述你的基准测试设置，并给出每种设置下实测的每次训练迭代时间和花在梯度通信上的时间。

### 5.3 改进最小 DDP 实现

我们在 5.2 节看到的最小 DDP 实现有几个关键局限：

1. 它为每个参数张量单独做一次 all-reduce。每次通信调用都有开销，因此把通信调用合并成批以降低开销可能是有利的。
2. 它等反向传播结束后才通信梯度。然而，反向传播是增量计算的。因此，当某个参数的梯度就绪时，可以立即通信它，而不必等待其他参数的梯度。这让我们能把梯度通信与反向传播计算重叠（overlap），降低分布式数据并行训练的开销。

在本部分作业中，我们将逐一解决这些局限，并测量对训练速度的影响。

#### 5.3.1 减少通信调用次数

与其为每个参数张量发起一次通信调用，我们来看看把 all-reduce 合并成批能否改善性能。具体来说，我们把要做 all-reduce 的梯度拼接成单个张量，然后在所有 rank 上对合并后的梯度做一次 all-reduce。使用 `torch._utils._flatten_dense_tensors` 和 `torch._utils._unflatten_dense_tensors` 可能会有帮助。

**问题 (minimal_ddp_flat_benchmarking)：扁平化梯度的最小 DDP 基准测试（2 分）**

修改你的最小 DDP 实现，改为通信一个由所有参数的梯度扁平化而成的张量。在之前使用的条件下（1 个节点、2 块 GPU，2.1.2 节描述的 xl 模型规模），比较它与为每个参数张量各发一次 all-reduce 的最小 DDP 实现的性能。

**提交要求：** 在分布式数据并行训练下，使用单次批量 all-reduce 调用时实测的每次训练迭代时间和花在梯度通信上的时间。用 1-2 句话比较批量与逐一通信梯度的结果。

#### 5.3.2 计算与逐参数梯度通信的重叠

虽然把通信调用合并成批可能有助于降低发起大量小 all-reduce 操作的开销，但所有通信时间仍直接构成开销。为解决这个问题，我们可以利用一个观察：反向传播逐层增量地计算梯度（从损失开始向输入方向推进）——因此，参数梯度一就绪就可以做 all-reduce，通过把反向传播计算与梯度通信重叠来降低分布式数据并行训练的开销。

我们先实现一个分布式数据并行包装器（wrapper）并做基准测试：它在反向传播期间，随着各个参数张量就绪，异步地对它们做 all-reduce。以下指引可能有用：

- **反向钩子（backward hook）**：要在某个参数的梯度于反向传播中累积完成后自动调用一个函数，可以使用 `register_post_accumulate_grad_hook` 函数。⁴
- **异步通信**：所有 PyTorch 集合通信操作都支持同步（`async_op=False`）和异步（`async_op=True`）执行。同步调用会阻塞，直到集合操作被排入 GPU 队列。这并不意味着 CUDA 操作已完成，因为 CUDA 操作是异步的。话虽如此，之后使用该输出的函数调用会表现得符合预期。⁵ 相比之下，异步调用会返回一个分布式请求句柄（handle）——因此函数返回时，集合通信操作不保证已排入 GPU 队列，更不保证已完成。要等待操作排入 GPU 队列（从而使输出可用于后续操作），可以对返回的通信句柄调用 `handle.wait()`。

例如，下面两个例子分别以同步和异步调用对一个张量列表中的每个张量做 all-reduce：

```python
tensors = [torch.rand(5) for _ in range(10)]
# Synchronous, block until operation is queued on the GPU.
for tensor in tensors:
    dist.all_reduce(tensor, async_op=False)

# Asynchronous, return immediately after each call and
# wait on results at the end.
handles = []
for tensor in tensors:
    handle = dist.all_reduce(tensor, async_op=True)
    handles.append(handle)
# ...
# Possibly execute other commands that don't rely on the all_reduce results
# ...
# Ensure that all-reduce calls were queued and
# therefore other operations depending on the
# all-reduce output can be queued.
for handle in handles:
    handle.wait()
handles.clear()
```

---

⁴ 更多信息和用法示例见 https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html 。

⁵ 在高级场景中，如果你使用多个 CUDA 流（stream），可能需要跨流的显式同步，以确保输出已就绪可供后续操作使用。见 https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams 。

---

**问题 (ddp_overlap_individual_parameters)：逐参数重叠的 DDP（5 分）**

实现一个处理分布式数据并行训练的 Python 类。该类应包装任意 PyTorch `nn.Module`，负责在训练前广播权重（使所有 rank 有相同的初始参数），并发起用于梯度平均的通信调用。我们推荐以下公开接口：

- `def __init__(self, module: torch.nn.Module)`：给定一个已实例化、待并行化的 PyTorch `nn.Module`，构造一个负责跨 rank 梯度同步的 DDP 容器。
- `def forward(self, *inputs, **kwargs)`：用提供的位置参数和关键字参数调用被包装模块的 `forward()` 方法。
- `def finish_gradient_synchronization(self)`：调用时，等待异步通信调用在 GPU 上完成。

要用这个类进行分布式训练，我们把一个模块交给它包装，然后在运行 `optimizer.step()` 之前调用 `finish_gradient_synchronization()`，以确保优化器步骤这个依赖梯度的操作可以安全地排队：

```python
model = ToyModel().to(device)
ddp_model = DDP(model)
for _ in range(train_steps):
    x, y = get_batch()
    logits = ddp_model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()
```

**提交要求：** 实现一个处理分布式数据并行训练的容器类。该类应把梯度通信与反向传播计算重叠。为测试你的 DDP 类，先实现适配器 `[adapters.get_ddp]` 和 `[adapters.ddp_on_after_backward]`（后者可选，取决于你的实现，可能不需要）。

然后运行 `uv run pytest tests/test_ddp.py` 执行测试。我们建议多运行几次（例如 5 次）测试，以确保它能稳定通过。

**问题 (ddp_overlap_individual_parameters_benchmarking)：逐参数重叠 DDP 基准测试（1 分）**

(a) 对你的 DDP 实现在反向传播计算与逐参数梯度通信重叠时的性能做基准测试。在相同设置下（1 个节点、2 块 GPU、2.1.2 节描述的 xl 模型规模）与之前研究过的设置比较性能（为每个参数张量各发一次 all-reduce 的最小 DDP 实现，以及对所有参数张量拼接结果做单次 all-reduce 的实现）。

**提交要求：** 重叠反向传播与逐参数梯度通信时实测的每次训练迭代时间，外加 1-2 句话的结果比较。

(b) 用 Nsight 性能分析器对你的基准测试代码（1 个节点、2 块 GPU、xl 模型规模的设置）插桩，比较最初的 DDP 实现与这个重叠实现。直观地比较两份跟踪结果，并提供性能分析器截图，展示一个实现把计算与通信重叠了而另一个没有。

**提交要求：** 2 张截图（一张来自最初的 DDP 实现，另一张来自这个把计算与通信重叠的 DDP 实现），直观地展示通信是否与反向传播重叠。

## 6 优化器状态分片

分布式数据并行训练概念简单且通常非常有效，但要求每个 rank 各持有一份模型参数和优化器状态的副本。这种冗余会带来可观的内存成本。例如，AdamW 优化器为每个参数维护两个浮点数，意味着它消耗的内存是模型权重的两倍。S. Rajbhandari 等人 [5] 描述了几种在数据并行训练中降低这种冗余的方法：把（1）优化器状态、（2）梯度和（3）参数划分到不同 rank 上，并在 worker 之间按需通信。

在本部分作业中，我们将通过实现一个简化版的优化器状态分片（optimizer state sharding）来降低每个 rank 的内存消耗。每个 rank 的优化器实例不再保存所有参数的优化器状态，而是只处理参数的一个子集（约 1 / world_size）。当每个 rank 的优化器执行优化器步骤时，它只更新自己分片中的那部分模型参数。然后，每个 rank 把自己更新后的参数广播给其他 rank，确保模型参数在每次优化器步骤后保持同步。

**问题 (optimizer_state_sharding)：优化器状态分片（15 分）**

实现一个处理优化器状态分片的 Python 类。该类应包装任意输入的 PyTorch `optim.Optimizer`，并负责在每次优化器步骤后同步更新后的参数。我们推荐以下公开接口：

- `def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any)`：初始化分片状态优化器。`params` 是待优化参数的集合（或参数组，当用户想对模型的不同部分使用不同超参数如学习率时）；这些参数将被分片到所有 rank。`optimizer_cls` 参数指定要包装的优化器类型（如 `optim.AdamW`）。最后，其余关键字参数都转发给 `optimizer_cls` 的构造函数。确保在此方法中调用 `torch.optim.Optimizer` 超类构造函数。
- `def step(self, closure, **kwargs)`：用提供的 closure 和关键字参数调用被包装优化器的 `step()` 方法。更新参数后，与其他 rank 同步。
- `def add_param_group(self, param_group: dict[str, Any])`：该方法应向分片优化器添加一个参数组。它在分片优化器构造期间由超类构造函数调用，也可能在训练期间被调用（例如逐步解冻模型中的层）。因此，该方法应负责在各 rank 之间分配模型的参数。

**提交要求：** 实现一个处理优化器状态分片的容器类。为测试你的分片优化器，先实现适配器 `[adapters.get_sharded_optimizer]`。然后运行 `uv run pytest tests/test_sharded_optimizer.py` 执行测试。我们建议多运行几次（例如 5 次）测试，以确保它们能稳定通过。

现在我们已经实现了优化器状态分片，接下来分析它对训练期间峰值内存用量的影响和它的运行时开销。

**问题 (optimizer_state_sharding_accounting)：优化器状态分片核算（5 分）**

(a) 编写一个脚本，分析使用与不使用优化器状态分片训练语言模型时的峰值内存用量。使用标准配置（1 个节点、2 块 GPU、xl 模型规模），报告模型初始化之后、优化器步骤刚要执行之前、以及优化器步骤刚结束之后的峰值内存用量。结果符合你的预期吗？分解每种设置下的内存使用（例如，参数占多少内存、优化器状态占多少内存等）。

**提交要求：** 用 2-3 句话给出峰值内存用量结果，并分解内存在不同模型与优化器组件之间的分布。

(b) 我们的优化器状态分片实现对训练速度有何影响？在标准配置（1 个节点、2 块 GPU、xl 模型规模）下，测量使用与不使用优化器状态分片时每次迭代的耗时。

**提交要求：** 用 2-3 句话给出你的计时结果。

(c) 我们的优化器状态分片方法与 ZeRO stage 1（在 S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He [5] 中记为 ZeRO-DP $P_{os}$）有何不同？

**提交要求：** 用 2-3 句话总结任何差异，尤其是与内存和通信量相关的差异。

## 7 完全分片数据并行

有了优化器状态分片和数据并行，我们可以把优化器状态和激活沿数据并行轴切分。然而，模型权重仍是冗余的——我们在每块 GPU 上都存了一份完整副本。

我们可以通过把数据并行（DP）轴变成完全分片数据并行（fully-sharded data parallel，FSDP）轴来解决这个问题。在 FSDP 下，每块 GPU 只存每个权重张量自己的一片，但在准备前向或反向传播时，必须用 all-gather 从其他 GPU 拉取分片、拼成完整权重张量。

为避免 GPU 计算空等通信完成，大多数 FSDP 实现会提前调度层的 all-gather，即相关权重在被需要之前就绪，防止通信阻塞计算。这把权重分片通信挪出关键路径，意味着只要通信能跟上计算且调度得当，它就没有成本。

有些层在内存和计算上都足够小，一次传输的延迟开销不值得。你应当标记这些层不做 FSDP 分片。在我们的架构中，这主要是各种 norm。这样就剩下嵌入层和每个线性层。

虽然必须以 FP32 存储主权重（master weight）（任何会被反复累加的值都对精度敏感），但权重本身不需要以 FP32 使用。在混合精度中，我们总是会在使用前转换为低精度计算数据类型，因此不妨在通信权重之前就转换，以节省带宽。

**问题 (fsdp)：完全分片数据并行（15 分）**

实现一个用于完全分片数据并行训练的 Python 类。该类应包装任意 PyTorch `nn.Module`（你的完整模型），并钩入（hook）或包装其中的任何 Linear 或 Embedding 层。我们推荐以下公开接口：

- `def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None)`：给定一个已实例化、待并行化的 PyTorch `nn.Module`，构造一个负责权重 all-gather 和梯度 reduce-scatter 的 FSDP 模块。确保你的钩子或模块包装器及时为前向传播 all-gather 好权重。为限制内存使用，仅当当前层往前数第二层完成其前向传播之后才开始 gather。在反向传播中，你的钩子或模块包装器应做 all-gather，使权重可用于计算。当梯度可用时，应把它们 reduce-scatter 到相应的 rank。确保在使用后释放 gather 来的权重。当提供了 `compute_dtype` 时，在通信或用于计算之前把权重转换为该 dtype，同时保持主权重和优化器更新为 FP32。
- `def forward(self, *inputs, **kwargs)`：用提供的位置参数和关键字参数调用被包装模块的 `forward()` 方法。
- `def finish_gradient_synchronization(self)`：调用时，等待异步通信调用在 GPU 上完成。

**提交要求：** 实现一个处理完全分片数据并行训练的容器类。该容器的每个分片都应与作业 1 的标准 AdamW 实现兼容。为测试你的 FSDP 实现，实现适配器 `[adapters.get_fsdp]`。运行 `uv run pytest tests/test_fsdp.py` 执行测试。我们建议多运行几次（例如 5 次）测试，以捕获任何竞态条件（race condition）。

**问题 (fsdp_accounting)：FSDP 核算（5 分）**

(a) 根据你在第 6 节的分析，实现 FSDP 预计能从峰值内存中省下多少？计算中可以忽略为把权重 all-gather 到每块 GPU 所需的预分配缓冲区的大小。

**提交要求：** 用 2-3 句话给出你的发现。

(b) 在两块 GPU 上对 xl 模型做性能分析，关注权重的 all-gather。通信是否及时完成，赶上了前向传播？

**提交要求：** 用 2-3 句话给出你的计时结果。附上 Nsight 截图支撑你的结论。

## 8 并行策略分析

我们还可以沿更多轴来并行化训练过程。一些常见策略包括：

- **数据并行（data parallelism，DP）**：把数据批次切分到多个设备上，每个设备为自己的批次计算梯度。然后跨设备对梯度取平均。
- **完全分片数据并行（fully-sharded data parallelism，FSDP）**：在数据并行之上，还把优化器状态、梯度和权重切分到不同设备以降低内存占用。设备随后需要在前向和反向传播期间从其他设备收集权重分片。
- **张量并行（tensor parallelism，TP）**：沿输入或输出维度切分权重矩阵。各设备计算自己分片对应的激活，然后激活跨设备归约或收集。
- **流水线并行（pipeline parallelism，PP）**：把模型按层切分为多个阶段（stage），每个阶段在不同设备上运行。
- **专家并行（expert parallelism，EP）**：把混合专家（Mixture-of-Experts）模型中的专家切分到不同设备上，每个设备计算自己专家的输出。

在本节中，我们将在简化设定下做一些基本计算，以便在并行策略之间做选择并决定如何组合它们。首先聚焦 DP、FSDP、TP 及它们的组合。我们的方法是计算每种策略的通信成本，并与计算成本比较，这告诉我们在通信成本成为瓶颈之前可以扩展到多少设备。

关于 TPU/GPU 拓扑与并行策略更详细的论述，《TPU Scaling Book》（J. Austin et al. [6]）是很好的资料。关于流水线并行更详细的讨论，见《The Ultra-Scale Playbook》附录（H. Z. P. N. M. M. L. W. T. W. Nouamane Tazi Ferdinand Mom [7]）。这两本书的其余部分也有很多你可能觉得有用的信息。

### 8.1 通信原语

第一步是理解通信原语（communication primitive）。在我们的简化设定中，假设有 $N$ 个设备，编号 $0, \dots, N-1$，每对设备之间由一条链路连接。我们还假设每个设备有 $W$ 的出口（egress，即外出）带宽；换句话说，每个设备能以每秒 $W$ 字节的速率向另一个设备发送数据。我们会如何实现 gather 和 reduce？

实现 all-gather 的一种常见方式是环形 all-gather（ring all-gather）。回忆一下，在 all-gather 中，每个设备 $i$ 起初持有一个大小为 $\frac{S}{N}$ 的分块 $x_i$，最终持有大小为 $S$（字节）的完整 $x = [x_0, \dots, x_{N-1}]$。在环形 all-gather 中，我们把设备排成一个环。每一步，每个设备把自己当前的分块发送给右边的下一个设备，并存储从左边设备收到的分块。这个过程重复进行：每个设备把刚收到的分块传给右边，并从左边收到一个新分块。经过 $N-1$ 步后，每个设备都有完整张量。

在我们的理想化设定中，每个设备在每一步同时传输一个大小为 $\frac{S}{N}$ 的分块，出口带宽为 $W$，共有 $N-1$ 步，因此环形 all-gather 耗时 $\frac{N-1}{N} \cdot \frac{S}{W}$ 秒。

接下来分析环形 reduce-scatter。在 reduce-scatter 中，每个设备 $i$ 起初持有一个大小为 $S$ 的完整张量 $x^{(i)}$。我们想计算归约 $y = \sum_{i=0}^{N-1} x^{(i)}$，但每个设备 $i$ 最终只持有一个大小为 $\frac{S}{N}$ 的分块 $y_i$。与环形 all-gather 一样，我们先把设备排成一个环。每个设备先把自己的张量 $x^{(i)}$ 分成 $N$ 个分块 $[x^{(i)}_0, \dots, x^{(i)}_{N-1}]$，每个大小为 $\frac{S}{N}$。然后像环形 all-gather 一样传递分块，只是在把分块传下去之前，每个设备把自己的贡献加进分块（分块中存的是部分和）。具体来说：

在第 $t = 1, \dots, N-1$ 步中，设备 $i$ 执行以下操作：

- 若 $t = 1$，初始化 $y \leftarrow x^{(i)}$，它存储目前为止的部分和；
- 把分块 $y_{(i-t) \bmod N}$ 发送给设备 $(i+1) \bmod N$；
- 从设备 $(i-1) \bmod N$ 接收分块 $z_{(i-t-1) \bmod N}$；
- 更新你的部分和副本：$y_{(i-t-1) \bmod N} \leftarrow y_{(i-t-1) \bmod N} + z_{(i-t-1) \bmod N}$。

经过 $N-1$ 步后，设备 $i$ 拥有分块 $y_i$ 的完整求和结果，因此环形 reduce-scatter 耗时 $\frac{N-1}{N} \cdot \frac{S}{W}$ 秒，与环形 all-gather 相同。

最后实现环形 all-reduce。在 all-reduce 中，每个设备 $i$ 起初持有大小为 $S$ 的完整张量 $x^{(i)}$，最终持有归约结果 $y = \sum_{i=0}^{N-1} x^{(i)}$。我们把 all-reduce 实现为一次环形 reduce-scatter 接一次环形 all-gather，因此环形 all-reduce 耗时 $2 \cdot \frac{N-1}{N} \cdot \frac{S}{W}$ 秒。

**问题 (alternate_ring_all_reduce)：另一种环形 all-reduce（1 分）**

不把 all-reduce 实现为一次环形 reduce-scatter 接一次环形 all-gather，而是使用如下算法：

在第 $t = 1, \dots, N-1$ 步中，设备 $i$ 执行以下操作：

- 若 $t = 1$，初始化 $y \leftarrow x^{(i)}$，它存储目前为止的部分和；
- 把 $x^{((i-t+1) \bmod N)}$ 发送给设备 $(i+1) \bmod N$；
- 从设备 $(i-1) \bmod N$ 接收 $x^{((i-t) \bmod N)}$；
- 更新你的部分和副本：$y \leftarrow y + x^{((i-t) \bmod N)}$。

在与上文相同的设定下（每个设备出口带宽 $W$，每个 $x^{(i)}$ 大小为 $S$），这个算法耗时多久？

**提交要求：** 用 $S$、$N$ 和 $W$ 表示的答案，外加一句话的论证。

### 8.2 数据并行分析

有了通信原语，我们就可以分析并行策略了。我们将分析单个 FFN 层的并行化。回忆一下，给定输入 $x$，前向传播为：

$$x_1 = xW_1 \tag{20}$$

$$x_2 = xW_2 \tag{21}$$

$$z = f(x_1) * x_2 \tag{22}$$

$$y = zW_3, \tag{23}$$

其中 $x$ 的形状为 $(B, D)$，$W_1$ 和 $W_2$ 的形状为 $(D, D_{\mathrm{FF}})$，$W_3$ 的形状为 $(D_{\mathrm{FF}}, D)$。$f$ 是逐元素激活函数（如 SiLU），$*$ 表示逐元素乘法。

把反向传播也显式写出来会很有用。回忆一下，给定形状为 $(B, D)$ 的 $dy$，反向传播为：

$$dz = dy\, W_3^\top \tag{24}$$

$$dx_2 = dz * f(x_1) \tag{25}$$

$$dx_1 = dz * f'(x_1) * x_2 \tag{26}$$

$$dx = dx_1 W_1^\top + dx_2 W_2^\top \tag{27}$$

$$dW_3 = z^\top dy \tag{28}$$

$$dW_2 = x^\top dx_2 \tag{29}$$

$$dW_1 = x^\top dx_1, \tag{30}$$

其中 $*$ 表示逐元素乘法。

回忆一下，在有 $N_{\mathrm{DP}}$ 个设备的数据并行中，我们把输入 $x$ 分片为大小为 $\left(\frac{B}{N_{\mathrm{DP}}}, D\right)$ 的分片 $x^{(i)}$。前向传播照常进行，不需要任何集合通信，产生大小为 $\left(\frac{B}{N_{\mathrm{DP}}}, D\right)$ 的激活 $y^{(i)}$。在反向传播中，按 batch 分片的激活照常计算，设备 $i$ 最终得到梯度

$$dW_3^{(i)} = z^{(i)\top} dy^{(i)} \tag{31}$$

$$dW_2^{(i)} = x^{(i)\top} dx_2^{(i)} \tag{32}$$

$$dW_1^{(i)} = x^{(i)\top} dx_1^{(i)}, \tag{33}$$

其中我们不是对全部 $B$ 个外积求和，而是只对属于我们输入分片的 $\frac{B}{N_{\mathrm{DP}}}$ 个外积求部分和。然后，需要跨设备做一次 all-reduce 才能得到完整梯度 $dW_3$、$dW_2$ 和 $dW_1$。

**问题 (data_parallel_calcs)：数据并行计算（3 分）**

现在我们已经具备计算数据并行何时会被通信卡脖子的一切要素。令 $C$（单位 FLOP/s）表示设备的加速器速度，$W$（单位字节/秒）表示每个设备的出口带宽。然后我们就能计算计算时间和通信时间。由于计算与通信可以重叠，当通信时间大于计算时间时我们就遇到了瓶颈。我们假设所有权重和激活都是 FP16（即两字节）。

(a) 在有 $N_{\mathrm{DP}}$ 路数据并行时，计算反向传播需要多少 FLOP？可以忽略所有非矩阵乘法操作。回忆一下，矩阵乘法 $(A, B) \times (B, C) \to (A, C)$ 需要 $2ABC$ 次浮点运算。

**提交要求：** 用 $B$、$D$、$D_{\mathrm{FF}}$ 和 $N_{\mathrm{DP}}$ 表示的答案，外加一句话的论证。

(b) 在有 $N_{\mathrm{DP}}$ 路数据并行时，反向传播需要多少通信时间？

**提交要求：** 用 $B$、$D$、$D_{\mathrm{FF}}$、$N_{\mathrm{DP}}$ 和 $W$ 的一个子集表示的答案，外加一句话的论证。

(c) 固定其他参数，$N_{\mathrm{DP}}$ 最大可以到多少才会被通信卡脖子？

**提交要求：** 一个不等式，一侧是 $N_{\mathrm{DP}}$，另一侧是用 $B$、$D$、$D_{\mathrm{FF}}$、$C$ 和 $W$ 的一个子集表示的表达式，外加一句话的论证。

### 8.3 完全分片数据并行分析

接下来分析 FSDP。回忆一下，与 DP 一样，FSDP 切分输入和激活的 batch 维度。此外，为节省内存，我们还切分优化器状态、梯度和权重。我们可以沿任一维度切分权重，在设备 $i$ 上得到分片 $W_1^{(i)}$、$W_2^{(i)}$ 和 $W_3^{(i)}$，每个大小为 $\frac{D \cdot D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$。

在前向传播中，我们只需在按 batch 分片的输入上做数据并行前向传播。但为此，回忆一下，我们需要先跨设备 all-gather 权重分片：

$$W_1 = \text{all-gather}\left(\{W_1^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{34}$$

$$W_2 = \text{all-gather}\left(\{W_2^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{35}$$

$$W_3 = \text{all-gather}\left(\{W_3^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{36}$$

$$\text{（执行按 batch 分片的前向传播）} \tag{37}$$

注意，何时做 all-gather 我们有一定的自由：只要在权重被使用之前完成 all-gather 即可，之后应将其丢弃以保持较低的内存成本。为简单起见，本节把三次 all-gather 列在一起。

与前向一样，在反向传播中我们也需要先跨设备 all-gather 权重分片。然后可以做数据并行反向传播，只是不再需要 all-reduce，因为每个设备只需要自己分片的梯度。因此，我们改用 reduce-scatter：

$$W_1 = \text{all-gather}\left(\{W_1^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{38}$$

$$W_2 = \text{all-gather}\left(\{W_2^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{39}$$

$$W_3 = \text{all-gather}\left(\{W_3^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{40}$$

$$\text{（执行按 batch 分片的反向传播）} \tag{41}$$

$$dW_1^{(i)} = \text{reduce-scatter}\left(\{dW_1^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{42}$$

$$dW_2^{(i)} = \text{reduce-scatter}\left(\{dW_2^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{43}$$

$$dW_3^{(i)} = \text{reduce-scatter}\left(\{dW_3^{(i)}\}_{i=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{44}$$

注意，分片记号（上标 $(i)$）在这里是重载的：reduce-scatter 的输入是完整权重上的部分和，而输出是分片权重上的完整和。

**问题 (fsdp_calcs)：完全分片数据并行计算（3 分）**

在与数据并行计算相同的设定下，我们来计算 FSDP 何时会被通信卡脖子。

(a) 在有 $N_{\mathrm{FSDP}}$ 路 FSDP 时，计算反向传播需要多少 FLOP？前向传播呢？

**提交要求：** 两个用 $B$、$D$、$D_{\mathrm{FF}}$ 和 $N_{\mathrm{FSDP}}$ 表示的答案，外加两句话的论证（各一句）。

(b) 在有 $N_{\mathrm{FSDP}}$ 路 FSDP 时，反向传播需要多少通信时间？前向传播呢？

**提交要求：** 两个用 $B$、$D$、$D_{\mathrm{FF}}$、$N_{\mathrm{FSDP}}$ 和 $W$ 的一个子集表示的答案，外加两句话的论证（各一句）。

(c) 固定其他参数，$N_{\mathrm{FSDP}}$ 最大可以到多少，反向传播才会被通信卡脖子？前向传播呢？

**提交要求：** 两个不等式，一侧是 $N_{\mathrm{FSDP}}$，另一侧是用 $B$、$D$、$D_{\mathrm{FF}}$、$C$ 和 $W$ 的一个子集表示的表达式，外加两句话的论证（各一句）。

### 8.4 张量并行分析

实践中，FSDP 常与一种称为张量并行（tensor parallelism，TP）的并行策略结合。在 TP 中，我们沿每个权重矩阵的输入或输出维度把权重切分到不同设备。沿输入维度切分常称为“行并行”（row parallel），沿输出维度切分常称为“列并行”（column parallel）。

具体来说，假设我们要切分矩阵乘法 $xW$，其中 $x$ 形状为 $(B, D)$，$W$ 形状为 $(D, D_{\mathrm{FF}})$。在列并行中，我们有形状为 $\left(D, \frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\right)$ 的分片 $W^{(i)}$，并且有

$$xW = \text{all-gather}\left(\{xW^{(i)}\}_{i=0}^{N_{\mathrm{TP}}-1}\right). \tag{45}$$

另一方面，在行并行中，我们有形状为 $\left(\frac{D}{N_{\mathrm{TP}}}, D_{\mathrm{FF}}\right)$ 的分片 $W^{(i)}$，每个设备还要在做矩阵乘法之前把输入 $x$ 收窄为形状为 $\left(B, \frac{D}{N_{\mathrm{TP}}}\right)$ 的切片 $x^{(i)}$。于是有

$$xW = \text{all-reduce}\left(\{x^{(i)} W^{(i)}\}_{i=0}^{N_{\mathrm{TP}}-1}\right). \tag{46}$$

为了并行化我们的 FFN，我们使用一种特定的张量并行配置：$W_1$ 和 $W_2$ 用列并行（按输出维度切分），而 $W_3$ 用行并行（按输入维度切分）。由于行并行权重只需要输入的一个切片，这种配置让我们可以省掉列并行权重之后的 all-gather。该策略给出如下的前向传播（给定大小为 $(B, D)$ 的输入 $x$）：

$$x_1^{(i)} = xW_1^{(i)} \tag{47}$$

$$x_2^{(i)} = xW_2^{(i)} \tag{48}$$

$$z^{(i)} = f\left(x_1^{(i)}\right) * x_2^{(i)} \tag{49}$$

$$y^{(i)} = z^{(i)} W_3^{(i)} \tag{50}$$

$$y = \text{all-reduce}\left(\{y^{(i)}\}_{i=0}^{N_{\mathrm{TP}}-1}\right), \tag{51}$$

其中 $W_1^{(i)}$ 和 $W_2^{(i)}$ 形状为 $\left(D, \frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\right)$，$W_3^{(i)}$ 形状为 $\left(\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}, D\right)$。

**问题 (tp_calcs)：张量并行计算（4 分）**

在与 DP 和 FSDP 计算相同的设定下，我们来计算 TP 何时会被通信卡脖子。

(a) 给定大小为 $(B, D)$ 的输入 $dy$，写出上述张量并行策略的反向传播（其中 $W_1^{(i)}$ 和 $W_2^{(i)}$ 形状为 $\left(D, \frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\right)$，$W_3^{(i)}$ 形状为 $\left(\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}, D\right)$）。

**提交要求：** 一组描述反向传播的方程，用 $dy$、分片权重（$W_1^{(i)}$、$W_2^{(i)}$、$W_3^{(i)}$）、前向传播保存的激活（$x$、$x_1^{(i)}$、$x_2^{(i)}$、$z^{(i)}$、$y^{(i)}$）、通信原语以及你想定义的任何中间变量表示。这些方程应产出每个设备的梯度 $dW_1^{(i)}$、$dW_2^{(i)}$、$dW_3^{(i)}$ 和反向传播输出 $dx$。可以随意引用 8.2 节中未分片的反向传播并加以修改。

(b) 在有 $N_{\mathrm{TP}}$ 路 TP 时，计算前向传播需要多少 FLOP？反向传播呢？

**提交要求：** 两个用 $B$、$D$、$D_{\mathrm{FF}}$ 和 $N_{\mathrm{TP}}$ 表示的答案，外加两句话的论证（各一句）。

(c) 在有 $N_{\mathrm{TP}}$ 路 TP 时，前向传播需要多少通信时间？反向传播呢？

**提交要求：** 两个用 $B$、$D$、$D_{\mathrm{FF}}$、$N_{\mathrm{TP}}$ 和 $W$ 的一个子集表示的答案，外加两句话的论证（各一句）。

(d) 固定其他参数，$N_{\mathrm{TP}}$ 最大可以到多少，反向传播才会被通信卡脖子？前向传播呢？

**提交要求：** 两个不等式，一侧是 $N_{\mathrm{TP}}$，另一侧是用 $B$、$D$、$D_{\mathrm{FF}}$、$C$ 和 $W$ 的一个子集表示的表达式，外加两句话的论证（各一句）。

### 8.5 二维并行（FSDP + TP）

我们终于可以组合并行策略了！在本节中，我们将研究如何以最优方式组合 FSDP 和 TP。提示：在前面几节中，你应该已经发现，批量大小和模型规模参数限制了你可以扩展到的设备数量，把这些都变大可以让你继续扩展设备而不被通信卡脖子。不幸的是，批量大小超过某个点后，性能会开始下降，因为梯度噪声显著缩小，失去 SGD 的隐式正则化特性；这个点常被称为“临界批量大小”（critical batch size）。而缩放定律（scaling law）通常会告诉我们想要多大的模型（这将是你在下一次作业中的任务！）。

在本节中，我们考虑一个简化设定：有人带着所有问题参数（批量大小、模型规模、带宽、加速器速度）来找你。你的任务是选择一种 FSDP 和 TP 的配置，使其能扩展到尽可能多的设备，同时保持受计算限制（compute-bound）而非受通信限制（communication-bound）。

我们先过一遍把 FSDP 与 TP 结合的机制。每个设备将有一个 TP rank $i = 0, \dots, N_{\mathrm{TP}} - 1$ 和一个 FSDP rank $j = 0, \dots, N_{\mathrm{FSDP}} - 1$，构成一个总共 $N = N_{\mathrm{TP}} N_{\mathrm{FSDP}}$ 个设备的二维网格。按照 TP，我们首先沿输出维度切分 $W_1$ 和 $W_2$，沿输入维度切分 $W_3$。结果是，我们必须在激活上插入一次 TP 风格的 all-reduce。接下来应用 FSDP，我们切分输入的 batch 维度，并沿 TP 未切分的那个维度进一步切分每个权重矩阵。然后，我们必须在 TP 风格的前向/反向传播之前对权重插入 FSDP 风格的 all-gather，并在 TP 风格的反向传播之后对权重梯度插入 reduce-scatter。

结果是每个设备 $(i, j)$ 持有权重分片 $W_1^{(i,j)}$ 和 $W_2^{(i,j)}$，形状为 $\left(\frac{D}{N_{\mathrm{FSDP}}}, \frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\right)$，以及 $W_3^{(i,j)}$，形状为 $\left(\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}, \frac{D}{N_{\mathrm{FSDP}}}\right)$。给定大小为 $\left(\frac{B}{N_{\mathrm{FSDP}}}, D\right)$ 的按 batch 分片输入 $x^{(j)}$，我们可以把前向传播写成：

$$W_1^{(i)} = \text{all-gather}\left(\{W_1^{(i,j)}\}_{j=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{52}$$

$$W_2^{(i)} = \text{all-gather}\left(\{W_2^{(i,j)}\}_{j=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{53}$$

$$W_3^{(i)} = \text{all-gather}\left(\{W_3^{(i,j)}\}_{j=0}^{N_{\mathrm{FSDP}}-1}\right) \tag{54}$$

$$x_1^{(i,j)} = x^{(j)} W_1^{(i)} \tag{55}$$

$$x_2^{(i,j)} = x^{(j)} W_2^{(i)} \tag{56}$$

$$z^{(i,j)} = f\left(x_1^{(i,j)}\right) * x_2^{(i,j)} \tag{57}$$

$$y^{(i,j)} = z^{(i,j)} W_3^{(i)} \tag{58}$$

$$y^{(j)} = \text{all-reduce}\left(\{y^{(i,j)}\}_{i=0}^{N_{\mathrm{TP}}-1}\right), \tag{59}$$

最终得到大小为 $\left(\frac{B}{N_{\mathrm{FSDP}}}, D\right)$ 的按 batch 分片输出 $y^{(j)}$。为简洁起见，本节省略反向传播，只关注前向传播。但此时，你应该已经掌握了自行写出反向传播所需的全部信息。

**问题 (fsdp_tp_calcs)：二维并行计算（6 分）**

在与目前所有计算相同的设定下，我们来计算二维并行何时会遇到瓶颈。

(a) 在有 $N_{\mathrm{FSDP}}$ 路 FSDP + $N_{\mathrm{TP}}$ 路 TP 时，计算前向传播需要多少 FLOP？

**提交要求：** 用 $B$、$D$、$D_{\mathrm{FF}}$、$N_{\mathrm{FSDP}}$ 和 $N_{\mathrm{TP}}$ 表示的答案，外加一句话的论证。

(b) 在有 $N_{\mathrm{FSDP}}$ 路 FSDP + $N_{\mathrm{TP}}$ 路 TP 时，前向传播需要多少通信时间？假设沿每个轴的通信可以重叠（换句话说，沿 FSDP 轴的集合通信可以与沿 TP 轴的集合通信重叠）。

**提交要求：** 用 $B$、$D$、$D_{\mathrm{FF}}$、$N_{\mathrm{FSDP}}$、$N_{\mathrm{TP}}$ 和 $W$ 的一个子集表示的答案，外加一句话的论证。提示：答案应表示为两个量（FSDP 与 TP 集合通信成本）之间的 max，因为两者可以重叠。

(c) 在 $N_{\mathrm{TP}}$ 和 $N_{\mathrm{FSDP}}$ 的最优设置下，$N = N_{\mathrm{TP}} N_{\mathrm{FSDP}}$ 最大可以到多少，前向传播才会被通信卡脖子？

**提交要求：** 一个不等式，一侧是 $N$，另一侧是用 $B$、$D$、$D_{\mathrm{FF}}$、$C$ 和 $W$ 的一个子集表示的表达式，外加几句话和若干方程作为论证。

(d) 现在假设 FSDP 轴和 TP 轴的集合通信不能重叠，因为它们共享同一网络资源。在 $N_{\mathrm{TP}}$ 和 $N_{\mathrm{FSDP}}$ 的最优设置下，$N = N_{\mathrm{TP}} N_{\mathrm{FSDP}}$ 最大可以到多少，前向传播才会被通信卡脖子？不必纠结于把 $N_{\mathrm{TP}}$ 和 $N_{\mathrm{FSDP}}$ 截断为整数。

**提交要求：** 一个不等式，一侧是 $N$，另一侧是用 $B$、$D$、$D_{\mathrm{FF}}$、$C$ 和 $W$ 的一个子集表示的表达式，外加几句话和若干方程作为论证。

## 9 排行榜

作业 2 的排行榜将测试一个 8B 模型完整训练步骤的速度。我们挑战你：对代码做基准测试，并用你能想到的任何技巧优化内存与运行时间。关键限制是你不能改变模型的输入/输出行为。我们将对照 `cs336_basics` 目录中的模型测试你的实现。你的输入将以 BF16 和因果掩码进行测试，并且必须通过与常规实现相同的测试。实现也必须是你自己的，不能使用或复制已有的实现。计时应在两块 B200 GPU 上测量，使用批量大小 2、序列长度 32,768 的样本。把模型放进内存本身就是有意设计得有难度的。完整配置见下面的代码。我们将验证前 5-10 名提交的正确性与性能。我们将运行以下测试来为你的实现计时：

```python
class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


cfg = Config()


def test_timing_forward_backward():
    labels, targets = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size, cfg.ctx_len))
    model = BasicsTransformerLM(Config())
    optimizer = AdamW(model.parameters())

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(labels)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        optimizer.step()

    timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    print(timing_results)
```

测试时，你可以把重复次数和预热时间（单位为 ms）调短一些。

一些改进思路，以及确保模型放得下的方法：

- 调优内核的分块大小（用 Triton autotune！）
- 调优其他 Triton/`torch.compile` 配置参数
- 实现融合的 AdamW
- 基础实现会物化完整 logits（[batch, seq_len, vocab_size]）。写一个把 LM head 和交叉熵损失融合的内核。你还可以让它以融合的方式立即计算反向传播
- 改进 FlashAttention
  - 用 Triton 实现反向传播，而不只是 `torch.compile`（见 4.2.3 节）
  - 反向传播对输入做两趟处理：一趟算 $dQ$，另一趟算 $dK$ 和 $dV$，以避免块之间的原子操作或同步
  - 做因果掩码时提前终止程序实例，跳过保证全零的分块
  - 把不带掩码的分块与对角线上的分块分开：前者完全不做索引比较，后者只做一次比较
  - 在 Hopper 之后的架构上使用 TMA（Tensor Memory Accelerator）功能，遵循与我们教程类似的模式
- 仅在确有需要时，才使用激活检查点以运行速度换取内存节省

**问题 (leaderboard)：排行榜——最快训练步（10 分）**

基准测试将在两块 B200 GPU 上以批量大小 2 运行。你的提交将按一个完整训练步骤（前向传播、损失、反向传播和 AdamW 更新）的墙上时钟时间（wall-clock time）评估。

从空的 PyTorch/Triton 缓存开始，你的基准测试运行必须在 10 分钟内完成，因此要当心过于激进的 `torch.compile` 和 Triton autotuning。

**提交要求：** 你的完整前向-反向训练步（含 AdamW）的最佳墙上时钟时间。

我们期望排行榜提交能击败 10 秒的朴素基线。

在此提交你的结果到排行榜：

github.com/stanford-cs336/assignment2-systems-leaderboard

## 参考文献

[1] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” [Online]. Available: https://arxiv.org/abs/2307.08691

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https://openreview.net/forum?id=H4DqfPSibmx

[3] M. Milakov and N. Gimelshein, “Online normalizer calculation for softmax.” [Online]. Available: https://arxiv.org/abs/1805.02867

[4] H. He, “Making Deep Learning Go Brrrr From First Principles,” 2022, [Online]. Available: https://horace.io/brrr_intro.html

[5] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” 2020.

[6] J. Austin et al., “How to Scale Your Model,” 2025.

[7] H. Z. P. N. M. M. L. W. T. W. Nouamane Tazi Ferdinand Mom, “The Ultra-Scale Playbook: Training LLMs on GPU Clusters.” 2025.
