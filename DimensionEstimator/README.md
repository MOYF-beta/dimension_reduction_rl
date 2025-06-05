# 自然语言相关维数估计器

本实现基于论文 [《统计流形中自然语言的相关维数》](https://arxiv.org/abs/2405.06321)，作者：杜鑫 和 田中-石井久美子 (2024)。

## 概述

本工具使用在统计流形中重新表述的Grassberger-Procaccia算法和Fisher-Rao距离来估计自然语言文本的相关维数。该方法利用大型语言模型获取文本序列的概率分布，并测量其分形维数。

### 主要特性

- **统计流形分析**：使用Fisher-Rao距离代替欧几里得距离
- **HuggingFace集成**：支持任何HuggingFace因果语言模型
- **熵过滤**：过滤低熵区域，专注于全局分形模式
- **维度约简**：可选的词汇表缩减以提高计算效率
- **批处理**：高效处理多个文本
- **内存优化**：自动块处理和GPU内存管理

## 理论基础

### 统计流形中的动力系统

根据Du和Tanaka-Ishii的论文，自然语言可以建模为统计流形中的动力系统：

1. **状态空间**：语言状态 $x_t$ 被定义为词汇表上的概率分布
2. **映射函数**：$\phi: x_t \mapsto p_t$，其中 $p_t$ 是下一个词的概率分布
3. **统计流形**：概率分布空间 $\text{Mult}(V)$，维度为 $|V|-1$

### Grassberger-Procaccia算法

相关维数通过相关积分的幂律标度来估计：

$$C(\varepsilon) \sim \varepsilon^{\nu} \quad \text{当} \quad \varepsilon \to 0$$

其中：
$$C(\varepsilon) = \lim_{N \to \infty} \frac{1}{N^2} \sum_{1 \leq t,s \leq N} \mathbf{1}\{d(p_t, p_s) < \varepsilon\}$$

### Fisher-Rao距离

在统计流形中，我们使用Fisher-Rao距离：

$$d_{FR}(p_t, p_s) = 2 \arccos\left(\sum_{w \in V} \sqrt{p_t(w) p_s(w)}\right)$$

这等于两倍的Bhattacharyya角度，在多项分布的统计流形上具有恒定曲率的黎曼度量。

## 安装

```bash
pip install -r requirements.txt
```

### 依赖项

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
matplotlib>=3.5.0
datasets>=2.10.0
```

## 使用方法

### 基本用法

```python
from dimension_estimator import DimensionEstimator

# 使用HuggingFace模型初始化
estimator = DimensionEstimator(
    model_name_or_path="gpt2",
    max_context_length=512,
    eta_threshold=0.5,
    dimension_reduction=1000
)

# 估计文本列表的维数
texts = ["你的文本在这里...", "另一个文本..."]
results = estimator.estimate_dimension(texts)

print(f"相关维数: {results['correlation_dimension']:.3f}")
print(f"R平方: {results['r_squared']:.3f}")
```

### 高级用法

```python
# 使用更大的模型获得更好的结果
estimator = DimensionEstimator(
    model_name_or_path="gpt2-xl",
    device="cuda",  # 如果可用，使用GPU
    max_context_length=1024,
    eta_threshold=0.4,  # 更低的阈值保留更多序列
    dimension_reduction=2000,
    batch_size=16
)

# 从文件加载文本
with open('book.txt', 'r', encoding='utf-8') as f:
    long_text = f.read()

results = estimator.estimate_dimension([long_text])

# 绘制相关积分曲线
estimator.plot_correlation_integral(results)
```

### MATH-500数据集分析

```bash
# 快速测试
python quick_test_math500.py

# 完整分析
python estimate_math500_dimension.py --sample-size 100 --eta-threshold 0.4

# 参数扫描
python comprehensive_math500_analysis.py --run-sweep
```

## 算法详解

### 1. 概率提取

对于文本中的每个位置 $t$，使用语言模型提取词汇表上的概率分布 $p_t$：

```python
# 对于上下文 a_{<t} = [a_1, ..., a_{t-1}]
p_t(w) = P(a_t = w | a_{<t})  # 对所有 w ∈ V
```

### 2. 熵过滤

过滤掉最大概率超过阈值 $\eta$ 的分布，专注于高熵的全局分形模式：

```python
# 保留满足 max(p_t) < η 的分布
high_entropy_mask = torch.max(prob_distributions, dim=1)[0] < eta_threshold
```

**数学原理**：低熵分布（高 $\max(p_t)$）对应于语言中的局部确定性区域，而高熵分布揭示全局统计模式。

### 3. 距离计算

计算概率分布之间的Fisher-Rao距离：

```python
def fisher_rao_distance(p1, p2):
    # 计算Bhattacharyya系数
    bc = torch.sum(torch.sqrt(p1 * p2), dim=-1)
    # Fisher-Rao距离
    return 2 * torch.arccos(torch.clamp(bc, 0, 1))
```

### 4. 相关积分

计算 $C(\varepsilon) = \frac{\text{距离} < \varepsilon \text{的对数}}{\text{总对数}}$：

```python
# 对于多个ε值的向量化计算
for eps in epsilons:
    count = torch.sum(distances < eps)
    correlation_integral = count / total_pairs
```

### 5. 维数估计

在对数-对数图上进行线性回归找到斜率 $\nu$：

```python
# log C(ε) = ν log ε + constant
slope, intercept, r_value = linregress(log_epsilons, log_correlations)
correlation_dimension = slope
```

## 参数详解

### 核心参数

- **`model_name_or_path`**: HuggingFace模型标识符
  - 推荐: `"gpt2"` (快速), `"gpt2-xl"` (高质量)
  - 影响: 更大的模型提供更准确的概率估计

- **`max_context_length`**: 模型的最大上下文长度
  - 默认: 512
  - 影响: 更长的上下文捕获更多长期依赖性
  - 权衡: 更长的上下文需要更多内存

- **`eta_threshold`**: 熵过滤的最大概率阈值
  - 默认: 0.5
  - 范围: 0.3-0.7
  - 影响: 
    - 更低 → 保留更多序列，但包含更多噪声
    - 更高 → 更少但更纯净的高熵序列

### 性能参数

- **`dimension_reduction`**: 词汇表缩减大小
  - 作用: 将 $|V| \times N^2$ 复杂度降低到 $M \times N^2$，其中 $M \ll |V|$
  - 方法: 使用模运算 $\Phi(w) = \text{index}(w) \bmod M$
  - 理论基础: Marstrand投影定理保证线性映射几乎肯定保持维数

- **`batch_size`**: 批处理大小
  - 影响内存使用和计算速度
  - GPU: 8-16，CPU: 4-8

- **`device`**: 计算设备
  - `"auto"`: 自动选择（优先GPU）
  - `"cuda"`: 强制使用GPU
  - `"cpu"`: 使用CPU

## 实验结果与解释

### 期望结果

根据论文，不同类型的序列具有不同的相关维数：

| 序列类型 | 相关维数 | 含义 |
|---------|----------|------|
| 自然语言 | ~6.5 | 适中的复杂性和结构性 |
| 乱序文本 | ~13.0 | 失去语言结构，更随机 |
| 随机权重模型 | ~80 | 高度随机，缺乏结构 |
| Barabási-Albert网络 | ~2.0 | 高度结构化 |
| 白噪声 | >100 | 完全随机 |

### 语言间的普遍性

- **英语**: 6.39±0.40
- **中文**: 6.81±0.58  
- **日语**: 7.30±0.41
- **德语**: 5.84±0.70

### MATH-500数据集结果

在数学竞赛问题上的实验结果：
- **维数**: ~7.9 (使用η=0.4)
- **解释**: 数学语言比一般自然语言更结构化，但比网络更复杂

## 计算复杂度与优化

### 时间复杂度

- **原始**: $O(|V| \times N^2)$
- **维度约简后**: $O(M \times N^2)$，其中 $M \ll |V|$
- **内存优化**: 使用分块处理处理大型数据集

### 内存优化策略

1. **自动分块**: 根据可用内存自动调整块大小
2. **批处理**: 向量化距离计算
3. **GPU内存管理**: 自动检测和适应GPU内存限制

```python
# 自动内存管理示例
if memory_required > available_memory * 0.1:
    chunk_size = max(50, int(np.sqrt(max_pairs_per_chunk)))
    return self._compute_correlation_integral_chunked(...)
```

## 数值稳定性

### 处理零概率

```python
# 为数值稳定性添加小的ε
eps = 1e-12
p1 = torch.clamp(p1, min=eps)
p2 = torch.clamp(p2, min=eps)
```

### 反余弦函数的界限

```python
# 避免arccos的数值问题
bc = torch.clamp(bc, min=0.0, max=1.0)
distance = 2 * torch.arccos(bc)
```

## 最佳实践

### 1. 文本预处理
- 确保文本长度 >1000 tokens 以获得可靠估计
- 合并相关文档以增加序列长度

### 2. 参数调优
- 开始时使用较小的样本测试参数
- 调整 `eta_threshold` 以获得足够的高熵序列
- 使用维度约简来平衡速度和准确性

### 3. 结果验证
- 检查 R² > 0.8 以确保良好的线性拟合
- 验证高熵序列分数 > 0.1
- 比较不同参数设置的结果

## 故障排除

### 常见问题

1. **内存不足**
```bash
# 减少批大小
python estimate_math500_dimension.py --batch-size 4

# 使用维度约简
python estimate_math500_dimension.py --dimension-reduction 500
```

2. **高熵序列太少**
```bash
# 降低熵阈值
python estimate_math500_dimension.py --eta-threshold 0.3
```

3. **处理速度慢**
```bash
# 使用较小的样本
python estimate_math500_dimension.py --sample-size 100

# 减少上下文长度
python estimate_math500_dimension.py --max-context 256
```