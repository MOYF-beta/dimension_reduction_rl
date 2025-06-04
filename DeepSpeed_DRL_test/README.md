# DeepSpeed Zero3 Offload 测试

这个测试套件验证DeepSpeed Zero3配置，特别是权重和梯度到内存的offload功能。

## 功能特性

### 🔧 测试覆盖范围

1. **基础环境检查**
   - PyTorch、Transformers、NumPy等基础包
   - DeepSpeed包和版本检查
   - CUDA可用性和GPU内存检查

2. **Zero3配置测试**
   - Zero3配置文件创建和验证
   - 参数offload到CPU内存配置
   - 优化器状态offload配置

3. **模型和训练测试**
   - 大型模型初始化
   - DeepSpeed引擎初始化
   - 参数offload功能验证
   - 梯度offload功能验证

4. **高级功能测试**
   - 模型检查点保存和加载
   - 内存效率测试
   - 分布式设置兼容性检查

### 🚀 快速开始

#### 1. 环境设置

```bash
# 运行自动设置脚本
./setup_and_test.sh
```

或者手动设置：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
python test_deepspeed_zero3.py
```

#### 2. 查看结果

测试完成后会生成以下文件：
- `deepspeed_zero3_report.txt` - 详细测试报告
- `deepspeed_zero3_test.log` - 测试日志
- `ds_config_zero3.json` - 生成的Zero3配置文件

### 📋 Zero3 Offload配置

测试使用的Zero3配置包括：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu", 
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  }
}
```

### 🔍 关键测试项目

#### 参数Offload测试
- 验证模型参数可以offload到CPU内存
- 测试参数在需要时的自动加载
- 监控CPU和GPU内存使用情况

#### 梯度Offload测试  
- 验证梯度可以offload到CPU内存
- 测试梯度累积功能
- 验证优化器状态的offload

#### 内存效率测试
- 比较不同模型大小的内存使用
- 测试Zero3的内存节省效果
- 监控训练过程中的内存波动

### 🛠️ 依赖要求

- Python 3.7+
- PyTorch 1.13.0+
- DeepSpeed 0.8.0+
- Transformers 4.20.0+
- CUDA (推荐，但不是必需)

### 📊 性能预期

使用Zero3 Offload，您应该看到：

1. **GPU内存减少**: 大部分参数和梯度存储在CPU内存中
2. **训练大型模型**: 能够训练超出GPU内存限制的模型
3. **轻微性能损失**: 由于CPU-GPU数据传输的开销

### 🐛 故障排除

#### 常见问题

1. **DeepSpeed安装失败**
   ```bash
   # 尝试从源码安装
   pip install deepspeed --global-option="build_ext" --global-option="-j8"
   ```

2. **CUDA兼容性问题**
   ```bash
   # 检查CUDA版本兼容性
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **内存不足**
   - 减小模型大小或批次大小
   - 启用更多offload选项
   - 增加交换空间

#### 调试模式

运行测试时添加详细日志：

```bash
export CUDA_LAUNCH_BLOCKING=1
export DEEPSPEED_LOG_LEVEL=DEBUG
python test_deepspeed_zero3.py
```

### 📚 参考资源

- [DeepSpeed Zero3官方文档](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed配置参考](https://www.deepspeed.ai/docs/config-json/)
- [Zero Offload论文](https://arxiv.org/abs/2101.06840)

### 🤝 贡献

欢迎提交问题报告和改进建议！
