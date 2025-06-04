# RDLM环境配置验证

这个项目提供了一套完整的测试工具来验证RDLM（强化学习降维）环境是否被正确配置。测试基于[Hugging Face TRL快速入门指南](https://hugging-face.cn/docs/trl/quickstart)。

## 文件结构

- `test_rdlm_environment.py` - 主要的环境验证测试脚本
- `requirements.txt` - 项目依赖项
- `setup_and_test.sh` - 自动安装和测试脚本
- `README.md` - 本文档

## 功能特性

### 测试覆盖范围

1. **基础包导入测试** - 验证PyTorch、Transformers、NumPy等核心包
2. **TRL包导入测试** - 验证TRL库及其核心组件
3. **CUDA可用性测试** - 检查GPU支持和配置
4. **分词器加载测试** - 验证GPT2分词器功能
5. **模型加载测试** - 验证模型加载和参数统计
6. **PPO训练器初始化测试** - 验证强化学习训练器设置
7. **完整PPO管道测试** - 端到端的训练流程验证
8. **环境变量测试** - 检查重要的环境配置
9. **内存使用测试** - 监控系统资源使用情况

### 输出特性

- 彩色控制台输出，清晰显示测试结果
- 详细的日志文件记录
- 自动生成的测试报告
- 错误诊断和建议

## 快速开始

### 方法1：使用自动脚本（推荐）

```bash
# 下载或克隆项目后，运行：
chmod +x setup_and_test.sh
./setup_and_test.sh

# 如果需要创建虚拟环境：
./setup_and_test.sh --create-venv
```

### 方法2：手动安装和测试

```bash
# 1. 安装依赖项
pip install -r requirements.txt

# 2. 运行测试
python test_rdlm_environment.py
```

### 方法3：最小化测试

如果只需要核心依赖：

```bash
# 安装最小依赖
pip install torch transformers trl datasets numpy

# 运行测试
python test_rdlm_environment.py
```

## 系统要求

### 最低要求
- Python 3.8+
- 8GB RAM（推荐16GB+）
- 可用磁盘空间：10GB+

### 推荐配置
- Python 3.9+
- NVIDIA GPU with CUDA 11.0+
- 32GB+ RAM
- SSD存储

### 支持的平台
- Linux（推荐）
- macOS
- Windows 10/11

## 测试结果解释

### ✅ 成功指标
- 所有导入测试通过
- 模型能够成功加载
- PPO训练器正常初始化
- 完整管道执行无错误

### ❌ 常见问题

1. **TRL导入失败**
   ```
   解决方案：pip install trl>=0.7.0
   ```

2. **CUDA不可用**
   ```
   这不是错误，CPU模式也可以运行，只是速度较慢
   ```

3. **内存不足**
   ```
   尝试减少batch_size或使用更小的模型
   ```

4. **模型下载失败**
   ```
   检查网络连接，或设置代理：
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 输出文件

测试完成后会生成以下文件：

- `rdlm_test.log` - 详细的测试日志
- `rdlm_environment_report.txt` - 测试结果报告

## 环境变量配置

可以设置以下环境变量来优化体验：

```bash
# Hugging Face模型缓存目录
export HF_HOME=/path/to/cache

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0

# 使用中国镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com
```

## 故障排除

### 网络问题
如果模型下载缓慢或失败：

```bash
# 使用中国镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple trl transformers
```

### 内存问题
如果出现内存不足：

```bash
# 监控内存使用
watch -n 1 'free -h && nvidia-smi'

# 清理缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 权限问题
如果脚本无法执行：

```bash
chmod +x setup_and_test.sh
sudo chown $USER:$USER test_rdlm_environment.py
```

## 进阶使用

### 自定义测试
你可以修改`test_rdlm_environment.py`来添加自定义测试：

```python
def test_custom_feature(self) -> bool:
    """自定义测试"""
    test_name = "自定义功能测试"
    try:
        # 你的测试逻辑
        result = your_custom_test()
        self.log_result(test_name, True, "测试通过")
        return True
    except Exception as e:
        self.log_result(test_name, False, "测试失败", e)
        return False
```

### 集成到CI/CD
可以将此测试集成到持续集成流程中：

```yaml
# GitHub Actions示例
name: RDLM Environment Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Test RDLM Environment
      run: |
        chmod +x setup_and_test.sh
        ./setup_and_test.sh
```

## 贡献指南

欢迎提交问题报告和改进建议！请确保：

1. 描述清楚遇到的问题
2. 提供完整的错误日志
3. 说明你的系统环境
4. 如果可能，提供复现步骤

## 许可证

MIT License

## 相关资源

- [Hugging Face TRL文档](https://hugging-face.cn/docs/trl)
- [PyTorch文档](https://pytorch.org/docs/)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [强化学习教程](https://stable-baselines3.readthedocs.io/)

---

**注意**：首次运行时需要下载预训练模型（约500MB），请确保网络连接稳定。
