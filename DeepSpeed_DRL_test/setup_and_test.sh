#!/bin/bash

# DeepSpeed Zero3 Offload环境设置和测试脚本

set -e  # 遇到错误时退出

echo "🚀 DeepSpeed Zero3 Offload环境设置和测试"
echo "=========================================="

# 检查Python版本
echo "📋 检查Python环境..."
python3 --version
which python3

# 创建虚拟环境（可选）
# echo "🔧 创建虚拟环境..."
# python3 -m venv deepspeed_env
# source deepspeed_env/bin/activate

# 升级pip
echo "⬆️  升级pip..."
python3 -m pip install --upgrade pip

# 安装依赖
echo "📦 安装依赖包..."
python3 -m pip install -r requirements.txt

# 检查CUDA和GPU
echo "🖥️  检查CUDA和GPU..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 检查DeepSpeed安装
echo "🔍 检查DeepSpeed安装..."
python3 -c "import deepspeed; print(f'DeepSpeed版本: {deepspeed.__version__}')"

# 运行DeepSpeed环境报告
echo "📊 生成DeepSpeed环境报告..."
ds_report

# 运行测试
echo "🧪 运行DeepSpeed Zero3 Offload测试..."
python3 test_deepspeed_zero3.py

echo "✅ 测试完成！查看生成的报告文件了解详细结果。"
