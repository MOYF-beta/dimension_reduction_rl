#!/bin/bash

# RDLM环境安装和测试脚本
# 此脚本将安装必要的依赖项并运行配置验证测试

set -e  # 遇到错误立即退出

echo "🚀 RDLM环境配置和测试脚本"
echo "================================"

# 检查Python版本
echo "📋 检查Python版本..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python版本: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ 错误: 需要Python 3.8或更高版本"
    exit 1
fi

echo "✅ Python版本符合要求"

# 创建虚拟环境（可选）
if [ "$1" = "--create-venv" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv rdlm_env
    source rdlm_env/bin/activate
    echo "✅ 虚拟环境已创建并激活"
fi

# 升级pip
echo "📦 升级pip..."
python3 -m pip install --upgrade pip

# 安装依赖项
echo "📦 安装依赖项..."
if [ -f "requirements.txt" ]; then
    echo "从requirements.txt安装依赖项..."
    pip install -r requirements.txt
else
    echo "安装核心依赖项..."
    pip install torch transformers trl datasets tokenizers numpy pandas matplotlib psutil
fi

echo "✅ 依赖项安装完成"

# 运行配置验证测试
echo "🧪 运行RDLM环境配置验证测试..."
echo "================================"

python3 test_rdlm_environment.py

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 RDLM环境配置验证成功！"
    echo "📝 详细报告已保存到 rdlm_environment_report.txt"
    echo "📋 日志文件: rdlm_test.log"
else
    echo ""
    echo "❌ RDLM环境配置验证失败"
    echo "请查看错误信息并解决问题"
    echo "📋 详细日志: rdlm_test.log"
    exit 1
fi

echo ""
echo "🔧 环境配置完成！现在你可以开始使用RDLM进行强化学习降维实验。"
