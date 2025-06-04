#!/usr/bin/env python3
"""
简化的RDLM环境测试脚本
用于快速验证基本配置，不需要下载大型模型
"""

import sys
import logging

# 配置简单日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_python_environment():
    """测试基本Python环境"""
    print("🐍 检查Python环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 测试基本库导入
    try:
        import json
        import os
        import subprocess
        print("✅ 基本库导入成功")
        return True
    except ImportError as e:
        print(f"❌ 基本库导入失败: {e}")
        return False

def test_pip_packages():
    """测试pip包管理"""
    print("\n📦 检查包管理器...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pip版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ pip不可用")
            return False
    except Exception as e:
        print(f"❌ pip检查失败: {e}")
        return False

def test_optional_packages():
    """测试可选包的可用性"""
    print("\n🔍 检查可选依赖包...")
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'trl': 'TRL',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    results = {}
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}: 已安装")
            results[package] = True
        except ImportError:
            print(f"⚠️  {name}: 未安装")
            results[package] = False
    
    return results

def suggest_installation():
    """提供安装建议"""
    print("\n💡 安装建议:")
    print("如果需要完整的RDLM环境，请运行：")
    print("  pip install torch transformers trl numpy pandas")
    print("或者运行完整的安装脚本：")
    print("  ./setup_and_test.sh")

def main():
    print("🚀 RDLM环境快速检查")
    print("=" * 40)
    
    # 基本测试
    basic_ok = test_basic_python_environment()
    pip_ok = test_pip_packages()
    package_results = test_optional_packages()
    
    # 总结
    print("\n📊 检查总结:")
    print("=" * 40)
    
    if basic_ok and pip_ok:
        print("✅ 基本环境正常")
        
        installed_count = sum(package_results.values())
        total_count = len(package_results)
        
        if installed_count == total_count:
            print("🎉 所有RDLM依赖包都已安装！")
            print("你可以运行完整测试：python test_rdlm_environment.py")
        elif installed_count > 0:
            print(f"⚠️  部分依赖包已安装 ({installed_count}/{total_count})")
            suggest_installation()
        else:
            print("❌ RDLM依赖包未安装")
            suggest_installation()
    else:
        print("❌ 基本环境有问题，请检查Python和pip安装")
    
    print("\n🔗 更多信息请查看 README.md")

if __name__ == "__main__":
    main()
