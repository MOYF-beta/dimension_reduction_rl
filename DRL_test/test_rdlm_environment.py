#!/usr/bin/env python3
"""
RDLM环境配置验证测试

此脚本验证RDLM（强化学习降维）环境是否被正确配置，
包括TRL、transformers、torch等必要依赖项的安装和配置。

基于Hugging Face TRL快速入门指南：
https://hugging-face.cn/docs/trl/quickstart
"""

import sys
import warnings
import logging
from typing import List, Tuple, Optional
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rdlm_test.log')
    ]
)
logger = logging.getLogger(__name__)

class RDLMEnvironmentTester:
    """RDLM环境测试器"""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        
    def log_result(self, test_name: str, success: bool, message: str = "", error: Optional[Exception] = None):
        """记录测试结果"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'error': str(error) if error else None
        }
        self.test_results.append(result)
        
        if success:
            logger.info(f"✅ {test_name}: {message}")
        else:
            logger.error(f"❌ {test_name}: {message}")
            if error:
                logger.error(f"   错误详情: {error}")
                self.errors.append(f"{test_name}: {error}")
    
    def test_basic_imports(self) -> bool:
        """测试基础包导入"""
        test_name = "基础包导入测试"
        try:
            import torch
            import numpy as np
            import transformers
            
            self.log_result(test_name, True, 
                          f"PyTorch: {torch.__version__}, "
                          f"Transformers: {transformers.__version__}, "
                          f"NumPy: {np.__version__}")
            return True
        except ImportError as e:
            self.log_result(test_name, False, "基础包导入失败", e)
            return False
    
    def test_trl_imports(self) -> bool:
        """测试TRL包导入"""
        test_name = "TRL包导入测试"
        try:
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
            import trl
            
            self.log_result(test_name, True, f"TRL版本: {trl.__version__}")
            return True
        except ImportError as e:
            self.log_result(test_name, False, "TRL包导入失败", e)
            return False
    
    def test_cuda_availability(self) -> bool:
        """测试CUDA可用性"""
        test_name = "CUDA可用性测试"
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                self.log_result(test_name, True, 
                              f"CUDA可用，设备数量: {device_count}, "
                              f"当前设备: {current_device}, "
                              f"设备名称: {device_name}")
            else:
                self.log_result(test_name, True, "CUDA不可用，将使用CPU")
            
            return True
        except Exception as e:
            self.log_result(test_name, False, "CUDA检查失败", e)
            return False
    
    def test_tokenizer_loading(self) -> bool:
        """测试分词器加载"""
        test_name = "分词器加载测试"
        try:
            from transformers import GPT2Tokenizer
            
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            # 测试编码
            test_text = "This is a test sentence."
            encoded = tokenizer.encode(test_text, return_tensors="pt")
            
            self.log_result(test_name, True, 
                          f"GPT2分词器加载成功，词汇表大小: {tokenizer.vocab_size}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "分词器加载失败", e)
            return False
    
    def test_model_loading(self) -> bool:
        """测试模型加载"""
        test_name = "模型加载测试"
        try:
            from trl import AutoModelForCausalLMWithValueHead
            import torch
            
            # 加载小型模型进行测试
            model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
            
            # 检查模型结构
            param_count = sum(p.numel() for p in model.parameters())
            
            self.log_result(test_name, True, 
                          f"GPT2模型加载成功，参数数量: {param_count:,}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "模型加载失败", e)
            return False
    
    def test_ppo_trainer_initialization(self) -> bool:
        """测试PPO训练器组件初始化"""
        test_name = "PPO训练器组件测试"
        try:
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig
            from transformers import GPT2Tokenizer, AutoModelForSequenceClassification
            import torch
            import datasets
            
            # 加载模型和分词器
            model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            # 创建简单的奖励模型和价值模型
            reward_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
            value_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
            
            # 创建简单的数据集
            dummy_data = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
            train_dataset = datasets.Dataset.from_dict(dummy_data)
            
            # 初始化PPO配置
            ppo_config = {"mini_batch_size": 1, "batch_size": 1}
            config = PPOConfig(**ppo_config)
            
            # 验证所有组件都能正确创建
            components = {
                "主模型": model,
                "参考模型": ref_model,
                "奖励模型": reward_model,
                "价值模型": value_model,
                "分词器": tokenizer,
                "配置": config,
                "数据集": train_dataset
            }
            
            self.log_result(test_name, True, 
                          f"PPO训练器所需组件创建成功: {list(components.keys())}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "PPO训练器组件创建失败", e)
            return False
    
    def test_trl_api_compatibility(self) -> bool:
        """测试TRL API兼容性"""
        test_name = "TRL API兼容性测试"
        try:
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig
            from transformers import GPT2Tokenizer
            import torch
            
            # 检查关键类和方法的存在
            model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            # 检查模型结构
            has_v_head = hasattr(model, 'v_head')
            has_pretrained_model = hasattr(model, 'pretrained_model')
            has_base_model = hasattr(model, 'base_model') or has_pretrained_model
            
            # 检查PPOConfig
            config = PPOConfig(mini_batch_size=1, batch_size=1)
            has_config_attrs = all(hasattr(config, attr) for attr in ['mini_batch_size', 'batch_size'])
            
            # 检查生成功能
            test_input = tokenizer("Hello", return_tensors="pt")
            can_generate = hasattr(model, 'generate')
            
            api_features = {
                "价值头": has_v_head,
                "预训练模型": has_pretrained_model,
                "基础模型": has_base_model,
                "PPO配置": has_config_attrs,
                "生成功能": can_generate
            }
            
            all_features_ok = all(api_features.values())
            
            self.log_result(test_name, all_features_ok, 
                          f"TRL API特性检查: {api_features}")
            return all_features_ok
        except Exception as e:
            self.log_result(test_name, False, "TRL API兼容性检查失败", e)
            return False
    
    def test_complete_pipeline(self) -> bool:
        """测试TRL生成管道"""
        test_name = "TRL生成管道测试"
        try:
            from trl import AutoModelForCausalLMWithValueHead
            from transformers import GPT2Tokenizer
            import torch
            
            # 设置模型和分词器
            model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            # 编码查询
            query_txt = "This morning I went to the "
            query_tensor = tokenizer.encode(query_txt, return_tensors="pt")
            
            # 使用模型生成响应
            with torch.no_grad():
                generation_kwargs = {
                    "min_length": -1,
                    "top_k": 0.0,
                    "top_p": 1.0,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "max_new_tokens": 20,
                }
                
                # 直接使用模型生成
                outputs = model.generate(
                    query_tensor,
                    **generation_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # 解码生成的文本
                generated_ids = outputs.sequences[0][len(query_tensor[0]):]
                response_txt = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 测试价值函数
                if hasattr(model, 'v_head') and hasattr(model, 'pretrained_model'):
                    # 获取最后一层隐藏状态
                    with torch.no_grad():
                        model_output = model.pretrained_model(query_tensor, output_hidden_states=True)
                        hidden_states = model_output.hidden_states[-1]  # 最后一层
                        value = model.v_head(hidden_states)
                        value_info = f"价值张量形状: {value.shape}"
                else:
                    value_info = "价值头不可用"
                
            self.log_result(test_name, True, 
                          f"TRL生成管道测试成功。生成文本: '{response_txt.strip()}', "
                          f"{value_info}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "TRL生成管道测试失败", e)
            return False
    
    def test_environment_variables(self) -> bool:
        """测试环境变量"""
        test_name = "环境变量测试"
        try:
            import os
            
            important_vars = ['CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TRANSFORMERS_CACHE']
            found_vars = {}
            
            for var in important_vars:
                value = os.environ.get(var)
                if value:
                    found_vars[var] = value
            
            self.log_result(test_name, True, 
                          f"找到环境变量: {found_vars}" if found_vars else "未找到特殊环境变量")
            return True
        except Exception as e:
            self.log_result(test_name, False, "环境变量检查失败", e)
            return False
    
    def test_memory_usage(self) -> bool:
        """测试内存使用情况"""
        test_name = "内存使用测试"
        try:
            import psutil
            import torch
            
            # 系统内存
            memory = psutil.virtual_memory()
            
            # GPU内存（如果可用）
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_info = f", GPU内存: {gpu_memory / 1024**3:.1f}GB"
            
            self.log_result(test_name, True, 
                          f"系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB){gpu_info}")
            return True
        except ImportError:
            self.log_result(test_name, True, "psutil未安装，跳过详细内存检查")
            return True
        except Exception as e:
            self.log_result(test_name, False, "内存检查失败", e)
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("🚀 开始RDLM环境配置验证测试...")
        logger.info("=" * 60)
        
        # 测试列表
        tests = [
            self.test_basic_imports,
            self.test_trl_imports,
            self.test_cuda_availability,
            self.test_environment_variables,
            self.test_memory_usage,
            self.test_tokenizer_loading,
            self.test_model_loading,
            self.test_trl_api_compatibility,
            self.test_ppo_trainer_initialization,
            self.test_complete_pipeline,
        ]
        
        # 运行测试
        total_tests = len(tests)
        passed_tests = 0
        
        for test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"测试执行异常: {e}")
                traceback.print_exc()
        
        # 输出总结
        logger.info("=" * 60)
        logger.info(f"📊 测试总结: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            logger.info("🎉 所有测试通过！RDLM环境配置正确。")
            return True
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} 个测试失败。")
            logger.info("💡 失败的测试:")
            for error in self.errors:
                logger.info(f"   - {error}")
            return False
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("RDLM环境配置验证报告")
        report.append("=" * 40)
        report.append("")
        
        for result in self.test_results:
            status = "✅ 通过" if result['success'] else "❌ 失败"
            report.append(f"{status} {result['test']}")
            if result['message']:
                report.append(f"   {result['message']}")
            if result['error']:
                report.append(f"   错误: {result['error']}")
            report.append("")
        
        # 总结
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        report.append(f"总计: {passed}/{total} 测试通过")
        
        return "\n".join(report)


def main():
    """主函数"""
    print("RDLM环境配置验证测试")
    print("基于Hugging Face TRL快速入门指南")
    print("=" * 60)
    
    # 创建测试器实例
    tester = RDLMEnvironmentTester()
    
    # 运行所有测试
    success = tester.run_all_tests()
    
    # 生成并保存报告
    report = tester.generate_report()
    with open('rdlm_environment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("📝 详细报告已保存到 rdlm_environment_report.txt")
    
    # 返回适当的退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
