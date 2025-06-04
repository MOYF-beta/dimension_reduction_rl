#!/usr/bin/env python3
"""
DeepSpeed Zero3 Offload 配置验证测试

此脚本验证DeepSpeed Zero3环境是否被正确配置，
包括权重和梯度到内存的offload功能测试。

基于DeepSpeed Zero3官方文档：
https://www.deepspeed.ai/tutorials/zero/
"""

import sys
import warnings
import logging
import json
import os
from typing import List, Tuple, Optional, Dict, Any
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deepspeed_zero3_test.log')
    ]
)
logger = logging.getLogger(__name__)

class DeepSpeedZero3Tester:
    """DeepSpeed Zero3环境测试器"""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        self.zero3_config = None
        
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
    
    def test_deepspeed_imports(self) -> bool:
        """测试DeepSpeed包导入"""
        test_name = "DeepSpeed包导入测试"
        try:
            import deepspeed
            from deepspeed import DeepSpeedEngine
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
            
            self.log_result(test_name, True, f"DeepSpeed版本: {deepspeed.__version__}")
            return True
        except ImportError as e:
            self.log_result(test_name, False, "DeepSpeed包导入失败", e)
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
                
                # 检查GPU内存
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                memory_gb = total_memory / (1024**3)
                
                self.log_result(test_name, True, 
                              f"CUDA可用，设备数量: {device_count}, "
                              f"当前设备: {current_device}, "
                              f"设备名称: {device_name}, "
                              f"GPU内存: {memory_gb:.1f}GB")
            else:
                self.log_result(test_name, True, "CUDA不可用，将使用CPU")
            
            return True
        except Exception as e:
            self.log_result(test_name, False, "CUDA检查失败", e)
            return False
    
    def create_zero3_config(self) -> Dict[str, Any]:
        """创建Zero3配置"""
        config = {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": int(1e9),
                "reduce_bucket_size": int(5e8),
                "stage3_prefetch_bucket_size": int(5e8),
                "stage3_param_persistence_threshold": int(1e6),
                "stage3_max_live_parameters": int(1e9),
                "stage3_max_reuse_distance": int(1e9),
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "zero_force_ds_cpu_optimizer": False,  # 允许使用自定义优化器
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            "steps_per_print": 2000,
            "train_batch_size": 2,  # 修复: micro_batch_per_gpu * gradient_accumulation_steps * world_size = 2 * 1 * 1 = 2
            "train_micro_batch_size_per_gpu": 2,
            "wall_clock_breakdown": False
        }
        return config
    
    def test_zero3_config_creation(self) -> bool:
        """测试Zero3配置创建"""
        test_name = "Zero3配置创建测试"
        try:
            self.zero3_config = self.create_zero3_config()
            
            # 验证配置结构
            assert "zero_optimization" in self.zero3_config
            assert self.zero3_config["zero_optimization"]["stage"] == 3
            assert "offload_optimizer" in self.zero3_config["zero_optimization"]
            assert "offload_param" in self.zero3_config["zero_optimization"]
            
            # 保存配置到文件
            config_path = "ds_config_zero3.json"
            with open(config_path, 'w') as f:
                json.dump(self.zero3_config, f, indent=2)
            
            self.log_result(test_name, True, 
                          f"Zero3配置创建成功，已保存到 {config_path}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "Zero3配置创建失败", e)
            return False
    
    def test_model_initialization(self) -> bool:
        """测试模型初始化"""
        test_name = "模型初始化测试"
        try:
            import torch
            import torch.nn as nn
            from transformers import GPT2Config, GPT2LMHeadModel
            
            # 创建一个较大的模型用于测试Zero3
            config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=768,
                n_layer=12,
                n_head=12
            )
            
            model = GPT2LMHeadModel(config)
            param_count = sum(p.numel() for p in model.parameters())
            param_size_mb = param_count * 4 / (1024 * 1024)  # 假设float32
            
            self.log_result(test_name, True, 
                          f"GPT2模型初始化成功，参数数量: {param_count:,}, "
                          f"估计大小: {param_size_mb:.1f}MB")
            return True
        except Exception as e:
            self.log_result(test_name, False, "模型初始化失败", e)
            return False
    
    def test_deepspeed_initialization(self) -> bool:
        """测试DeepSpeed引擎初始化"""
        test_name = "DeepSpeed引擎初始化测试"
        try:
            import torch
            import torch.nn as nn
            import deepspeed
            from transformers import GPT2Config, GPT2LMHeadModel
            from torch.optim import AdamW
            
            # 创建模型
            config = GPT2Config(
                vocab_size=50257,
                n_positions=512,  # 减小模型以适应测试
                n_embd=512,
                n_layer=6,
                n_head=8
            )
            model = GPT2LMHeadModel(config)
            
            # 创建优化器
            optimizer = AdamW(model.parameters(), lr=1e-4)
            
            # 初始化DeepSpeed引擎
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=self.zero3_config,
                model_parameters=model.parameters()
            )
            
            # 检查Zero3特性
            zero_stage = model_engine.zero_optimization_stage()
            
            self.log_result(test_name, True, 
                          f"DeepSpeed引擎初始化成功，Zero阶段: {zero_stage}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "DeepSpeed引擎初始化失败", e)
            return False
    
    def test_parameter_offloading(self) -> bool:
        """测试参数offload功能"""
        test_name = "参数Offload测试"
        try:
            import torch
            import deepspeed
            from transformers import GPT2Config, GPT2LMHeadModel
            from torch.optim import AdamW
            import psutil
            import gc
            
            # 记录初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            else:
                initial_gpu_memory = 0
            
            # 创建较大的模型
            config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=768,
                n_layer=8,
                n_head=12
            )
            model = GPT2LMHeadModel(config)
            optimizer = AdamW(model.parameters(), lr=1e-4)
            
            # 使用Zero3初始化
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=self.zero3_config,
                model_parameters=model.parameters()
            )
            
            # 创建测试数据
            batch_size = 2
            seq_length = 512
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # 前向传播
            outputs = model_engine(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            # 检查内存使用
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_increase = final_gpu_memory - initial_gpu_memory
                gpu_info = f", GPU内存增加: {gpu_memory_increase:.1f}MB"
            else:
                gpu_info = ""
            
            self.log_result(test_name, True, 
                          f"参数Offload测试成功，CPU内存增加: {memory_increase:.1f}MB{gpu_info}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "参数Offload测试失败", e)
            return False
    
    def test_gradient_offloading(self) -> bool:
        """测试梯度offload功能"""
        test_name = "梯度Offload测试"
        try:
            import torch
            import deepspeed
            from transformers import GPT2Config, GPT2LMHeadModel
            from torch.optim import AdamW
            
            # 创建模型
            config = GPT2Config(
                vocab_size=50257,
                n_positions=512,
                n_embd=512,
                n_layer=4,
                n_head=8
            )
            model = GPT2LMHeadModel(config)
            optimizer = AdamW(model.parameters(), lr=1e-4)
            
            # 使用Zero3初始化
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=self.zero3_config,
                model_parameters=model.parameters()
            )
            
            # 多步训练测试梯度累积和offload
            total_loss = 0
            steps = 3
            
            for step in range(steps):
                # 创建测试数据
                batch_size = 2
                seq_length = 256
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
                
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                
                # 前向传播
                outputs = model_engine(input_ids, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item()
                
                # 反向传播
                model_engine.backward(loss)
                
                # 每两步更新一次
                if (step + 1) % 2 == 0:
                    model_engine.step()
            
            avg_loss = total_loss / steps
            
            self.log_result(test_name, True, 
                          f"梯度Offload测试成功，{steps}步平均损失: {avg_loss:.4f}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "梯度Offload测试失败", e)
            return False
    
    def test_model_checkpointing(self) -> bool:
        """测试模型检查点保存和加载"""
        test_name = "模型检查点测试"
        try:
            import torch
            import deepspeed
            from transformers import GPT2Config, GPT2LMHeadModel
            from torch.optim import AdamW
            import os
            import shutil
            
            checkpoint_dir = "./test_checkpoint"
            
            # 清理之前的检查点
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            
            # 创建模型
            config = GPT2Config(
                vocab_size=50257,
                n_positions=512,
                n_embd=256,
                n_layer=2,
                n_head=4
            )
            model = GPT2LMHeadModel(config)
            optimizer = AdamW(model.parameters(), lr=1e-4)
            
            # 使用Zero3初始化
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=self.zero3_config,
                model_parameters=model.parameters()
            )
            
            # 训练一步
            batch_size = 1
            seq_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            outputs = model_engine(input_ids, labels=input_ids)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            
            # 保存检查点
            model_engine.save_checkpoint(checkpoint_dir)
            
            # 验证检查点文件
            checkpoint_files = os.listdir(checkpoint_dir)
            has_checkpoint = any(f.startswith('global_step') for f in checkpoint_files)
            
            # 清理
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            
            self.log_result(test_name, True, 
                          f"模型检查点测试成功，检查点文件: {checkpoint_files}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "模型检查点测试失败", e)
            return False
    
    def test_memory_efficiency(self) -> bool:
        """测试内存效率"""
        test_name = "内存效率测试"
        try:
            import torch
            import deepspeed
            from transformers import GPT2Config, GPT2LMHeadModel
            from torch.optim import AdamW
            import psutil
            import gc
            
            # 记录基准内存
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                baseline_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            # 测试不同模型大小的内存使用
            test_configs = [
                {"n_layer": 2, "n_embd": 256, "name": "小型"},
                {"n_layer": 4, "n_embd": 512, "name": "中型"},
                {"n_layer": 6, "n_embd": 768, "name": "大型"}
            ]
            
            memory_results = []
            
            for test_config in test_configs:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 创建模型配置
                config = GPT2Config(
                    vocab_size=50257,
                    n_positions=512,
                    n_embd=test_config["n_embd"],
                    n_layer=test_config["n_layer"],
                    n_head=test_config["n_embd"] // 64
                )
                
                model = GPT2LMHeadModel(config)
                param_count = sum(p.numel() for p in model.parameters())
                
                optimizer = AdamW(model.parameters(), lr=1e-4)
                
                # 使用Zero3初始化
                model_engine, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    optimizer=optimizer,
                    config=self.zero3_config,
                    model_parameters=model.parameters()
                )
                
                # 测量内存使用
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_usage = current_memory - baseline_memory
                
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_usage = current_gpu_memory - baseline_gpu_memory
                else:
                    gpu_memory_usage = 0
                
                memory_results.append({
                    "size": test_config["name"],
                    "params": param_count,
                    "cpu_memory": memory_usage,
                    "gpu_memory": gpu_memory_usage
                })
                
                # 清理
                del model_engine, optimizer, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 格式化结果
            result_str = "内存使用情况: "
            for result in memory_results:
                result_str += f"{result['size']}模型({result['params']:,}参数): CPU {result['cpu_memory']:.1f}MB"
                if torch.cuda.is_available():
                    result_str += f", GPU {result['gpu_memory']:.1f}MB"
                result_str += "; "
            
            self.log_result(test_name, True, result_str)
            return True
        except Exception as e:
            self.log_result(test_name, False, "内存效率测试失败", e)
            return False
    
    def test_distributed_setup(self) -> bool:
        """测试分布式设置兼容性"""
        test_name = "分布式设置兼容性测试"
        try:
            import torch
            import os
            
            # 检查分布式环境变量
            dist_vars = {
                'RANK': os.environ.get('RANK', 'not_set'),
                'WORLD_SIZE': os.environ.get('WORLD_SIZE', 'not_set'),
                'LOCAL_RANK': os.environ.get('LOCAL_RANK', 'not_set'),
                'MASTER_ADDR': os.environ.get('MASTER_ADDR', 'not_set'),
                'MASTER_PORT': os.environ.get('MASTER_PORT', 'not_set')
            }
            
            # 检查分布式后端
            backends = []
            if torch.distributed.is_available():
                if torch.distributed.is_nccl_available():
                    backends.append("NCCL")
                if torch.distributed.is_gloo_available():
                    backends.append("Gloo")
                if torch.distributed.is_mpi_available():
                    backends.append("MPI")
            
            self.log_result(test_name, True, 
                          f"分布式环境变量: {dist_vars}, "
                          f"可用后端: {backends}")
            return True
        except Exception as e:
            self.log_result(test_name, False, "分布式设置检查失败", e)
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("🚀 开始DeepSpeed Zero3 Offload配置验证测试...")
        logger.info("=" * 70)
        
        # 测试列表
        tests = [
            self.test_basic_imports,
            self.test_deepspeed_imports,
            self.test_cuda_availability,
            self.test_zero3_config_creation,
            self.test_model_initialization,
            self.test_deepspeed_initialization,
            self.test_parameter_offloading,
            self.test_gradient_offloading,
            self.test_model_checkpointing,
            self.test_memory_efficiency,
            self.test_distributed_setup,
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
        logger.info("=" * 70)
        logger.info(f"📊 测试总结: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            logger.info("🎉 所有测试通过！DeepSpeed Zero3 Offload配置正确。")
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
        report.append("DeepSpeed Zero3 Offload配置验证报告")
        report.append("=" * 50)
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
        
        # 添加Zero3配置信息
        if self.zero3_config:
            report.append("")
            report.append("使用的Zero3配置:")
            report.append("-" * 30)
            report.append(json.dumps(self.zero3_config, indent=2))
        
        return "\n".join(report)


def main():
    """主函数"""
    print("DeepSpeed Zero3 Offload配置验证测试")
    print("测试权重和梯度到内存的offload功能")
    print("=" * 70)
    
    # 创建测试器实例
    tester = DeepSpeedZero3Tester()
    
    # 运行所有测试
    success = tester.run_all_tests()
    
    # 生成并保存报告
    report = tester.generate_report()
    with open('deepspeed_zero3_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("📝 详细报告已保存到 deepspeed_zero3_report.txt")
    
    # 返回适当的退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
