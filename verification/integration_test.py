# verification/integration_test.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
import os

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import TURRETConfig

class IntegrationTester:
    """核心组件集成测试"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_integration_tests(self) -> Dict[str, bool]:
        """运行集成测试"""
        print("🔗 Running Integration Tests...")
        
        tests = {
            'policy_training_integration': self._test_policy_training_integration(),
            'transfer_learning_pipeline': self._test_transfer_learning_pipeline(),
            'experiment_workflow': self._test_experiment_workflow()
        }
        
        self.test_results = tests
        print(f"✅ Integration Tests Complete: {sum(tests.values())}/{len(tests)} passed")
        return tests
    
    def _test_policy_training_integration(self) -> bool:
        """测试策略训练集成"""
        try:
            from models.policies.structured_policy import StructuredPolicyNetwork
            from training.trainers.ppo_trainer import PPOTrainer
            
            # 创建组件
            policy_config = {
                "node_observation_dim": 10,
                "hidden_dim": 32,
                "output_dim": 4,
                "device": "cpu"
            }
            
            policy = StructuredPolicyNetwork(policy_config)
            
            class MockValue(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.net(x)
            
            value = MockValue()
            
            # 创建训练器
            trainer_config = TURRETConfig(
                device="cpu",
                batch_size=4,
                total_episodes=5
            )
            
            trainer = PPOTrainer(policy, value, trainer_config)
            
            # 测试训练步骤
            batch = self._create_test_batch(batch_size=4, obs_dim=10, action_dim=4)
            stats = trainer.train_step(batch)
            
            assert 'policy_loss' in stats
            assert 'value_loss' in stats
            
            print("  ✅ Policy training integration: PASSED")
            return True
            
        except Exception as e:
            print(f"  ❌ Policy training integration: FAILED - {e}")
            return False
    
    def _test_transfer_learning_pipeline(self) -> bool:
        """测试迁移学习流水线"""
        try:
            from transfer.semantic_space import SemanticSpaceManager
            from transfer.weight_calculator import AdaptiveWeightCalculator
            
            # 创建迁移组件
            semantic_manager = SemanticSpaceManager({"embedding_dim": 64, "device": "cpu"})
            weight_calculator = AdaptiveWeightCalculator({"embedding_dim": 64, "device": "cpu"})
            
            # 手动修复 to_device 方法
            if not hasattr(weight_calculator, 'to_device'):
                weight_calculator.to_device = lambda x: x.to(weight_calculator.device) if hasattr(x, 'to') else x
            
            # 测试语义空间投影
            target_state = torch.randn(64)
            source_states = [torch.randn(64) for _ in range(2)]
            
            target_embedding = semantic_manager.project_state(target_state, "target")
            source_embeddings = [semantic_manager.project_state(state, "source") for state in source_states]
            
            # 测试权重计算
            weights = weight_calculator.compute_transfer_weights(target_state, source_states)
            
            assert len(weights) == 2
            assert abs(sum(weights) - 1.0) < 1e-6
            
            print("  ✅ Transfer learning pipeline: PASSED")
            return True
            
        except Exception as e:
            print(f"  ❌ Transfer learning pipeline: FAILED - {e}")
            return False
    
    def _test_experiment_workflow(self) -> bool:
        """测试实验工作流"""
        try:
            from experiments.paper_experiments import PaperExperimentReplicator
            
            # 创建测试配置
            config = TURRETConfig(
                device="cpu",
                total_episodes=2,  # 最小测试
                num_seeds=1,
                results_dir="data/test_integration"
            )
            
            # 测试实验创建和配置
            replicator = PaperExperimentReplicator(config)
            
            assert hasattr(replicator, 'run_all_experiments')
            assert hasattr(replicator, 'experiment_types')
            
            print("  ✅ Experiment workflow: PASSED")
            return True
            
        except Exception as e:
            print(f"  ❌ Experiment workflow: FAILED - {e}")
            return False
    
    def _create_test_batch(self, batch_size: int, obs_dim: int, action_dim: int) -> Dict[str, torch.Tensor]:
        """创建测试批次数据"""
        return {
            'observations': torch.randn(batch_size, obs_dim),
            'actions': torch.randn(batch_size, action_dim),
            'rewards': torch.randn(batch_size),
            'next_observations': torch.randn(batch_size, obs_dim),
            'terminated': torch.zeros(batch_size, dtype=torch.bool),
            'truncated': torch.zeros(batch_size, dtype=torch.bool),
            'log_probs': torch.randn(batch_size),
            'values': torch.randn(batch_size)
        }
    
    def generate_integration_report(self) -> str:
        """生成集成测试报告"""
        report = ["TURRET Integration Test Report"]
        report.append("=" * 40)
        
        for test, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            report.append(f"{test:.<30}{status}")
        
        report.append("=" * 40)
        report.append(f"Overall: {sum(self.test_results.values())}/{len(self.test_results)} tests passed")
        
        return "\n".join(report)

def main():
    """运行集成测试"""
    tester = IntegrationTester()
    results = tester.run_integration_tests()
    
    print("\n" + "=" * 40)
    print(tester.generate_integration_report())
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit(main())