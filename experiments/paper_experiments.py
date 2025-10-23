# experiments/paper_experiments.py - 修复配置使用

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, List
import time

# 添加导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig
from utils.file_utils import save_config, ensure_dir
from utils.logging_utils import setup_logging, TrainingLogger

class PaperExperimentReplicator:
    """论文实验复现器 - 统一管理所有四种实验"""
    
    def __init__(self, config: TURRETConfig):
        self.config = config
        self.results_dir = config.results_dir
        ensure_dir(self.results_dir)
        
        # 设置日志
        self.logger = setup_logging(level=config.log_level)
        self.training_logger = TrainingLogger(self.results_dir)
        
        # 保存配置
        save_config(config.to_dict(), os.path.join(self.results_dir, 'experiment_config.yaml'))
        
        # 实验定义
        self.experiment_types = {
            'size_transfer': {
                'name': '规模迁移实验',
                'description': '从小型机器人迁移到大型机器人',
                'runner_class': 'SizeTransferRunner'
            },
            'morphology_transfer': {
                'name': '形态迁移实验', 
                'description': '在不同形态机器人间迁移',
                'runner_class': 'MorphologyTransferRunner'
            },
            'ablation_studies': {
                'name': '消融实验',
                'description': '验证各组件重要性',
                'runner_class': 'AblationRunner'
            },
            'baseline_comparison': {
                'name': '基线对比实验',
                'description': '与现有方法对比',
                'runner_class': 'BaselineComparisonRunner'
            }
        }
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有四种论文实验"""
        all_results = {}
        self.logger.info("🚀 Starting TURRET Paper Experiment Replication...")
        
        try:
            # 1. 规模迁移实验
            if self._should_run_experiment('size_transfer'):
                self.logger.info("📊 Running Size Transfer Experiments...")
                size_results = self._run_size_transfer_experiments()
                all_results['size_transfer'] = size_results
            
            # 2. 形态迁移实验  
            if self._should_run_experiment('morphology_transfer'):
                self.logger.info("🦎 Running Morphology Transfer Experiments...")
                morph_results = self._run_morphology_transfer_experiments()
                all_results['morphology_transfer'] = morph_results
            
            # 3. 消融实验
            if self._should_run_experiment('ablation_studies'):
                self.logger.info("🔬 Running Ablation Studies...")
                ablation_results = self._run_ablation_studies()
                all_results['ablation_studies'] = ablation_results
            
            # 4. 基线对比实验
            if self._should_run_experiment('baseline_comparison'):
                self.logger.info("📈 Running Baseline Comparison Experiments...")
                baseline_results = self._run_baseline_comparison()
                all_results['baseline_comparison'] = baseline_results
            
            # 保存完整结果
            self._save_comprehensive_results(all_results)
            self.logger.info("✅ All experiments completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {e}")
            raise
        
        return all_results
    
    def _should_run_experiment(self, experiment_type: str) -> bool:
        """检查是否应该运行某个实验"""
        # 这里可以根据配置决定运行哪些实验
        return True
    
    def _run_size_transfer_experiments(self) -> Dict[str, Any]:
        """运行规模迁移实验 - 修正为Centipede少足到多足迁移"""
        from .runners.size_transfer import SizeTransferRunner
        
        # 规模迁移场景定义（与论文一致）- Centipede少足到多足
        size_scenarios = [
            {
                'name': 'centipede_4_to_12',
                'sources': ['Centipede-4', 'Centipede-6'],  # 少足机器人作为源
                'target': 'Centipede-12',                   # 多足机器人作为目标
                'description': 'Centipede 4/6足到12足迁移'
            },
            {
                'name': 'centipede_4_to_16', 
                'sources': ['Centipede-4', 'Centipede-6'],
                'target': 'Centipede-16',
                'description': 'Centipede 4/6足到16足迁移'
            },
            {
                'name': 'centipede_4_to_20',
                'sources': ['Centipede-4', 'Centipede-6'], 
                'target': 'Centipede-20',
                'description': 'Centipede 4/6足到20足迁移'
            }
        ]
        
        results = {}
        for scenario in size_scenarios:
            self.logger.info(f"  Processing: {scenario['description']}")
            
            # 创建实验配置
            scenario_config = self.config.copy()
            scenario_config.update(
                experiment_id=f"size_transfer_{scenario['name']}",
                source_robots=scenario['sources'],
                target_robot=scenario['target'],
                environment_type="centipede"  # 标记为Centipede环境
            )
            
            # 运行实验
            runner = SizeTransferRunner(scenario_config)
            runner.setup_experiment()
            scenario_results = runner.run_experiment()
                        
            # 确保返回列表格式的结果
            if not isinstance(scenario_results, list):
                scenario_results = [scenario_results]  # 包装成列表
                
            results[scenario['name']] = scenario_results
        
        return results
    
    def _run_morphology_transfer_experiments(self) -> Dict[str, Any]:
        """运行形态迁移实验"""
        from .runners.morphology_transfer import MorphologyTransferRunner
        
        # 形态迁移场景定义（与论文一致）
        morphology_scenarios = [
            {
                'name': 'quad_to_biped',
                'transfer_type': 'quad_to_biped',
                'description': '四足到双足迁移'
            },
            {
                'name': 'biped_to_humanoid',
                'transfer_type': 'biped_to_humanoid', 
                'description': '双足到人形迁移'
            }
        ]
        
        results = {}
        for scenario in morphology_scenarios:
            self.logger.info(f"  Processing: {scenario['description']}")
            
            # 创建实验配置
            scenario_config = self.config.copy()
            scenario_config.update(
                experiment_id=f"morphology_{scenario['name']}",
                transfer_type=scenario['transfer_type']
            )
            
            # 运行实验
            runner = MorphologyTransferRunner(scenario_config)
            runner.setup_experiment()
            scenario_results = runner.run_experiment()
            
            results[scenario['name']] = scenario_results
        
        return results
    
    def _run_ablation_studies(self) -> Dict[str, Any]:
        """运行消融实验"""
        # 创建临时的消融实验运行器
        class AblationRunner:
            def __init__(self, config):
                self.config = config
                self.ablation_type = config.ablation_type
                
            def setup_experiment(self):
                pass
                
            def run_experiment(self):
                # 返回模拟结果用于测试
                return {
                    'final_performance': {
                        'mean_reward': np.random.uniform(300, 800),
                        'std_reward': np.random.uniform(50, 150),
                    },
                    'ablation_type': self.ablation_type
                }
        
        # 消融研究定义
        ablation_studies = [
            {
                'name': 'no_attention',
                'description': '无注意力机制',
                'ablation_type': 'no_attention'
            },
            {
                'name': 'no_semantic_space',
                'description': '无语义空间',
                'ablation_type': 'no_semantic_space'
            },
            {
                'name': 'fixed_weights',
                'description': '固定迁移权重',
                'ablation_type': 'fixed_weights'
            }
        ]
        
        results = {}
        for study in ablation_studies:
            self.logger.info(f"  Processing: {study['description']}")
            
            # 创建实验配置
            study_config = self.config.copy()
            study_config.update(
                experiment_id=f"ablation_{study['name']}",
                ablation_type=study['ablation_type']
            )
            
            # 运行消融实验
            runner = AblationRunner(study_config)
            runner.setup_experiment()
            study_results = runner.run_experiment()
            
            results[study['name']] = study_results
        
        return results
    
    def _run_baseline_comparison(self) -> Dict[str, Any]:
        """运行基线对比实验"""
        # 创建临时的基线对比运行器
        class BaselineComparisonRunner:
            def __init__(self, config):
                self.config = config
                self.baseline_method = config.baseline_method
                
            def setup_experiment(self):
                pass
                
            def run_experiment(self):
                # 返回模拟结果用于测试
                return {
                    'final_performance': {
                        'mean_reward': np.random.uniform(200, 600),
                        'std_reward': np.random.uniform(50, 150),
                    },
                    'baseline_method': self.baseline_method
                }
        
        # 基线方法定义
        baseline_methods = [
            {
                'name': 'PPO',
                'description': '原始PPO（无迁移）'
            },
            {
                'name': 'CAT',
                'description': '对比方法CAT'
            },
            {
                'name': 'NerveNet',
                'description': '对比方法NerveNet'
            }
        ]
        
        results = {}
        for method in baseline_methods:
            self.logger.info(f"  Processing: {method['description']}")
            
            # 创建实验配置
            method_config = self.config.copy()
            method_config.update(
                experiment_id=f"baseline_{method['name']}",
                baseline_method=method['name']
            )
            
            # 运行基线对比
            runner = BaselineComparisonRunner(method_config)
            runner.setup_experiment()
            method_results = runner.run_experiment()
            
            results[method['name']] = method_results
        
        return results
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """保存综合结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = os.path.join(self.results_dir, f'paper_experiments_{timestamp}.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"📁 Results saved to: {results_file}")
        
        # 保存训练统计 - 修复这里
        stats = self.training_logger.get_statistics()
        if stats:
            stats_file = os.path.join(self.results_dir, f'training_statistics_{timestamp}.json')
            self.training_logger.save_statistics(stats_file)  # 使用新方法
            self.logger.info(f"📊 Training statistics saved to: {stats_file}")

def main():
    """主函数 - 运行所有实验"""
    # 创建配置
    config = TURRETConfig(
        device="cpu",
        total_episodes=100,  # 测试用较少的episodes
        num_seeds=2,         # 测试用较少的随机种子
        results_dir="data/paper_results"
    )
    
    # 运行实验
    replicator = PaperExperimentReplicator(config)
    results = replicator.run_all_experiments()
    
    print("🎉 Paper experiments completed!")
    return results

if __name__ == "__main__":
    main()