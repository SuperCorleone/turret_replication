import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple
import pandas as pd

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def analyze_experiment_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验结果"""
        analysis = {}
        
        for experiment_type, experiment_results in results.items():
            analysis[experiment_type] = self._analyze_single_experiment(experiment_results)
        
        return analysis
    
    def _analyze_single_experiment(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个实验"""
        analysis = {}
        
        for scenario, scenario_results in experiment_results.items():
            if isinstance(scenario_results, list) and scenario_results:
                # 提取最终性能指标
                final_rewards = []
                for result in scenario_results:
                    if 'final_performance' in result:
                        final_rewards.append(result['final_performance']['mean_reward'])
                
                if final_rewards:
                    analysis[scenario] = self._compute_descriptive_statistics(final_rewards)
        
        return analysis
    
    def _compute_descriptive_statistics(self, data: List[float]) -> Dict[str, float]:
        """计算描述性统计"""
        data_array = np.array(data)
        
        return {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'median': float(np.median(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'confidence_interval': self._compute_confidence_interval(data_array),
            'effect_size': self._compute_effect_size(data_array),
            'normality_test': self._test_normality(data_array)
        }
    
    def _compute_confidence_interval(self, data: np.ndarray) -> Tuple[float, float]:
        """计算置信区间"""
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        
        if n > 1:
            h = sem * stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
            return float(mean - h), float(mean + h)
        else:
            return float(mean), float(mean)
    
    def _compute_effect_size(self, data: np.ndarray) -> float:
        """计算效应大小（Cohen's d）"""
        if len(data) < 2:
            return 0.0
        
        # 与零的效应大小
        return float(np.mean(data) / np.std(data))
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, float]:
        """正态性检验"""
        if len(data) < 3:
            return {'shapiro_p': 1.0, 'normal': True}
        
        try:
            _, p_value = stats.shapiro(data)
            return {'shapiro_p': float(p_value), 'normal': p_value > 0.05}
        except:
            return {'shapiro_p': 1.0, 'normal': True}
    
    def compare_methods_statistical(self, method_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """方法间统计比较"""
        methods = list(method_results.keys())
        comparison = {}
        
        # 成对t检验
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    data1 = method_results[method1]
                    data2 = method_results[method2]
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        comparison[f"{method1}_vs_{method2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
        
        return comparison