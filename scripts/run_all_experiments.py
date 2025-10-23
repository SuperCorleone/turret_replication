import sys
import os
import argparse
import time

# 添加导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig
from experiments.paper_experiments import PaperExperimentReplicator
from analysis.result_analyzer import ResultAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Run TURRET Paper Experiments')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['size', 'morphology', 'ablation', 'baseline', 'all'],
                       default=['all'], help='Experiments to run')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--total_episodes', type=int, default=500, help='Total episodes per experiment')
    parser.add_argument('--output_dir', type=str, default='data/paper_results', 
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("🚀 Starting TURRET Paper Experiment Replication")
    print("=" * 60)
    
    # 创建配置
    config = TURRETConfig(
        device=args.device,
        total_episodes=args.total_episodes,
        num_seeds=args.num_seeds,
        results_dir=args.output_dir
    )
    
    start_time = time.time()
    
    try:
        # 运行论文实验
        print("\n📊 Running Paper Experiments...")
        replicator = PaperExperimentReplicator(config)
        results = replicator.run_all_experiments()
        
        # 结果分析
        print("\n📈 Analyzing Results...")
        analyzer = ResultAnalyzer(config.results_dir)
        analysis = analyzer.generate_comprehensive_analysis(results)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ All experiments completed in {elapsed_time:.2f} seconds")
        print(f"📁 Results saved to: {config.results_dir}")
        
    except Exception as e:
        print(f"❌ Experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())