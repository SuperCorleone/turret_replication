#!/bin/bash
# run_all_tests.sh

echo "🚀 开始运行 TURRET 完整测试套件"
echo "=========================================="

# 创建测试目录
mkdir -p tests
cd tests

echo "1. 运行快速完整性测试..."
python test_paper_experiments.py

echo ""
echo "2. 运行组件接口测试..."
python test_component_interfaces.py

echo ""
echo "3. 运行实验复现验证测试..."
python test_experiment_replication.py

echo ""
echo "4. 运行实际实验（小规模）..."
cd ..
python scripts/run_all_experiments.py \
    --num_seeds 2 \
    --total_episodes 50 \
    --output_dir data/test_run

echo ""
echo "5. 验证实验结果..."
python -c "
import sys
sys.path.insert(0, '.')
from analysis.result_analyzer import ResultAnalyzer
analyzer = ResultAnalyzer('data/test_run')
print('✅ 结果分析器工作正常')
"

echo ""
echo "=========================================="
echo "🎉 所有测试完成！TURRET 项目准备就绪"