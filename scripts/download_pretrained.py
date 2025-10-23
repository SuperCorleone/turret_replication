# scripts/download_pretrained.py - 修复版本
import os
import requests
import torch
import zipfile
import gdown
from pathlib import Path
import json
import shutil

def download_ppo_models():
    """下载PPO预训练模型 - 修复版本"""
    
    # 使用更可靠的模型源
    ppo_sources = {
        'HalfCheetah': 'https://huggingface.co/edbeeching/ppo-HalfCheetah-v3/resolve/main/ppo-HalfCheetah-v3.pt',
        'Ant': 'https://huggingface.co/edbeeching/ppo-Ant-v3/resolve/main/ppo-Ant-v3.pt',
        'Hopper': 'https://huggingface.co/edbeeching/ppo-Hopper-v3/resolve/main/ppo-Hopper-v3.pt',
        'Walker2d': 'https://huggingface.co/edbeeching/ppo-Walker2d-v3/resolve/main/ppo-Walker2d-v3.pt'
    }
    
    save_dir = Path("data/pretrained/source_policies")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 第一步：备份或删除现有的测试策略
    backup_test_policies(save_dir)
    
    for robot_type, url in ppo_sources.items():
        filename = save_dir / f"{robot_type.lower()}_policy.pth"
        temp_filename = save_dir / f"{robot_type.lower()}_downloading.pth"
        
        # 检查是否已经是真实模型（不是测试策略）
        if filename.exists():
            if is_real_model(filename):
                print(f"✓ {robot_type} real model already exists, skipping download")
                continue
            else:
                print(f"🔄 {robot_type} test model exists, replacing with real model")
                # 备份测试模型
                backup_path = save_dir / f"{robot_type.lower()}_test_backup.pth"
                shutil.move(filename, backup_path)
                print(f"  Backed up test model to {backup_path}")
        
        print(f"Downloading {robot_type} model...")
        try:
            # 使用requests下载
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 保存到临时文件
            with open(temp_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 验证文件
            if validate_model_file(temp_filename):
                # 重命名为正式文件
                shutil.move(temp_filename, filename)
                print(f"✓ Downloaded {robot_type} model successfully")
                
                # 创建兼容版本
                create_compatible_model(robot_type, filename)
            else:
                print(f"⚠️  {robot_type} model validation failed")
                if filename.exists():
                    os.remove(filename)
                create_fallback_model(robot_type, filename)
                
        except Exception as e:
            print(f"✗ Failed to download {robot_type}: {e}")
            # 清理临时文件
            if temp_filename.exists():
                os.remove(temp_filename)
            # 恢复测试模型或创建回退
            if not filename.exists():
                restore_test_policy(robot_type, save_dir)

def is_real_model(filepath: Path) -> bool:
    """检查是否是真实模型（不是测试策略）"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # 测试策略的标识
        if isinstance(checkpoint, dict):
            if checkpoint.get('is_test_policy', False):
                return False
            if checkpoint.get('is_fallback', False):
                return False
            # 真实模型应该有实际的网络权重
            if 'actor' in checkpoint or 'policy' in checkpoint or 'model_state_dict' in checkpoint:
                return True
        
        return True  # 默认认为是真实模型
    except:
        return False

def backup_test_policies(save_dir: Path):
    """备份现有的测试策略"""
    print("Checking for existing test policies...")
    
    test_robots = ['halfcheetah', 'ant', 'hopper', 'walker2d']
    
    for robot in test_robots:
        policy_path = save_dir / f"{robot}_policy.pth"
        if policy_path.exists() and is_test_policy(policy_path):
            backup_path = save_dir / f"{robot}_test_backup.pth"
            shutil.copy2(policy_path, backup_path)
            print(f"  Backed up test policy for {robot}")

def is_test_policy(filepath: Path) -> bool:
    """检查是否是测试策略"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        return checkpoint.get('is_test_policy', False) or checkpoint.get('is_fallback', False)
    except:
        return False

def restore_test_policy(robot_type: str, save_dir: Path):
    """恢复测试策略（当下载失败时）"""
    backup_path = save_dir / f"{robot_type.lower()}_test_backup.pth"
    policy_path = save_dir / f"{robot_type.lower()}_policy.pth"
    
    if backup_path.exists():
        shutil.copy2(backup_path, policy_path)
        print(f"  Restored test policy for {robot_type}")
    else:
        # 创建新的测试策略
        create_test_policy(robot_type, save_dir)

def create_test_policy(robot_type: str, save_dir: Path):
    """创建单个测试策略"""
    filename = save_dir / f"{robot_type.lower()}_policy.pth"
    
    # 创建简单的测试策略
    test_policy = {
        'policy_state_dict': {
            'propagation_network.0.weight': torch.randn(256, 17) * 0.1,
            'propagation_network.0.bias': torch.randn(256) * 0.1,
            'propagation_network.2.weight': torch.randn(256, 256) * 0.1,
            'propagation_network.2.bias': torch.randn(256) * 0.1,
            'simple_output.0.weight': torch.randn(256, 256) * 0.1,
            'simple_output.0.bias': torch.randn(256) * 0.1,
            'simple_output.2.weight': torch.randn(12, 256) * 0.1,
            'simple_output.2.bias': torch.randn(12) * 0.1,
        },
        'robot_type': robot_type,
        'is_test_policy': True,
        'performance_level': 'basic'
    }
    
    torch.save(test_policy, filename)
    print(f"✓ Created test policy for {robot_type}")

# 其他函数保持不变...
def validate_model_file(filepath: Path) -> bool:
    """验证模型文件格式"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            valid_keys = ['actor', 'critic', 'policy', 'model_state_dict', 'state_dict']
            if any(key in checkpoint for key in valid_keys):
                return True
        
        return False
        
    except Exception as e:
        print(f"  Validation failed: {e}")
        return False

def create_compatible_model(robot_type: str, original_filepath: Path):
    """创建兼容的模型格式"""
    try:
        original_checkpoint = torch.load(original_filepath, map_location='cpu', weights_only=False)
        
        compatible_checkpoint = {
            'policy_state_dict': extract_policy_weights(original_checkpoint),
            'robot_type': robot_type,
            'source': 'huggingface',
            'compatible_format': True
        }
        
        compatible_path = original_filepath.parent / f"{robot_type.lower()}_compatible.pth"
        torch.save(compatible_checkpoint, compatible_path)
        print(f"✓ Created compatible model for {robot_type}")
        
    except Exception as e:
        print(f"✗ Failed to create compatible model: {e}")

def extract_policy_weights(checkpoint: dict) -> dict:
    """从检查点提取策略权重"""
    policy_weights = {}
    
    if 'actor' in checkpoint:
        actor_weights = checkpoint['actor']
        for key, value in actor_weights.items():
            if 'pi' in key or 'actor' in key:
                new_key = key.replace('pi_features_extractor.', '').replace('pi_', '')
                policy_weights[new_key] = value
                
    elif 'policy' in checkpoint:
        policy_weights = checkpoint['policy']
        
    elif 'model_state_dict' in checkpoint:
        policy_weights = checkpoint['model_state_dict']
        
    else:
        policy_weights = checkpoint
    
    return policy_weights

def create_fallback_model(robot_type: str, filepath: Path):
    """创建回退模型"""
    print(f"  Creating fallback model for {robot_type}")
    
    fallback_weights = {
        'propagation_network.0.weight': torch.randn(256, 17) * 0.01,
        'propagation_network.0.bias': torch.randn(256) * 0.01,
        'propagation_network.2.weight': torch.randn(256, 256) * 0.01,
        'propagation_network.2.bias': torch.randn(256) * 0.01,
        'simple_output.0.weight': torch.randn(256, 256) * 0.01,
        'simple_output.0.bias': torch.randn(256) * 0.01,
        'simple_output.2.weight': torch.randn(12, 256) * 0.01,
        'simple_output.2.bias': torch.randn(12) * 0.01,
        'robot_type': robot_type,
        'is_fallback': True
    }
    
    torch.save(fallback_weights, filepath)

def verify_pretrained_models():
    """验证预训练模型"""
    models_dir = Path("data/pretrained/source_policies")
    
    expected_models = ['halfcheetah', 'ant', 'hopper', 'walker2d']
    
    print("\n🔍 Verifying models...")
    for model_name in expected_models:
        model_path = models_dir / f"{model_name}_policy.pth"
        
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if checkpoint.get('is_test_policy', False):
                    print(f"⚠️  {model_name}: TEST POLICY (limited performance)")
                elif checkpoint.get('is_fallback', False):
                    print(f"⚠️  {model_name}: FALLBACK MODEL (basic functionality)")
                elif checkpoint.get('compatible_format', False):
                    print(f"✓ {model_name}: REAL MODEL (compatible format)")
                elif 'actor' in checkpoint or 'policy' in checkpoint:
                    print(f"✓ {model_name}: REAL MODEL (original format)")
                else:
                    print(f"❓ {model_name}: Unknown format")
                    
            except Exception as e:
                print(f"✗ {model_name}: Corrupted file - {e}")
        else:
            print(f"✗ {model_name}: Missing file")

if __name__ == "__main__":
    
    print("🚀 Starting pretrained model setup...")
    
    # 直接下载真实模型，不先创建测试策略
    download_ppo_models()
    
    # 验证所有模型
    verify_pretrained_models()
    
    print("Models are ready for use!")