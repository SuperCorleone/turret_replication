import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any, List


class DistributedTURRETTrainer:
    """分布式 TURRET 训练器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_size = config.get('world_size', torch.cuda.device_count())
        self.lr = config.get('learning_rate', 1e-3)
        self.optimizer = None

    def setup_distributed(self, rank: int, world_size: int):
        """设置分布式训练环境"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print(f"[Rank {rank}] Initialized process group with world size {world_size}")

    def train_distributed(self, model: nn.Module, dataloader, num_epochs: int):
        """执行分布式训练"""
        model = model.to(torch.cuda.current_device())
        model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(num_epochs):
            for batch in dataloader:
                # 分布式训练步骤
                loss = self.distributed_train_step(model, batch)

                if dist.get_rank() == 0:  # 仅主进程记录
                    self.log_training_progress(epoch, loss.item())

    def distributed_train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步分布式训练逻辑"""
        model.train()

        # 前向传播
        outputs = model(**batch)
        loss = self.compute_loss(outputs, batch)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 同步梯度
        self.sync_gradients(model)

        # 优化器更新
        self.optimizer.step()

        return loss

    def sync_gradients(self, model: nn.Module):
        """同步所有进程的梯度"""
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def compute_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失函数 (可自定义)"""
        # 示例：假设 batch 中包含 'target'
        target = batch.get('target')
        criterion = nn.MSELoss()
        return criterion(outputs, target)

    def log_training_progress(self, epoch: int, loss_value: float):
        """主进程打印训练日志"""
        print(f"[Epoch {epoch}] Loss = {loss_value:.4f}")


# ======================
# 启动多进程训练（示例）
# ======================

def run_worker(rank, world_size, model, dataloader, num_epochs, config):
    trainer = DistributedTURRETTrainer(config)
    trainer.setup_distributed(rank, world_size)
    trainer.train_distributed(model, dataloader, num_epochs)
    dist.destroy_process_group()


def launch_training(model, dataloader, num_epochs=5, config=None):
    """多进程启动入口"""
    if config is None:
        config = {'world_size': torch.cuda.device_count(), 'learning_rate': 1e-3}

    world_size = config['world_size']
    mp.spawn(
        run_worker,
        args=(world_size, model, dataloader, num_epochs, config),
        nprocs=world_size,
        join=True
    )