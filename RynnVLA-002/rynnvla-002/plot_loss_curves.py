#!/usr/bin/env python3
"""
绘制训练和验证loss曲线的脚本

使用方法:
    python plot_loss_curves.py --log_dir <日志目录路径>
    
例如:
    python plot_loss_curves.py --log_dir outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_log_file(log_file_path):
    """从JSON日志文件中加载数据"""
    data = []
    if not os.path.exists(log_file_path):
        return data
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: 跳过无效的JSON行: {e}")
    return data


def plot_loss_curves(log_dir, output_file=None, show_plot=True):
    """绘制loss曲线"""
    
    log_dir = Path(log_dir)
    
    # 加载训练日志
    train_log_file = log_dir / "log_train.txt"
    train_data = load_log_file(train_log_file)
    
    if not train_data:
        print(f"错误: 未找到训练日志文件 {train_log_file}")
        return
    
    # 加载验证日志（如果存在）
    val_ind_log_file = log_dir / "log_eval_ind.txt"
    val_ood_log_file = log_dir / "log_eval_ood.txt"
    val_ind_data = load_log_file(val_ind_log_file)
    val_ood_data = load_log_file(val_ood_log_file)
    
    # 提取epochs
    epochs = [d['epoch'] for d in train_data]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    
    # 1. 主要loss曲线 (closs, loss_ct, z_loss)
    ax1 = axes[0, 0]
    
    # 训练loss
    if 'train_closs' in train_data[0]:
        train_closs = [d['train_closs'] for d in train_data]
        ax1.plot(epochs, train_closs, 'b-', label='Train Closs', linewidth=2)
    
    if 'train_loss_ct' in train_data[0]:
        train_loss_ct = [d['train_loss_ct'] for d in train_data]
        ax1.plot(epochs, train_loss_ct, 'g-', label='Train Loss CT', linewidth=2)
    
    if 'train_z_loss' in train_data[0]:
        train_z_loss = [d['train_z_loss'] for d in train_data]
        ax1.plot(epochs, train_z_loss, 'r-', label='Train Z Loss', linewidth=2, alpha=0.7)
    
    # 验证loss
    if val_ind_data and 'val_closs' in val_ind_data[0]:
        val_ind_epochs = [d['epoch'] for d in val_ind_data]
        val_ind_closs = [d['val_closs'] for d in val_ind_data]
        ax1.plot(val_ind_epochs, val_ind_closs, 'b--', label='Val (ID) Closs', linewidth=2, alpha=0.7)
    
    if val_ood_data and 'val_closs' in val_ood_data[0]:
        val_ood_epochs = [d['epoch'] for d in val_ood_data]
        val_ood_closs = [d['val_closs'] for d in val_ood_data]
        ax1.plot(val_ood_epochs, val_ood_closs, 'r--', label='Val (OOD) Closs', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Main Loss Components', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Action L1 Loss (平均所有action timesteps)
    ax2 = axes[0, 1]
    
    # 提取所有action l1 loss
    action_loss_keys = [k for k in train_data[0].keys() if k.startswith('train_l1_loss_action_')]
    if action_loss_keys:
        # 计算平均action loss
        train_action_loss = []
        for d in train_data:
            action_losses = [d[k] for k in action_loss_keys if k in d]
            if action_losses:
                train_action_loss.append(np.mean(action_losses))
            else:
                train_action_loss.append(None)
        
        # 过滤None值
        valid_epochs = [e for e, v in zip(epochs, train_action_loss) if v is not None]
        valid_losses = [v for v in train_action_loss if v is not None]
        ax2.plot(valid_epochs, valid_losses, 'b-', label='Train Avg Action L1 Loss', linewidth=2)
    
    # 验证action loss
    if val_ind_data:
        val_action_loss_keys = [k for k in val_ind_data[0].keys() if k.startswith('val_l1_loss_action_')]
        if val_action_loss_keys:
            val_ind_action_loss = []
            val_ind_epochs_list = [d['epoch'] for d in val_ind_data]
            for d in val_ind_data:
                action_losses = [d[k] for k in val_action_loss_keys if k in d]
                if action_losses:
                    val_ind_action_loss.append(np.mean(action_losses))
                else:
                    val_ind_action_loss.append(None)
            valid_val_epochs = [e for e, v in zip(val_ind_epochs_list, val_ind_action_loss) if v is not None]
            valid_val_losses = [v for v in val_ind_action_loss if v is not None]
            ax2.plot(valid_val_epochs, valid_val_losses, 'b--', label='Val (ID) Avg Action L1 Loss', linewidth=2, alpha=0.7)
    
    if val_ood_data:
        val_ood_action_loss_keys = [k for k in val_ood_data[0].keys() if k.startswith('val_l1_loss_action_')]
        if val_ood_action_loss_keys:
            val_ood_action_loss = []
            val_ood_epochs_list = [d['epoch'] for d in val_ood_data]
            for d in val_ood_data:
                action_losses = [d[k] for k in val_ood_action_loss_keys if k in d]
                if action_losses:
                    val_ood_action_loss.append(np.mean(action_losses))
                else:
                    val_ood_action_loss.append(None)
            valid_ood_epochs = [e for e, v in zip(val_ood_epochs_list, val_ood_action_loss) if v is not None]
            valid_ood_losses = [v for v in val_ood_action_loss if v is not None]
            ax2.plot(valid_ood_epochs, valid_ood_losses, 'r--', label='Val (OOD) Avg Action L1 Loss', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('L1 Loss', fontsize=12)
    ax2.set_title('Action L1 Loss (Averaged)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 学习率和梯度范数
    ax3 = axes[1, 0]
    
    if 'train_lr' in train_data[0]:
        train_lr = [d['train_lr'] for d in train_data]
        ax3_twin = ax3.twinx()
        ax3_twin.plot(epochs, train_lr, 'g-', label='Learning Rate', linewidth=2, alpha=0.7)
        ax3_twin.set_ylabel('Learning Rate', fontsize=12, color='g')
        ax3_twin.tick_params(axis='y', labelcolor='g')
        ax3_twin.legend(loc='upper right', fontsize=10)
    
    if 'train_grad_norm' in train_data[0]:
        train_grad_norm = [d['train_grad_norm'] for d in train_data]
        ax3.plot(epochs, train_grad_norm, 'r-', label='Gradient Norm', linewidth=2)
        ax3.set_ylabel('Gradient Norm', fontsize=12, color='r')
        ax3.tick_params(axis='y', labelcolor='r')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_title('Learning Rate and Gradient Norm', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy指标（如果有）
    ax4 = axes[1, 1]
    
    # 提取accuracy keys（排除异常值）
    acc_keys = [k for k in train_data[0].keys() if k.startswith('train_acc_')]
    if acc_keys:
        # 只绘制合理的accuracy值（0-1范围或较小的值）
        plotted = False
        for key in acc_keys[:3]:  # 只绘制前3个accuracy指标
            try:
                acc_values = [d[key] for d in train_data]
                # 过滤异常值（大于1000的可能是bug）
                valid_acc = [(e, v) for e, v in zip(epochs, acc_values) if v is not None and v < 1000 and v >= 0]
                if valid_acc:
                    valid_epochs, valid_values = zip(*valid_acc)
                    ax4.plot(valid_epochs, valid_values, label=key.replace('train_', ''), linewidth=2, alpha=0.7)
                    plotted = True
            except Exception as e:
                continue
        
        if plotted:
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Accuracy', fontsize=12)
            ax4.set_title('Training Accuracy Metrics', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No valid accuracy data', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Training Accuracy Metrics', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No accuracy data available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training Accuracy Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    if output_file is None:
        output_file = log_dir / "loss_curves.png"
    else:
        output_file = Path(output_file)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loss曲线已保存到: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制训练和验证loss曲线')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='日志文件目录路径（包含log_train.txt的目录）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（默认：log_dir/loss_curves.png）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图表（仅保存文件）')
    
    args = parser.parse_args()
    
    plot_loss_curves(args.log_dir, args.output, show_plot=not args.no_show)


if __name__ == '__main__':
    main()

