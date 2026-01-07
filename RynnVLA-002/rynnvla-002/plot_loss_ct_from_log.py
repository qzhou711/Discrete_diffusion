#!/usr/bin/env python3
"""
从 output.log 文件中提取 loss_ct 数据并绘制随 epoch 变化的曲线图
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_output_log(log_file_path):
    """
    从 output.log 文件中解析 loss_ct 和 epoch 信息
    
    Returns:
        epochs: list of epoch numbers
        loss_cts: list of loss_ct values (每个epoch的所有batch值的列表)
    """
    epochs = []
    loss_ct_by_epoch = {}
    
    # 匹配模式: Epoch: [X]  [Y/Z] ... loss_ct: value (average)
    pattern = r'Epoch:\s+\[(\d+)\].*?loss_ct:\s+([\d.]+)\s+\(([\d.]+)\)'
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                current_loss_ct = float(match.group(2))
                avg_loss_ct = float(match.group(3))
                
                if epoch not in loss_ct_by_epoch:
                    loss_ct_by_epoch[epoch] = []
                    epochs.append(epoch)
                
                # 使用平均值（更准确反映整个epoch的进度）
                loss_ct_by_epoch[epoch].append(avg_loss_ct)
    
    # 对每个epoch的数据，取最后一个值（该epoch的平均值）
    epoch_list = sorted(set(epochs))
    loss_ct_list = []
    
    for epoch in epoch_list:
        if loss_ct_by_epoch[epoch]:
            # 取该epoch最后一个batch的平均值（这应该是最接近epoch平均值的）
            loss_ct_list.append(loss_ct_by_epoch[epoch][-1])
        else:
            loss_ct_list.append(np.nan)
    
    return epoch_list, loss_ct_list


def plot_loss_ct_curve(epochs, loss_cts, output_path="loss_ct_curve.png", show_plot=True):
    """
    绘制 loss_ct 随 epoch 变化的曲线图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(epochs, loss_cts, marker='o', linestyle='-', linewidth=2, markersize=6, 
             color='green', label='Train Loss_CT (Continuous Token Loss)')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss_CT (Continuous Token Loss)', fontsize=12)
    plt.title('Continuous Token Loss (Loss_CT) vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加数值标注（如果数据点不多的话）
    if len(epochs) <= 20:
        for i, (epoch, loss_ct) in enumerate(zip(epochs, loss_cts)):
            if not np.isnan(loss_ct):
                plt.annotate(f'{loss_ct:.3f}', 
                           (epoch, loss_ct), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    # 设置文件路径
    script_dir = Path(__file__).parent
    log_file = script_dir / "outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw/output.log"
    output_image = script_dir / "outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw/loss_ct_curve.png"
    
    # 如果文件不存在，尝试从当前目录查找
    if not log_file.exists():
        log_file = Path("outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw/output.log")
        if not log_file.exists():
            print(f"错误: 找不到日志文件 {log_file}")
            print("请确保 output.log 文件存在")
            return
    
    print(f"正在读取日志文件: {log_file}")
    
    # 解析日志文件
    epochs, loss_cts = parse_output_log(log_file)
    
    if not epochs:
        print("错误: 未能从日志文件中提取到任何数据")
        print("请检查日志文件格式是否正确")
        return
    
    print(f"成功提取 {len(epochs)} 个 epoch 的数据")
    print(f"Epoch 范围: {min(epochs)} - {max(epochs)}")
    
    valid_loss_cts = [l for l in loss_cts if not np.isnan(l)]
    if valid_loss_cts:
        print(f"Loss_CT 范围: {min(valid_loss_cts):.4f} - {max(valid_loss_cts):.4f}")
    
    # 显示数据摘要
    print("\n数据摘要:")
    print("Epoch | Loss_CT")
    print("-" * 25)
    for epoch, loss_ct in zip(epochs, loss_cts):
        if not np.isnan(loss_ct):
            print(f"{epoch:5d} | {loss_ct:.4f}")
    
    # 绘制曲线
    plot_loss_ct_curve(epochs, loss_cts, output_path=str(output_image), show_plot=False)
    
    print(f"\n完成！图表已保存到: {output_image}")


if __name__ == "__main__":
    main()

