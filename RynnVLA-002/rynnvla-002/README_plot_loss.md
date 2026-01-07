# Loss曲线绘制脚本使用说明

## 功能

`plot_loss_curves.py` 脚本用于从训练日志文件中提取并绘制loss曲线。

## 使用方法

### 基本用法

```bash
python plot_loss_curves.py --log_dir <日志目录路径>
```

### 示例

```bash
# 使用当前实验的输出目录
python plot_loss_curves.py --log_dir outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw

# 指定输出图片路径
python plot_loss_curves.py --log_dir outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw --output my_loss_curves.png

# 不显示图表，仅保存文件（适用于服务器环境）
python plot_loss_curves.py --log_dir outputs/libero_goal/his_2_third_view_wrist_w_state_5_256_abiw --no-show
```

## 参数说明

- `--log_dir`: (必需) 日志文件目录路径，应包含 `log_train.txt` 文件
- `--output`: (可选) 输出图片路径，默认为 `log_dir/loss_curves.png`
- `--no-show`: (可选) 不显示图表，仅保存文件（适用于无GUI环境）

## 输出的图表

脚本会生成一个包含4个子图的图表：

1. **主要Loss组件** (左上)
   - Train Closs (分类loss)
   - Train Loss CT (分类token loss)
   - Train Z Loss
   - 验证Loss（如果有验证数据）

2. **Action L1 Loss** (右上)
   - 所有action时间步的平均L1 loss
   - 训练和验证对比

3. **学习率和梯度范数** (左下)
   - 学习率曲线（绿色，右侧Y轴）
   - 梯度范数曲线（红色，左侧Y轴）

4. **Accuracy指标** (右下)
   - 训练准确率指标（如果数据有效）

## 数据格式要求

脚本期望日志文件格式为JSON Lines格式，每行一个JSON对象，例如：

```json
{"train_closs": 3.2588, "train_loss_ct": 0.085, "train_z_loss": 39.86, "epoch": 0}
{"train_closs": 4.0639, "train_loss_ct": 0.612, "train_z_loss": 73.26, "epoch": 1}
```

## 依赖库

```bash
pip install matplotlib numpy
```

## 注意事项

1. 脚本会自动过滤异常值（如accuracy > 1000的值）
2. 如果验证日志文件（`log_eval_ind.txt` 或 `log_eval_ood.txt`）存在，会自动加载并绘制验证曲线
3. 图片保存为PNG格式，分辨率为300 DPI

