# 数据集规模统计

## 数据来源

从训练日志 (`output.log`) 和配置文件 (`args.json`) 中提取的信息。

---

## 1. 数据集配置

### 配置文件
- **训练集配置**: `../configs/libero_goal/his_2_third_view_wrist_w_state_5_256_pretokenize.yaml`
- **验证集（in-distribution）配置**: 同上
- **验证集（out-of-distribution）配置**: 同上

**注意**: 从日志看，训练集、验证集（in-distribution）、验证集（out-of-distribution）都使用**相同的配置文件**，因此它们的数据规模是**相同的**。

### 数据文件
- **数据路径**: `../processed_data/concate_tokens/libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.json`

---

## 2. 数据集规模

### 2.1 样本数量

从训练日志中可以看到：

```
[rank0:INFO|dataset.py:90] ... libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.json, typedefault: len 25920
```

**每个数据集的样本数**：
- **训练集 (train)**: **25,920** 个样本
- **验证集 in-distribution (val_ind)**: **25,920** 个样本  
- **验证集 out-of-distribution (val_ood)**: **25,920** 个样本

**总样本数**: 25,920 × 3 = **77,760** 个样本

---

## 3. 训练配置

从 `args.json` 中：

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 2 | 每个GPU的batch大小 |
| `accum_iter` | 4 | 梯度累积迭代次数 |
| `world_size` | 5 | GPU数量（数据并行） |
| `drop_last` | True | 丢弃最后一个不完整的batch |

### 3.1 有效Batch Size

在FSDP训练中，有效batch size有两种理解方式：

**方式1：每个GPU的有效batch size**
```
每个GPU的有效batch size = batch_size × accum_iter
                         = 2 × 4
                         = 8
```
每个GPU在每个优化步骤处理的样本数：**8 个样本**

**方式2：全局有效batch size（所有GPU）**
```
全局有效batch size = batch_size × accum_iter × world_size
                    = 2 × 4 × 5
                    = 40
```
所有GPU在每个优化步骤处理的总样本数：**40 个样本**

**说明**：
- 在FSDP中，每个GPU独立处理数据分片，但梯度会在所有GPU之间同步
- **每个GPU的有效batch size = 8**（每个GPU在每个优化步骤处理的样本数）
- **全局有效batch size = 40**（所有GPU加起来，用于学习率调度的参考值）
- 代码中打印的"effective batch size: 40"是全局值
- 从实际训练角度看，每个GPU的有效batch size是**8**，这是更常用的说法

---

## 4. 每个Epoch的Batch数量

从 `output.log` 中提取的信息：

每个epoch的batch数量：**2,592** 个batch

### 4.1 计算验证

**每个GPU的样本数**：
```
每个GPU的样本数 = 25920 ÷ 5 = 5,184 个样本
```

**每个GPU的batch数**：
```
每个GPU的batch数 = 5184 ÷ 2 = 2,592 个batch
```

**全局batch数（概念上的）**：
```
全局batch数 = 2592 × 5 = 12,960 个batch（如果每个GPU同步处理）
```

但注意：由于使用分布式训练，每个GPU独立处理自己的数据分片，所以：
- **每个GPU在训练时看到的batch数**：**2,592**
- 这是每个GPU在每个epoch中处理的batch数
- 每个GPU独立处理，所以没有"全局batch"的概念

---

## 5. 每个Epoch处理的样本数

### 5.1 每个GPU

```
每个GPU每epoch处理的样本数 = batch_size × batches_per_gpu
                             = 2 × 2592
                             = 5,184 个样本
```

### 5.2 所有GPU（全局）

```
全局每epoch处理的样本数 = 每个GPU的样本数 × GPU数量
                        = 5184 × 5
                        = 25,920 个样本
```

或者：
```
全局每epoch处理的样本数 = batch_size × batches_per_gpu × world_size
                        = 2 × 2592 × 5
                        = 25,920 个样本
```

**注意**：虽然全局处理的样本数是25,920，但在FSDP中：
- 每个GPU只看到自己的数据分片（5,184个样本）
- 每个GPU的有效batch size是 2 × 4 = 8
- 梯度在所有GPU之间同步，但数据是分片的

**每个epoch处理整个数据集一次**（因为数据集大小为25,920，正好等于全局处理的样本数）。

---

## 6. 训练迭代统计

### 6.1 每个Epoch

- **Batch数量**（每个GPU）: 2,592
- **样本数量**（每个GPU）: 5,184
- **优化步骤数**: 2,592 ÷ 4 = **648** 步（考虑梯度累积）

### 6.2 总训练迭代

假设训练 **40 个epoch**（从 `args.json` 中 `epochs: 40`）：

- **总batch数**（每个GPU）: 2,592 × 40 = **103,680**
- **总样本数**（全局）: 25,920 × 40 = **1,036,800**
- **总优化步骤数**: 648 × 40 = **25,920** 步

---

## 7. 数据集规模总结

### 7.1 样本数量

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| **训练集 (train)** | **25,920** | 用于训练模型 |
| **验证集 in-distribution (val_ind)** | **25,920** | 用于验证（分布内） |
| **验证集 out-of-distribution (val_ood)** | **25,920** | 用于验证（分布外） |
| **总计** | **77,760** | 所有数据集的样本总数 |

### 7.2 训练配置

| 配置项 | 值 |
|--------|-----|
| **Batch size** (每个GPU) | 2 |
| **梯度累积** | 4 |
| **GPU数量** | 5 |
| **有效batch size** (每个GPU) | 8 |
| **全局有效batch size** (所有GPU) | 40 |
| **每个epoch的batch数** (每个GPU) | 2,592 |
| **每个epoch的样本数** (全局) | 25,920 |

### 7.3 训练规模（40 epochs）

| 统计项 | 值 |
|--------|-----|
| **总batch数** (每个GPU) | 103,680 |
| **总样本数** (全局) | 1,036,800 |
| **总优化步骤数** | 25,920 |

---

## 8. 注意事项

### 8.1 数据分割

从日志信息看，训练集、验证集（in-distribution）、验证集（out-of-distribution）都使用**相同的配置文件和数据文件**。

这可能意味着：
1. **所有数据集使用相同的25,920个样本**
2. 或者配置文件中的路径指向的是已经预处理好的数据，实际的数据分割在预处理阶段完成

**建议**：检查数据预处理脚本，确认实际的数据分割情况。

### 8.2 数据来源

数据文件路径：
```
../processed_data/concate_tokens/libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.json
```

这个文件包含了预处理的token数据，样本数为 **25,920**。

---

## 9. 参考信息

### 9.1 数据集配置

配置文件位置：
```
../configs/libero_goal/his_2_third_view_wrist_w_state_5_256_pretokenize.yaml
```

配置内容：
```yaml
META:
  - path: '../processed_data/concate_tokens/libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.json'
```

### 9.2 日志示例

从 `output.log` 中提取的关键信息：
```
[rank0:INFO|dataset.py:90] ... libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.json, typedefault: len 25920
```

这个日志在训练集、验证集（ind）、验证集（ood）加载时都出现了，说明它们的数据规模相同。

---

## 总结

- **训练集样本数**: 25,920
- **验证集（ind）样本数**: 25,920
- **验证集（ood）样本数**: 25,920
- **每个epoch处理的样本数**: 25,920（全局）
- **每个epoch的batch数**: 2,592（每个GPU）
- **有效batch size**: 40

这些数据规模信息可以帮助你：
1. 评估训练数据是否充足
2. 计算训练时间
3. 规划验证和测试策略
4. 分析过拟合风险

