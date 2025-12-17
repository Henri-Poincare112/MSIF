"""
NHANES年龄预测神经网络分析
使用PyTorch框架实现10个不同配置的神经网络模型
包含验证集分割，模型保存，和预测结果保存功能
数据集：NHANES 2013-2014年龄预测数据集
"""

# ==================== NHANES 年龄预测数据集 - ANN 部分 ====================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 读取 NHANES 数据
nhanes_df = pd.read_csv('NHANES_age_prediction.csv')

# 特征列和目标列
nhanes_feature_columns = ['RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']
X_nhanes = nhanes_df[nhanes_feature_columns].copy()
y_nhanes = nhanes_df['RIDAGEYR'].values.astype(np.float32)

# 2. 特征标准化
nhanes_scaler = StandardScaler()
X_nhanes_scaled = nhanes_scaler.fit_transform(X_nhanes)

# 3. 划分训练 / 验证 / 测试集：80% / 10% / 10%
# 第一步：先拿出 10% 做测试集
X_temp_nhanes, X_test_nhanes, y_temp_nhanes, y_test_nhanes = train_test_split(
    X_nhanes_scaled, y_nhanes,
    test_size=0.1,
    random_state=42
)

# 第二步：在剩下 90% 中划出 1/9 做验证集（≈总样本的 10%）
X_train_nhanes, X_val_nhanes, y_train_nhanes, y_val_nhanes = train_test_split(
    X_temp_nhanes, y_temp_nhanes,
    test_size=0.1111,
    random_state=42
)

print("NHANES 数据集划分：")
print(f"训练集：{X_train_nhanes.shape[0]} 条")
print(f"验证集：{X_val_nhanes.shape[0]} 条")
print(f"测试集：{X_test_nhanes.shape[0]} 条")

# 4. 转成 PyTorch Tensor + DataLoader
X_train_tensor = torch.tensor(X_train_nhanes, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_nhanes, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_nhanes, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_nhanes, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_nhanes, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_nhanes, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = X_train_nhanes.shape[1]

# 5. 定义网络结构
class AgePredictionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(AgePredictionNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # 回归输出 1 个值
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_regression(model, X_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor).item()
        preds = outputs.cpu().numpy().flatten()
    y_true = y_tensor.cpu().numpy().flatten()
    rmse_val = rmse(y_true, preds)
    mae_val = mean_absolute_error(y_true, preds)
    r2_val = r2_score(y_true, preds)
    return y_true, preds, loss, rmse_val, mae_val, r2_val

# 6. 10 组超参数配置（保持和原来一致）
nhanes_ann_configs = [
    {"name": "NHANES_NN_1_Simple",       "hidden_dims": [32],                "lr": 0.001,  "dropout": 0.1},
    {"name": "NHANES_NN_2_Deep",         "hidden_dims": [64, 32],            "lr": 0.001,  "dropout": 0.2},
    {"name": "NHANES_NN_3_Wide",         "hidden_dims": [128],               "lr": 0.01,   "dropout": 0.3},
    {"name": "NHANES_NN_4_DeepWide",     "hidden_dims": [128, 64, 32],       "lr": 0.0001, "dropout": 0.2},
    {"name": "NHANES_NN_5_SmallLR",      "hidden_dims": [64, 32],            "lr": 0.0001, "dropout": 0.1},
    {"name": "NHANES_NN_6_LargeLR",      "hidden_dims": [64, 32],            "lr": 0.01,   "dropout": 0.1},
    {"name": "NHANES_NN_7_HighDropout",  "hidden_dims": [64, 32],            "lr": 0.001,  "dropout": 0.5},
    {"name": "NHANES_NN_8_VeryDeep",     "hidden_dims": [128, 64, 32, 16],   "lr": 0.001,  "dropout": 0.2},
    {"name": "NHANES_NN_9_Narrow",       "hidden_dims": [16, 8],             "lr": 0.001,  "dropout": 0.2},
    {"name": "NHANES_NN_10_VeryWide",    "hidden_dims": [256, 128],          "lr": 0.001,  "dropout": 0.3},
]

os.makedirs("nhanes_ann_models", exist_ok=True)
os.makedirs("nhanes_ann_predictions", exist_ok=True)

criterion_reg = nn.MSELoss()
num_epochs = 200  # 保持和原来一致

nhanes_ann_results = []

for idx, cfg in enumerate(nhanes_ann_configs):
    print(f"\n========== 训练 NHANES ANN 模型 {idx + 1} / {len(nhanes_ann_configs)} ==========")
    print(f"配置：{cfg}")

    model = AgePredictionNet(
        input_dim=input_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout_rate=cfg["dropout"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_rmse = float("inf")
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion_reg(outputs, yb)
            loss.backward()
            optimizer.step()

            batch_size_actual = yb.size(0)
            epoch_loss += loss.item() * batch_size_actual
            n_samples += batch_size_actual

        train_loss_epoch = epoch_loss / n_samples

        # 每个 epoch 在验证集上评估一次，用验证 RMSE 挑“最佳模型”
        _, _, val_loss_epoch, val_rmse_epoch, _, _ = evaluate_regression(
            model, X_val_tensor, y_val_tensor, criterion_reg
        )

        if val_rmse_epoch < best_val_rmse:
            best_val_rmse = val_rmse_epoch
            best_state_dict = model.state_dict()

        if (epoch + 1) % 50 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss_epoch:.4f} "
                f"Val Loss: {val_loss_epoch:.4f} "
                f"Val RMSE: {val_rmse_epoch:.4f}"
            )

    # 用验证集上表现最好的参数
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 最终在 train / val / test 上评估并保存预测
    y_train_true, y_train_pred, train_loss, train_rmse, train_mae, train_r2 = evaluate_regression(
        model, X_train_tensor, y_train_tensor, criterion_reg
    )
    y_val_true, y_val_pred, val_loss, val_rmse, val_mae, val_r2 = evaluate_regression(
        model, X_val_tensor, y_val_tensor, criterion_reg
    )
    y_test_true, y_test_pred, test_loss, test_rmse, test_mae, test_r2 = evaluate_regression(
        model, X_test_tensor, y_test_tensor, criterion_reg
    )

    base_name = f"nhanes_ann_model_{idx + 1}"

    # 保存预测 CSV：训练 / 验证 / 测试
    train_df = pd.DataFrame({'y_true': y_train_true, 'y_pred': y_train_pred})
    train_df.to_csv(
        os.path.join("nhanes_ann_predictions", f"{base_name}_train_predictions.csv"),
        index=False
    )

    val_df = pd.DataFrame({'y_true': y_val_true, 'y_pred': y_val_pred})
    val_df.to_csv(
        os.path.join("nhanes_ann_predictions", f"{base_name}_val_predictions.csv"),
        index=False
    )

    test_df = pd.DataFrame({'y_true': y_test_true, 'y_pred': y_test_pred})
    test_df.to_csv(
        os.path.join("nhanes_ann_predictions", f"{base_name}_test_predictions.csv"),
        index=False
    )

    # 保存模型
    model_path = os.path.join("nhanes_ann_models", f"{base_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': cfg["hidden_dims"],
        'dropout': cfg["dropout"],
        'lr': cfg["lr"],
        'feature_columns': nhanes_feature_columns
    }, model_path)

    nhanes_ann_results.append({
        'model_index': idx + 1,
        'name': cfg["name"],
        'hidden_dims': cfg["hidden_dims"],
        'dropout': cfg["dropout"],
        'lr': cfg["lr"],
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'model_path': model_path
    })

# 7. 汇总结果 CSV（按验证 RMSE 排序）
nhanes_ann_results_df = pd.DataFrame(nhanes_ann_results)
nhanes_ann_results_df = nhanes_ann_results_df.sort_values(by='val_rmse', ascending=True)
nhanes_ann_results_df.to_csv("nhanes_ann_results_with_val.csv", index=False)

print("\nNHANES ANN 模型结果（按验证集 RMSE 排序）：")
print(nhanes_ann_results_df)


