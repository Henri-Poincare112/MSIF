
"""
学生表现预测支持向量机(SVM)分析
使用scikit-learn框架实现10个不同配置的SVM模型
数据集：学生表现预测数据集（分类任务）
"""

# ==================== 学生表现数据集（Student Performance） - SVM 部分 ====================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. 读取数据
data = pd.read_csv('data.csv', sep=';')

y = data['Target'].values
X = data.select_dtypes(include=[np.number]).copy()
if 'Target' in X.columns:
    X = X.drop(columns=['Target'])

# 2. 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练 / 验证 / 测试（80% / 10% / 10%）
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.1,
    random_state=42,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.1111,
    random_state=42,
    stratify=y_temp
)

print("学生数据集（SVM）划分：")
print(f"训练集：{X_train.shape[0]} 条")
print(f"验证集：{X_val.shape[0]} 条")
print(f"测试集：{X_test.shape[0]} 条")

# 5. 定义 10 组 SVM 参数
student_svm_configs = [
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'linear', 'C': 10.0},
    {'kernel': 'rbf',   'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'rbf',   'C': 10.0, 'gamma': 'scale'},
    {'kernel': 'rbf',   'C': 1.0, 'gamma': 0.01},
    {'kernel': 'poly',  'C': 1.0, 'degree': 2, 'gamma': 'scale'},
    {'kernel': 'poly',  'C': 1.0, 'degree': 3, 'gamma': 'scale'},
    {'kernel': 'poly',  'C': 10.0, 'degree': 2, 'gamma': 'scale'},
    {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 'scale'},
]

os.makedirs("student_svm_models", exist_ok=True)
os.makedirs("student_svm_predictions", exist_ok=True)

svm_results = []

def save_split_predictions(base_name, split_name, y_true, y_pred, encoder):
    """
    将某个划分（train/val/test）的真实标签和预测标签保存到 CSV
    """
    y_true_labels = encoder.inverse_transform(y_true)
    y_pred_labels = encoder.inverse_transform(y_pred)
    df = pd.DataFrame({
        'y_true_encoded': y_true,
        'y_pred_encoded': y_pred,
        'y_true': y_true_labels,
        'y_pred': y_pred_labels
    })
    df.to_csv(
        os.path.join("student_svm_predictions", f"{base_name}_{split_name}_predictions.csv"),
        index=False
    )

for idx, cfg in enumerate(student_svm_configs):
    print(f"\n========== 训练 SVM 模型 {idx + 1} / {len(student_svm_configs)} ==========")
    print(f"配置：{cfg}")

    # 把参数拆分出来，构建 SVC
    params = cfg.copy()
    kernel = params.pop('kernel')
    model = SVC(kernel=kernel, **params)

    # 训练
    model.fit(X_train, y_train)

    # 各划分上的预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    base_name = f"student_svm_model_{idx + 1}"

    # 保存预测标签（训练 / 验证 / 测试）
    save_split_predictions(base_name, "train", y_train, y_train_pred, label_encoder)
    save_split_predictions(base_name, "val", y_val, y_val_pred, label_encoder)
    save_split_predictions(base_name, "test", y_test, y_test_pred, label_encoder)

    # 保存模型
    model_path = os.path.join("student_svm_models", f"{base_name}.joblib")
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'config': cfg
    }, model_path)

    svm_results.append({
        'model_index': idx + 1,
        'config': cfg,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'model_path': model_path
    })

# 汇总结果
svm_results_df = pd.DataFrame(svm_results)
svm_results_df = svm_results_df.sort_values(by='val_accuracy', ascending=False)
svm_results_df.to_csv("student_svm_results_with_val.csv", index=False)

print("\n学生数据集 SVM 模型结果（按验证集准确率排序）：")
print(svm_results_df)
