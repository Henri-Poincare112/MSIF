"""
朴素贝叶斯二层融合（两个实验分别处理）

1）学生表现数据集：
    - 输入：student_svm_predictions/ 下 10 个 SVM 模型在 train/val/test 上的预测
            student_svm_model_{i}_{split}_predictions.csv
    - 特征：10 个模型的 y_pred_encoded（离散标签编码）
    - 标签：y_true_encoded
    - 输出目录：student_naive_bayes_fusion/

2）NHANES 年龄数据集：
    - 输入：nhanes_ann_predictions/ 下 10 个 ANN 模型在 train/val/test 上的预测
            nhanes_ann_model_{i}_{split}_predictions.csv
    - 特征：10 个模型的 y_pred（连续预测的年龄）
    - 标签：将 y_true 四舍五入/取整为整数年龄，作为类别
    - 输出目录：nhanes_naive_bayes_fusion/

在两个实验中，朴素贝叶斯都扮演“融合器”的角色：
它不直接使用原始特征，而是只对 10 个模型的预测结果进行学习。
"""

import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SVM_DIR = os.path.join(BASE_DIR, "student_svm_predictions")
ANN_DIR = os.path.join(BASE_DIR, "nhanes_ann_predictions")

STUDENT_FUSION_DIR = os.path.join(BASE_DIR, "student_naive_bayes_fusion")
NHANES_FUSION_DIR = os.path.join(BASE_DIR, "nhanes_naive_bayes_fusion")

N_MODELS = 10  # 每个实验 10 个模型


# ====================== 工具函数 ======================

def _check_dir(path: str, name: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"找不到目录 {path}，请确认已经运行过 {name} 并生成预测结果。"
        )


# ====================== 学生数据集：融合 10 个 SVM ======================

def _load_student_split(split: str):
    """
    加载学生数据集某个划分上 10 个 SVM 模型的预测。
    返回：
        X_split: (n_samples, 10)  特征矩阵（每列是一个模型的 y_pred_encoded）
        y_true_encoded: (n_samples,)
        y_true: (n_samples,)       字符串标签（方便写结果表）
    """
    features = []
    y_true_encoded = None
    y_true = None

    for i in range(1, N_MODELS + 1):
        file_name = f"student_svm_model_{i}_{split}_predictions.csv"
        path = os.path.join(SVM_DIR, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 SVM 预测文件：{path}")

        df = pd.read_csv(path)

        required_cols = {"y_true_encoded", "y_pred_encoded", "y_true"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"文件 {path} 缺少必要列 {required_cols}，请检查生成 SVM 预测的代码。"
            )

        if y_true_encoded is None:
            y_true_encoded = df["y_true_encoded"].values
            y_true = df["y_true"].values
        else:
            if len(df) != len(y_true_encoded):
                raise ValueError(
                    f"文件 {path} 的样本数({len(df)}) 与其他 SVM 模型不一致({len(y_true_encoded)})。"
                )

        features.append(df["y_pred_encoded"].values)

    X_split = np.vstack(features).T
    return X_split, y_true_encoded, y_true


def _build_label_map(y_true_encoded, y_true):
    """根据训练集的 y_true_encoded 和 y_true 构建编码 -> 字符串标签映射。"""
    mapping = {}
    for enc, label in zip(y_true_encoded, y_true):
        if enc not in mapping:
            mapping[enc] = label
    return mapping


def _save_student_predictions(output_dir, split, y_true_enc, y_true, y_pred_enc, label_map):
    y_pred = [label_map.get(code, "Unknown") for code in y_pred_enc]
    df = pd.DataFrame({
        "y_true_encoded": y_true_enc,
        "y_pred_encoded": y_pred_enc,
        "y_true": y_true,
        "y_pred": y_pred
    })
    out_path = os.path.join(output_dir, f"student_nb_{split}_predictions.csv")
    df.to_csv(out_path, index=False)
    return out_path


def run_student_naive_bayes_fusion():
    """
    学生表现实验：
    使用 10 个 SVM 模型在 train/val/test 上的预测，训练朴素贝叶斯融合器。
    """
    _check_dir(SVM_DIR, "svm.py (学生表现 SVM)")
    os.makedirs(STUDENT_FUSION_DIR, exist_ok=True)

    print("========== 学生数据集：朴素贝叶斯融合 10 个 SVM ==========")

    # 1) 加载三个划分
    print("加载 train / val / test SVM 预测 …")
    X_train, y_train_enc, y_train = _load_student_split("train")
    X_val,   y_val_enc,   y_val   = _load_student_split("val")
    X_test,  y_test_enc,  y_test  = _load_student_split("test")

    print("特征维度：")
    print(f"  train: {X_train.shape}")
    print(f"  val:   {X_val.shape}")
    print(f"  test:  {X_test.shape}")

    # 2) 训练朴素贝叶斯
    nb = GaussianNB()
    nb.fit(X_train, y_train_enc)

    # 3) 在三个划分上评估
    results = []
    for split, X, y_enc in [
        ("train", X_train, y_train_enc),
        ("val",   X_val,   y_val_enc),
        ("test",  X_test,  y_test_enc),
    ]:
        y_pred_enc = nb.predict(X)
        acc = accuracy_score(y_enc, y_pred_enc)
        results.append({"split": split, "accuracy": acc})
        print(f"  {split.capitalize()} accuracy: {acc:.4f}")

    # 4) 保存预测结果和模型
    label_map = _build_label_map(y_train_enc, y_train)

    train_pred_path = _save_student_predictions(
        STUDENT_FUSION_DIR, "train",
        y_train_enc, y_train, nb.predict(X_train), label_map
    )
    val_pred_path = _save_student_predictions(
        STUDENT_FUSION_DIR, "val",
        y_val_enc, y_val, nb.predict(X_val), label_map
    )
    test_pred_path = _save_student_predictions(
        STUDENT_FUSION_DIR, "test",
        y_test_enc, y_test, nb.predict(X_test), label_map
    )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(STUDENT_FUSION_DIR, "student_naive_bayes_results.csv")
    results_df.to_csv(results_path, index=False)

    model_path = os.path.join(STUDENT_FUSION_DIR, "student_naive_bayes_model.joblib")
    joblib.dump({
        "naive_bayes_model": nb,
        "label_map": label_map,
        "description": "Naive Bayes fusion over 10 SVM models (student dataset)."
    }, model_path)

    print("学生数据集融合完成。输出：")
    print(f"  预测结果：{train_pred_path}, {val_pred_path}, {test_pred_path}")
    print(f"  汇总指标：{results_path}")
    print(f"  模型文件：{model_path}")
    print("==========================================================")


# ====================== NHANES 年龄数据集：融合 10 个 ANN ======================

def _load_nhanes_split(split: str):
    """
    加载 NHANES 某个划分上 10 个 ANN 模型的预测。
    返回：
        X_split: (n_samples, 10)  特征矩阵（每列是某个模型的 y_pred）
        y_true: (n_samples,)      真实年龄（浮点）
    """
    features = []
    y_true = None

    for i in range(1, N_MODELS + 1):
        file_name = f"nhanes_ann_model_{i}_{split}_predictions.csv"
        path = os.path.join(ANN_DIR, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 ANN 预测文件：{path}")

        df = pd.read_csv(path)
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            raise ValueError(
                f"文件 {path} 中缺少 y_true 或 y_pred 列，请检查生成 ANN 预测的代码。"
            )

        if y_true is None:
            y_true = df["y_true"].values
        else:
            if len(df) != len(y_true):
                raise ValueError(
                    f"文件 {path} 的样本数({len(df)}) 与其他 ANN 模型不一致({len(y_true)})。"
                )

        features.append(df["y_pred"].values)

    X_split = np.vstack(features).T
    return X_split, y_true


def _save_nhanes_predictions(output_dir, split, y_true, y_true_int, y_pred_int):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_true_int": y_true_int,
        "y_pred_int": y_pred_int,
        "abs_error": np.abs(y_true - y_pred_int)
    })
    out_path = os.path.join(output_dir, f"nhanes_nb_{split}_predictions.csv")
    df.to_csv(out_path, index=False)
    return out_path


def run_nhanes_naive_bayes_fusion():
    """
    NHANES 年龄实验：
    使用 10 个 ANN 模型在 train/val/test 上的预测，训练朴素贝叶斯融合器。

    注意：这里把年龄看成离散类别（整数岁数），这是为了满足“朴素贝叶斯作为融合器”的要求。
    """
    _check_dir(ANN_DIR, "nn.py (NHANES ANN)")
    os.makedirs(NHANES_FUSION_DIR, exist_ok=True)

    print("========== NHANES 年龄数据集：朴素贝叶斯融合 10 个 ANN ==========")

    print("加载 train / val / test ANN 预测 …")
    X_train, y_train = _load_nhanes_split("train")
    X_val,   y_val   = _load_nhanes_split("val")
    X_test,  y_test  = _load_nhanes_split("test")

    print("特征维度：")
    print(f"  train: {X_train.shape}")
    print(f"  val:   {X_val.shape}")
    print(f"  test:  {X_test.shape}")

    # 将真实年龄取整，作为朴素贝叶斯的类别标签
    y_train_int = np.rint(y_train).astype(int)
    y_val_int   = np.rint(y_val).astype(int)
    y_test_int  = np.rint(y_test).astype(int)

    # 训练朴素贝叶斯（以年龄整数作为类别）
    nb = GaussianNB()
    nb.fit(X_train, y_train_int)

    # 评估：既看“年龄类别准确率”，也看把预测类别当作年龄时的 MAE / RMSE
    results = []
    for split, X, y_int, y_true in [
        ("train", X_train, y_train_int, y_train),
        ("val",   X_val,   y_val_int,   y_val),
        ("test",  X_test,  y_test_int,  y_test),
    ]:
        y_pred_int = nb.predict(X)
        acc = accuracy_score(y_int, y_pred_int)
        mae = mean_absolute_error(y_true, y_pred_int)

        # 兼容旧版 sklearn：不能用 squared=False
        mse = mean_squared_error(y_true, y_pred_int)
        rmse = np.sqrt(mse)

        results.append({
            "split": split,
            "class_accuracy": acc,
            "mae": mae,
            "rmse": rmse
        })
        print(
            f"  {split.capitalize()} - 类别准确率: {acc:.4f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}"
        )

        _save_nhanes_predictions(
            NHANES_FUSION_DIR, split,
            y_true, y_int, y_pred_int
        )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(NHANES_FUSION_DIR, "nhanes_naive_bayes_results.csv")
    results_df.to_csv(results_path, index=False)

    model_path = os.path.join(NHANES_FUSION_DIR, "nhanes_naive_bayes_model.joblib")
    joblib.dump({
        "naive_bayes_model": nb,
        "description": "Naive Bayes fusion over 10 ANN models (NHANES age dataset; age treated as discrete)."
    }, model_path)

    print("NHANES 年龄数据集融合完成。主要输出：")
    print(f"  汇总指标：{results_path}")
    print(f"  模型文件：{model_path}")
    print("============================================================")


# ====================== 主入口 ======================

if __name__ == "__main__":
    # 先融合学生数据集（SVM）
    try:
        run_student_naive_bayes_fusion()
    except Exception as e:
        print(f"学生数据集融合时出错：{e}")

    # 再融合 NHANES 年龄数据集（ANN）
    try:
        run_nhanes_naive_bayes_fusion()
    except Exception as e:
        print(f"NHANES 数据集融合时出错：{e}")
