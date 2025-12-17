"""
DS 证据理论融合器（两个实验分别处理）

1）学生表现数据集：
    - 输入：student_svm_predictions/ 下 10 个 SVM 模型在 train/val/test 上的预测
            student_svm_model_{i}_{split}_predictions.csv
    - 特征：10 个模型的预测标签，通过训练集混淆矩阵估计每个模型在不同预测下的“可信度”
    - 融合：对 10 个模型对应的质量分配(mass)向量使用 Dempster 规则组合（在只有单点假设时
            等价于归一化后的逐元素乘积）
    - 输出目录：student_ds_fusion/

2）NHANES 年龄数据集：
    - 输入：nhanes_ann_predictions/ 下 10 个 ANN 模型在 train/val/test 上的预测
            nhanes_ann_model_{i}_{split}_predictions.csv
    - 特征：10 个模型的预测年龄（连续），在训练集上先把真实年龄与预测年龄四舍五入到整数，
            用“预测年龄 -> 真实年龄”混淆矩阵估计质量分配。
    - 融合：同样使用 Dempster 规则组合 10 个模型的质量分配。
    - 输出目录：nhanes_ds_fusion/

在两个实验中，DS 证据理论都作为“融合器”，只对多个模型的输出进行处理，
不直接使用原始特征。
"""

import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SVM_DIR = os.path.join(BASE_DIR, "student_svm_predictions")
ANN_DIR = os.path.join(BASE_DIR, "nhanes_ann_predictions")

STUDENT_FUSION_DIR = os.path.join(BASE_DIR, "student_ds_fusion")
NHANES_FUSION_DIR = os.path.join(BASE_DIR, "nhanes_ds_fusion")

N_MODELS = 10  # 每个实验 10 个基模型
EPS = 1e-12    # 数值稳定用的极小值
ALPHA = 1.0    # Laplace 平滑系数，避免概率为 0


# ======================= 通用工具函数 =======================

def _check_dir(path: str, name: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"找不到目录 {path}，请确认已经运行过 {name} 并生成预测结果。"
        )


def _ds_combine_masses(mass_list):
    """
    对若干个只支持单点假设的质量分配向量进行 Dempster 组合。
    mass_list: [m1, m2, ...]，每个 mi 是 shape=(K,) 的 numpy 向量，元素非负且和为 1。

    在只有单点假设时，Dempster 组合退化为归一化后的逐元素乘积：
        m_comb(k) ∝ ∏_i m_i(k)
    这里用 log-domain 相加避免下溢。
    """
    log_mass = None
    for m in mass_list:
        m = np.clip(np.asarray(m, dtype=float), EPS, 1.0)
        if log_mass is None:
            log_mass = np.log(m)
        else:
            log_mass += np.log(m)
    # 归一化
    log_mass -= np.max(log_mass)
    mass = np.exp(log_mass)
    s = mass.sum()
    if s <= 0:
        # 极端数值问题，退化为均匀分布
        mass = np.ones_like(mass) / len(mass)
    else:
        mass /= s
    return mass


# ======================= 学生数据集：融合 10 个 SVM =======================

def _build_student_bpa(alpha: float = ALPHA):
    """
    基于学生数据集 train 划分上 10 个 SVM 模型的预测，构建每个模型的
    “预测标签 -> 真实标签分布” 质量分配矩阵。

    返回：
        classes: 列表，所有类别编码（例如 [0,1,2]）
        code_to_idx: dict，类别编码 -> 索引
        bpa_matrices: 长度为 N_MODELS 的列表
                      其中每个元素是 shape=(K, K) 的 numpy 数组：
                      bpa_matrices[m][j_idx, k_idx] = P(true_class = k | pred_class = j, model m)
    """
    _check_dir(SVM_DIR, "svm.py (学生表现 SVM)")

    # 用第一个模型的 train 文件确定类别集合
    first_train_path = os.path.join(SVM_DIR, "student_svm_model_1_train_predictions.csv")
    if not os.path.exists(first_train_path):
        raise FileNotFoundError(f"找不到 {first_train_path}，请先运行 svm.py 生成训练预测。")

    df_first = pd.read_csv(first_train_path)
    classes = sorted(np.unique(df_first["y_true_encoded"].values))
    K = len(classes)
    code_to_idx = {c: i for i, c in enumerate(classes)}

    bpa_matrices = []

    for m in range(1, N_MODELS + 1):
        train_path = os.path.join(SVM_DIR, f"student_svm_model_{m}_train_predictions.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到 SVM 训练预测文件：{train_path}")

        df = pd.read_csv(train_path)
        y_true_enc = df["y_true_encoded"].values
        y_pred_enc = df["y_pred_encoded"].values

        # 全局真实标签先验分布（用于某个预测标签在训练集中从未出现时）
        counts_global = np.zeros(K, dtype=float)
        for c in y_true_enc:
            counts_global[code_to_idx[c]] += 1
        prior = (counts_global + alpha) / (counts_global.sum() + alpha * K)

        # 为每个预测标签构建条件分布
        mat = np.zeros((K, K), dtype=float)  # 行：预测标签，列：真实标签
        for pred_code in classes:
            j_idx = code_to_idx[pred_code]
            mask = (y_pred_enc == pred_code)
            if not np.any(mask):
                probs = prior.copy()
            else:
                counts = np.zeros(K, dtype=float)
                for t in y_true_enc[mask]:
                    counts[code_to_idx[t]] += 1
                probs = (counts + alpha) / (counts.sum() + alpha * K)
            mat[j_idx, :] = probs

        bpa_matrices.append(mat)

    return classes, code_to_idx, bpa_matrices


def _run_student_ds_on_split(split: str, classes, code_to_idx, bpa_matrices):
    """
    在指定划分（train / val / test）上执行 DS 融合，返回：
        y_true_enc, y_true, y_pred_enc_ds
    """
    dfs = []
    for m in range(1, N_MODELS + 1):
        path = os.path.join(SVM_DIR, f"student_svm_model_{m}_{split}_predictions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 SVM 预测文件：{path}")
        dfs.append(pd.read_csv(path))

    n_samples = len(dfs[0])
    for i, df in enumerate(dfs[1:], start=2):
        if len(df) != n_samples:
            raise ValueError(f"SVM 模型 {i} 在 {split} 划分上的样本数不一致。")

    K = len(classes)
    y_true_enc = dfs[0]["y_true_encoded"].values
    y_true = dfs[0]["y_true"].values

    y_pred_enc_ds = np.zeros(n_samples, dtype=int)

    for idx in range(n_samples):
        mass_list = []
        for m_idx, df in enumerate(dfs):
            pred_code = int(df.loc[idx, "y_pred_encoded"])
            if pred_code not in code_to_idx:
                # 未知编码，退化为均匀分布
                probs = np.ones(K, dtype=float) / K
            else:
                j_idx = code_to_idx[pred_code]
                probs = bpa_matrices[m_idx][j_idx, :]
            mass_list.append(probs)

        m_comb = _ds_combine_masses(mass_list)
        k_idx = int(np.argmax(m_comb))
        y_pred_enc_ds[idx] = classes[k_idx]

    return y_true_enc, y_true, y_pred_enc_ds


def _save_student_ds_predictions(split, output_dir, y_true_enc, y_true, y_pred_enc):
    classes = sorted(np.unique(y_true_enc))
    # 构建编码 -> 字符串标签的映射
    label_map = {}
    for enc, lab in zip(y_true_enc, y_true):
        if enc not in label_map:
            label_map[enc] = lab

    y_pred = [label_map.get(code, "Unknown") for code in y_pred_enc]

    df = pd.DataFrame({
        "y_true_encoded": y_true_enc,
        "y_pred_encoded": y_pred_enc,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    out_path = os.path.join(output_dir, f"student_ds_{split}_predictions.csv")
    df.to_csv(out_path, index=False)
    return out_path


def run_student_ds_fusion():
    """
    学生表现实验：
    使用 train 划分上 10 个 SVM 模型的预测构建 BPA，
    再在 train / val / test 上用 DS 证据理论进行融合。
    """
    os.makedirs(STUDENT_FUSION_DIR, exist_ok=True)

    print("========== 学生数据集：DS 证据理论融合 10 个 SVM ==========")

    classes, code_to_idx, bpa_matrices = _build_student_bpa()
    K = len(classes)
    print(f"  识别到 {K} 个类别：{classes}")

    results = []

    for split in ["train", "val", "test"]:
        print(f"  在 {split} 划分上进行融合 …")
        y_true_enc, y_true, y_pred_enc = _run_student_ds_on_split(
            split, classes, code_to_idx, bpa_matrices
        )
        acc = float((y_true_enc == y_pred_enc).mean())
        results.append({"split": split, "accuracy": acc})
        print(f"    {split.capitalize()} accuracy: {acc:.4f}")
        _save_student_ds_predictions(split, STUDENT_FUSION_DIR, y_true_enc, y_true, y_pred_enc)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(STUDENT_FUSION_DIR, "student_ds_results.csv")
    results_df.to_csv(results_path, index=False)

    model_path = os.path.join(STUDENT_FUSION_DIR, "student_ds_bpa.joblib")
    joblib.dump({
        "classes": classes,
        "code_to_idx": code_to_idx,
        "bpa_matrices": bpa_matrices,
        "description": "DS evidence fusion over 10 SVM models (student dataset)."
    }, model_path)

    print("学生数据集 DS 融合完成。输出：")
    print(f"  结果汇总：{results_path}")
    print(f"  BPA 模型：{model_path}")
    print("===============================================================")


# ======================= NHANES 年龄数据集：融合 10 个 ANN =======================

def _build_nhanes_bpa(alpha: float = ALPHA):
    """
    基于 NHANES train 划分上 10 个 ANN 模型的预测，构建每个模型的
    “预测年龄(整数) -> 真实年龄(整数) 分布” 质量分配矩阵。

    返回：
        classes: 所有可能的整数年龄列表，例如 [12, …, 80]
        age_to_idx: dict，年龄 -> 索引
        bpa_matrices: 长度为 N_MODELS 的列表，
                      每个元素是 shape=(A, A) 的数组：
                      bpa_matrices[m][j_idx, k_idx] = P(true_age = k | pred_age = j, model m)
    """
    _check_dir(ANN_DIR, "nn.py (NHANES ANN)")

    first_train_path = os.path.join(ANN_DIR, "nhanes_ann_model_1_train_predictions.csv")
    if not os.path.exists(first_train_path):
        raise FileNotFoundError(f"找不到 {first_train_path}，请先运行 nn.py 生成训练预测。")

    df_first = pd.read_csv(first_train_path)
    y_true_first = np.rint(df_first["y_true"].values).astype(int)
    min_age, max_age = int(y_true_first.min()), int(y_true_first.max())
    classes = list(range(min_age, max_age + 1))
    A = len(classes)
    age_to_idx = {age: i for i, age in enumerate(classes)}

    bpa_matrices = []

    for m in range(1, N_MODELS + 1):
        train_path = os.path.join(ANN_DIR, f"nhanes_ann_model_{m}_train_predictions.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到 ANN 训练预测文件：{train_path}")

        df = pd.read_csv(train_path)
        y_true = np.rint(df["y_true"].values).astype(int)
        y_pred = np.rint(df["y_pred"].values).astype(int)

        # 将预测年龄裁剪到 [min_age, max_age] 范围内
        y_pred = np.clip(y_pred, min_age, max_age)

        # 全局真实年龄先验分布
        counts_global = np.zeros(A, dtype=float)
        for age in y_true:
            counts_global[age_to_idx[age]] += 1
        prior = (counts_global + alpha) / (counts_global.sum() + alpha * A)

        mat = np.zeros((A, A), dtype=float)  # 行：预测年龄，列：真实年龄

        for pred_age in classes:
            j_idx = age_to_idx[pred_age]
            mask = (y_pred == pred_age)
            if not np.any(mask):
                probs = prior.copy()
            else:
                counts = np.zeros(A, dtype=float)
                for t in y_true[mask]:
                    counts[age_to_idx[t]] += 1
                probs = (counts + alpha) / (counts.sum() + alpha * A)
            mat[j_idx, :] = probs

        bpa_matrices.append(mat)

    return classes, age_to_idx, bpa_matrices


def _run_nhanes_ds_on_split(split: str, classes, age_to_idx, bpa_matrices):
    """
    在 NHANES 指定划分（train / val / test）上执行 DS 融合。
    返回：
        y_true: 连续真实年龄
        y_true_int: 取整后的真实年龄（类别）
        y_pred_int_ds: DS 融合后预测的整数年龄
    """
    dfs = []
    for m in range(1, N_MODELS + 1):
        path = os.path.join(ANN_DIR, f"nhanes_ann_model_{m}_{split}_predictions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 ANN 预测文件：{path}")
        dfs.append(pd.read_csv(path))

    n_samples = len(dfs[0])
    for i, df in enumerate(dfs[1:], start=2):
        if len(df) != n_samples:
            raise ValueError(f"ANN 模型 {i} 在 {split} 划分上的样本数不一致。")

    y_true = dfs[0]["y_true"].values.astype(float)
    y_true_int = np.rint(y_true).astype(int)

    min_age, max_age = min(classes), max(classes)
    A = len(classes)

    y_pred_int_ds = np.zeros(n_samples, dtype=int)

    for idx in range(n_samples):
        mass_list = []
        for m_idx, df in enumerate(dfs):
            y_pred_cont = float(df.loc[idx, "y_pred"])
            pred_int = int(np.rint(y_pred_cont))
            pred_int = max(min(pred_int, max_age), min_age)  # 裁剪到合法范围
            j_idx = age_to_idx[pred_int]
            probs = bpa_matrices[m_idx][j_idx, :]
            mass_list.append(probs)

        m_comb = _ds_combine_masses(mass_list)
        k_idx = int(np.argmax(m_comb))
        y_pred_int_ds[idx] = classes[k_idx]

    return y_true, y_true_int, y_pred_int_ds


def _save_nhanes_ds_predictions(split, output_dir, y_true, y_true_int, y_pred_int):
    abs_error = np.abs(y_true - y_pred_int)
    df = pd.DataFrame({
        "y_true": y_true,
        "y_true_int": y_true_int,
        "y_pred_int": y_pred_int,
        "abs_error": abs_error,
    })
    out_path = os.path.join(output_dir, f"nhanes_ds_{split}_predictions.csv")
    df.to_csv(out_path, index=False)
    return out_path


def run_nhanes_ds_fusion():
    """
    NHANES 年龄实验：
    使用 train 划分上 10 个 ANN 模型的预测构建 BPA，
    再在 train / val / test 上用 DS 证据理论进行融合。
    """
    os.makedirs(NHANES_FUSION_DIR, exist_ok=True)

    print("========== NHANES 年龄数据集：DS 证据理论融合 10 个 ANN ==========")

    classes, age_to_idx, bpa_matrices = _build_nhanes_bpa()
    print(f"  年龄类别范围：{min(classes)} ~ {max(classes)}，共 {len(classes)} 个整数年龄。")

    results = []

    for split in ["train", "val", "test"]:
        print(f"  在 {split} 划分上进行融合 …")
        y_true, y_true_int, y_pred_int = _run_nhanes_ds_on_split(
            split, classes, age_to_idx, bpa_matrices
        )

        # 分类准确率（按整数年龄完全相等）
        acc = float((y_true_int == y_pred_int).mean())
        # 回归评价指标（把预测整数当作年龄）
        mae = float(np.mean(np.abs(y_true - y_pred_int)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred_int) ** 2)))

        results.append({
            "split": split,
            "class_accuracy": acc,
            "mae": mae,
            "rmse": rmse,
        })

        print(
            f"    {split.capitalize()} - 类别准确率: {acc:.4f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}"
        )

        _save_nhanes_ds_predictions(split, NHANES_FUSION_DIR, y_true, y_true_int, y_pred_int)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(NHANES_FUSION_DIR, "nhanes_ds_results.csv")
    results_df.to_csv(results_path, index=False)

    model_path = os.path.join(NHANES_FUSION_DIR, "nhanes_ds_bpa.joblib")
    joblib.dump({
        "classes": classes,
        "age_to_idx": age_to_idx,
        "bpa_matrices": bpa_matrices,
        "description": "DS evidence fusion over 10 ANN models (NHANES age dataset; ages treated as discrete)."
    }, model_path)

    print("NHANES 年龄数据集 DS 融合完成。输出：")
    print(f"  结果汇总：{results_path}")
    print(f"  BPA 模型：{model_path}")
    print("===============================================================")


# ======================= 主入口 =======================

if __name__ == "__main__":
    # 学生数据集 DS 融合
    try:
        run_student_ds_fusion()
    except Exception as e:
        print(f"学生数据集 DS 融合时出错：{e}")

    # NHANES 年龄数据集 DS 融合
    try:
        run_nhanes_ds_fusion()
    except Exception as e:
        print(f"NHANES 数据集 DS 融合时出错：{e}")
