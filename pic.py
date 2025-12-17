
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']      # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False





"""实验一"""

"""图 1 不同 SVM 模型的验证集准确率"""
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # === 关键：设置支持中文的字体 ===
# plt.rcParams['font.sans-serif'] = ['SimHei']      # 或者 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
#
# svm_df = pd.read_csv('student_svm_results_with_val.csv')
# svm_df = svm_df.sort_values('val_accuracy', ascending=False)
#
# plt.figure(figsize=(8, 4))
# plt.bar(svm_df['model_index'].astype(str), svm_df['val_accuracy'])
# plt.xlabel('SVM 模型编号')
# plt.ylabel('验证集准确率')
# plt.title('不同 SVM 模型的验证集准确率')
# plt.tight_layout()
# plt.savefig('fig_svm_val_acc_bar.png', dpi=1200)
# plt.show()


"""图 2 Train/Val/Test准确率对比"""
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# === 关键：设置支持中文的字体 ===
# plt.rcParams['font.sans-serif'] = ['SimHei']      # 或者 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
#
# svm_df = pd.read_csv('student_svm_results_with_val.csv')
# svm_df = svm_df.sort_values('model_index')
#
# indices = np.arange(len(svm_df))
# width = 0.25
#
# plt.figure(figsize=(10, 4))
# plt.bar(indices - width, svm_df['train_accuracy'], width=width, label='Train')
# plt.bar(indices,         svm_df['val_accuracy'],   width=width, label='Val')
# plt.bar(indices + width, svm_df['test_accuracy'],  width=width, label='Test')
#
# plt.xticks(indices, svm_df['model_index'].astype(str))
# plt.xlabel('SVM 模型编号')
# plt.ylabel('准确率')
# plt.title('10 个 SVM 模型在不同数据划分上的准确率对比')
# plt.legend()
# plt.tight_layout()
# plt.savefig('fig_svm_split_acc_bar.png', dpi=300)
# plt.show()



"""图 3 不同 ANN 模型的验证集 RMSE"""
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # === 关键：设置支持中文的字体 ===
# plt.rcParams['font.sans-serif'] = ['SimHei']      # 或者 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
#
# ann_df = pd.read_csv('nhanes_ann_results_with_val.csv')
# ann_df = ann_df.sort_values('val_rmse', ascending=True)
#
# plt.figure(figsize=(8, 4))
# plt.bar(ann_df['model_index'].astype(str), ann_df['val_rmse'])
# plt.xlabel('ANN 模型编号')
# plt.ylabel('验证集 RMSE')
# plt.title('不同 ANN 模型的验证集 RMSE')
# plt.tight_layout()
# plt.savefig('fig_ann_val_rmse_bar.png', dpi=300)
# plt.show()





"""图 4 模型规模与验证 RMSE 的关系"""
# import pandas as pd
# import matplotlib.pyplot as plt
# import ast
#
# # === 关键：设置支持中文的字体 ===
# plt.rcParams['font.sans-serif'] = ['SimHei']      # 或者 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
#
# ann_df = pd.read_csv('nhanes_ann_results_with_val.csv')
#
# def parse_hidden(s):
#     if isinstance(s, list):
#         return s
#     return ast.literal_eval(s)
#
# ann_df['hidden_list'] = ann_df['hidden_dims'].apply(parse_hidden)
# ann_df['model_size'] = ann_df['hidden_list'].apply(lambda h: sum(h))
#
# plt.figure(figsize=(6, 4))
# plt.scatter(ann_df['model_size'], ann_df['val_rmse'])
# for _, row in ann_df.iterrows():
#     plt.text(row['model_size'], row['val_rmse'], str(row['model_index']),
#              ha='center', va='bottom', fontsize=8)
#
# plt.xlabel('隐藏层神经元总数（模型规模）')
# plt.ylabel('验证集 RMSE')
# plt.title('ANN 模型规模与验证误差的关系')
# plt.tight_layout()
# plt.savefig('fig_ann_modelsize_vs_valrmse.png', dpi=300)
# plt.show()



"""实验二"""


"""图 2-1 朴素贝叶斯融合 vs 最优 SVM 准确率比较"""
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# svm_df = pd.read_csv('student_svm_results_with_val.csv')
# nb_df = pd.read_csv('student_naive_bayes_results.csv')
#
# # 找到验证集准确率最高的 SVM
# best_svm = svm_df.sort_values('val_accuracy', ascending=False).iloc[0]
#
# splits = ['train', 'val', 'test']
# best_svm_acc = [
#     best_svm['train_accuracy'],
#     best_svm['val_accuracy'],
#     best_svm['test_accuracy'],
# ]
# nb_acc = nb_df.set_index('split').loc[splits, 'accuracy'].values
#
# x = range(len(splits))
# width = 0.35
#
# plt.figure(figsize=(6, 4))
# plt.bar([i - width/2 for i in x], best_svm_acc, width=width, label='最优 SVM')
# plt.bar([i + width/2 for i in x], nb_acc,       width=width, label='朴素贝叶斯融合')
#
# plt.xticks(x, ['Train', 'Val', 'Test'])
# plt.ylabel('准确率')
# plt.title('朴素贝叶斯融合 vs 最优 SVM 在不同划分上的准确率')
# plt.legend()
# plt.tight_layout()
# plt.savefig('fig_exp2_student_nb_vs_svm.png', dpi=300)
# plt.show()


"""图 2-2 学生测试集混淆矩阵"""
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# df = pd.read_csv('student_nb_test_predictions.csv')
# labels = ['Dropout', 'Enrolled', 'Graduate']
#
# cm = confusion_matrix(df['y_true'], df['y_pred'], labels=labels)
#
# fig, ax = plt.subplots(figsize=(5, 4))
# im = ax.imshow(cm, interpolation='nearest')
# ax.set_xticks(np.arange(len(labels)))
# ax.set_yticks(np.arange(len(labels)))
# ax.set_xticklabels(labels, rotation=45, ha='right')
# ax.set_yticklabels(labels)
#
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, cm[i, j], ha='center', va='center')
#
# ax.set_xlabel('预测标签')
# ax.set_ylabel('真实标签')
# ax.set_title('朴素贝叶斯融合器在学生测试集上的混淆矩阵')
# fig.colorbar(im)
# plt.tight_layout()
# plt.savefig('fig_exp2_student_nb_confusion.png', dpi=300)
# plt.show()


"""图 2-3 朴素贝叶斯融合vs最优ANN的MAE/RMSE对比"""
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# ann_df = pd.read_csv('nhanes_ann_results_with_val.csv')
# nb_df  = pd.read_csv('nhanes_naive_bayes_results.csv')
#
# # 取验证集 RMSE 最小的 ANN 作为最优模型
# best_ann = ann_df.sort_values('val_rmse', ascending=True).iloc[0]
#
# # 只比较验证集和测试集
# splits = ['val', 'test']
#
# best_ann_mae = [best_ann['val_mae'],  best_ann['test_mae']]
# best_ann_rmse = [best_ann['val_rmse'], best_ann['test_rmse']]
#
# nb_mae  = nb_df.set_index('split').loc[splits, 'mae'].values
# nb_rmse = nb_df.set_index('split').loc[splits, 'rmse'].values
#
# x = range(len(splits))
# width = 0.35
#
# plt.figure(figsize=(7, 4))
#
# # MAE
# plt.subplot(1, 2, 1)
# plt.bar([i - width/2 for i in x], best_ann_mae, width=width, label='最优 ANN')
# plt.bar([i + width/2 for i in x], nb_mae,      width=width, label='朴素贝叶斯融合')
# plt.xticks(x, ['Val', 'Test'])
# plt.ylabel('MAE')
# plt.title('MAE 对比')
# plt.legend()
#
# # RMSE
# plt.subplot(1, 2, 2)
# plt.bar([i - width/2 for i in x], best_ann_rmse, width=width, label='最优 ANN')
# plt.bar([i + width/2 for i in x], nb_rmse,      width=width, label='朴素贝叶斯融合')
# plt.xticks(x, ['Val', 'Test'])
# plt.ylabel('RMSE')
# plt.title('RMSE 对比')
# plt.legend()
#
# plt.suptitle('NHANES 年龄预测：朴素贝叶斯融合 vs 最优 ANN')
# plt.tight_layout()
# plt.savefig('fig_exp2_nhanes_nb_vs_ann.png', dpi=300)
# plt.show()



"""图 2-4 / 图 2-5 NHANES 年龄预测误差与散点图"""
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# df = pd.read_csv('nhanes_nb_test_predictions.csv')
#
# # 图 2-4：绝对误差直方图
# plt.figure(figsize=(6, 4))
# plt.hist(df['abs_error'], bins=20)
# plt.xlabel('绝对误差 |y_true - y_pred_int|')
# plt.ylabel('样本数')
# plt.title('朴素贝叶斯融合器在 NHANES 测试集上的年龄预测误差分布')
# plt.tight_layout()
# plt.savefig('fig_exp2_nhanes_nb_abs_error_hist.png', dpi=300)
# plt.show()
#
# # 图 2-5：真实年龄 vs 预测年龄散点图
# min_age = min(df['y_true'].min(), df['y_pred_int'].min())
# max_age = max(df['y_true'].max(), df['y_pred_int'].max())
#
# plt.figure(figsize=(6, 4))
# plt.scatter(df['y_true'], df['y_pred_int'], alpha=0.6)
# plt.plot([min_age, max_age], [min_age, max_age], linestyle='--')
# plt.xlabel('真实年龄')
# plt.ylabel('预测整数年龄')
# plt.title('朴素贝叶斯融合器在 NHANES 测试集上的预测散点图')
# plt.tight_layout()
# plt.savefig('fig_exp2_nhanes_nb_scatter.png', dpi=300)
# plt.show()



"""实验三"""

"""图 3-1 学生数据集：方法准确率对比"""
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']      # 或 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# def pick_path(*cands):
#     for p in cands:
#         if os.path.exists(p):
#             return p
#     raise FileNotFoundError("未找到文件：\n" + "\n".join(cands))
#
# svm_path = pick_path(os.path.join(BASE_DIR,'student_svm_results_with_val.csv'))
# nb_path  = pick_path(os.path.join(BASE_DIR,'student_naive_bayes_results.csv'),
#                      os.path.join(BASE_DIR,'student_naive_bayes_fusion','student_naive_bayes_results.csv'))
# ds_path  = pick_path(os.path.join(BASE_DIR,'student_ds_results.csv'),
#                      os.path.join(BASE_DIR,'student_ds_fusion','student_ds_results.csv'))
#
# svm_df = pd.read_csv(svm_path).sort_values('val_accuracy', ascending=False)
# best = svm_df.iloc[0]
#
# nb_df = pd.read_csv(nb_path).set_index('split')
# ds_df = pd.read_csv(ds_path).set_index('split')
#
# splits = ['val','test']
# best_acc = [best['val_accuracy'], best['test_accuracy']]
# nb_acc   = [nb_df.loc[s,'accuracy'] for s in splits]
# ds_acc   = [ds_df.loc[s,'accuracy'] for s in splits]
#
# x = range(len(splits)); w = 0.28
# plt.figure(figsize=(7,4))
# plt.bar([i-w for i in x], best_acc, width=w, label='最优SVM(实验一)')
# plt.bar([i    for i in x], nb_acc,   width=w, label='朴素贝叶斯融合(实验二)')
# plt.bar([i+w for i in x], ds_acc,   width=w, label='DS证据融合(实验三)')
# plt.xticks(list(x), ['Val','Test'])
# plt.ylabel('准确率')
# plt.title('学生数据集：不同方法在验证/测试集上的准确率对比')
# plt.legend()
# plt.tight_layout()
# plt.savefig('fig_exp3_student_method_compare.png', dpi=300)
# plt.show()



"""图3-2学生数据集：D-S融合混淆矩阵"""
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# def pick_path(*cands):
#     for p in cands:
#         if os.path.exists(p):
#             return p
#     raise FileNotFoundError("未找到文件：\n" + "\n".join(cands))
#
# test_path = pick_path(os.path.join(BASE_DIR,'student_ds_test_predictions.csv'),
#                       os.path.join(BASE_DIR,'student_ds_fusion','student_ds_test_predictions.csv'))
# df = pd.read_csv(test_path)
#
# labels = ['Dropout','Enrolled','Graduate']
# cm = confusion_matrix(df['y_true'], df['y_pred'], labels=labels)
#
# plt.figure(figsize=(5,4))
# im = plt.imshow(cm, interpolation='nearest')
# plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
# plt.yticks(np.arange(len(labels)), labels)
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, cm[i,j], ha='center', va='center')
# plt.xlabel('预测标签'); plt.ylabel('真实标签')
# plt.title('学生数据集：DS 融合测试集混淆矩阵')
# plt.colorbar(im)
# plt.tight_layout()
# plt.savefig('fig_exp3_student_ds_confusion.png', dpi=300)
# plt.show()



"""图3-3 NHANES：MAE/RMSE方法对比"""
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# def pick_path(*cands):
#     for p in cands:
#         if os.path.exists(p):
#             return p
#     raise FileNotFoundError("未找到文件：\n" + "\n".join(cands))
#
# ann_path = pick_path(os.path.join(BASE_DIR,'nhanes_ann_results_with_val.csv'))
# nb_path  = pick_path(os.path.join(BASE_DIR,'nhanes_naive_bayes_results.csv'),
#                      os.path.join(BASE_DIR,'nhanes_naive_bayes_fusion','nhanes_naive_bayes_results.csv'))
# ds_path  = pick_path(os.path.join(BASE_DIR,'nhanes_ds_results.csv'),
#                      os.path.join(BASE_DIR,'nhanes_ds_fusion','nhanes_ds_results.csv'))
#
# ann_df = pd.read_csv(ann_path).sort_values('val_rmse', ascending=True)
# best = ann_df.iloc[0]
# nb = pd.read_csv(nb_path).set_index('split')
# ds = pd.read_csv(ds_path).set_index('split')
#
# # 比较测试集
# methods = ['最优ANN(实验一)','朴素贝叶斯融合(实验二)','DS证据融合(实验三)']
# mae = [best['test_mae'], nb.loc['test','mae'], ds.loc['test','mae']]
# rmse = [best['test_rmse'], nb.loc['test','rmse'], ds.loc['test','rmse']]
#
# plt.figure(figsize=(7,4))
# x = range(len(methods)); w = 0.35
# plt.bar([i-w/2 for i in x], mae,  width=w, label='MAE')
# plt.bar([i+w/2 for i in x], rmse, width=w, label='RMSE')
# plt.xticks(list(x), methods, rotation=15, ha='right')
# plt.ylabel('误差')
# plt.title('NHANES：不同方法在测试集上的误差对比')
# plt.legend()
# plt.tight_layout()
# plt.savefig('fig_exp3_nhanes_mae_rmse_compare.png', dpi=300)
# plt.show()



"""图 3-4 NHANES：误差分布（DS vs NB）"""
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def pick_path(*cands):
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("未找到文件：\n" + "\n".join(cands))

ds_test = pick_path(os.path.join(BASE_DIR,'nhanes_ds_test_predictions.csv'),
                    os.path.join(BASE_DIR,'nhanes_ds_fusion','nhanes_ds_test_predictions.csv'))
nb_test = pick_path(os.path.join(BASE_DIR,'nhanes_nb_test_predictions.csv'),
                    os.path.join(BASE_DIR,'nhanes_naive_bayes_fusion','nhanes_nb_test_predictions.csv'))

ds_df = pd.read_csv(ds_test)
nb_df = pd.read_csv(nb_test)

plt.figure(figsize=(7,4))
plt.hist(nb_df['abs_error'], bins=20, alpha=0.6, label='NB融合 abs_error')
plt.hist(ds_df['abs_error'], bins=20, alpha=0.6, label='DS融合 abs_error')
plt.xlabel('绝对误差 |y_true - y_pred_int|')
plt.ylabel('样本数')
plt.title('NHANES：融合方法误差分布对比（测试集）')
plt.legend()
plt.tight_layout()
plt.savefig('fig_exp3_nhanes_error_hist.png', dpi=300)
plt.show()


