# =====================================================
# 步骤5: 最终模型评估
# 训练最终模型并在测试集上评估
# =====================================================

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# 加载参数
with open('saved_data/params.pkl', 'rb') as f:
    params = pickle.load(f)
    RANDOM_STATE = params['RANDOM_STATE']
    SMOTE_RATIO = params['SMOTE_RATIO']

np.random.seed(RANDOM_STATE)

print("="*70)
print("步骤5: 最终模型评估")
print("="*70)


# =====================================================
# 加载数据
# =====================================================

print("\n加载数据...")
X_train = np.load('saved_data/X_train.npy')
X_test = np.load('saved_data/X_test.npy')
y_train = np.load('saved_data/y_train.npy')
y_test = np.load('saved_data/y_test.npy')

with open('saved_data/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

with open('saved_data/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print(f"训练集: {X_train.shape[0]:,} 样本")
print(f"测试集: {X_test.shape[0]:,} 样本")


# =====================================================
# 训练最终模型
# =====================================================

print("\n" + "="*70)
print("训练最终模型")
print("="*70)

# SMOTE
print(f"\n应用 SMOTE (比例: {SMOTE_RATIO*100:.0f}%)...")
unique, counts = np.unique(y_train, return_counts=True)
majority_count = max(counts)
target_count = int(majority_count * SMOTE_RATIO)

sampling_strategy = {}
for u, c in zip(unique, counts):
    if c < target_count:
        sampling_strategy[u] = target_count
    else:
        sampling_strategy[u] = c

min_class_size = min(counts[counts > 1]) if any(counts > 1) else 2
k_neighbors = min(3, min_class_size - 1) if min_class_size > 1 else 1

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=k_neighbors)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"  数据量: {X_train.shape[0]:,} → {X_smote.shape[0]:,}")

# Class Weight
print("\n计算类别权重...")
classes = np.unique(y_smote)
class_weights = compute_class_weight('balanced', classes=classes, y=y_smote)
sample_weights = np.array([class_weights[np.where(classes == c)[0][0]] for c in y_smote])

# 训练
print("\n训练 XGBoost...")
start = time.time()
final_model = xgb.XGBClassifier(**best_params, random_state=RANDOM_STATE,
                                 tree_method='hist', eval_metric='mlogloss',
                                 use_label_encoder=False, n_jobs=-1)
final_model.fit(X_smote, y_smote, sample_weight=sample_weights)
print(f"✅ 训练完成，耗时: {time.time() - start:.1f}s")


# =====================================================
# 测试集评估
# =====================================================

print("\n" + "="*70)
print("测试集评估")
print("="*70)

y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n整体性能:")
print(f"  准确率: {accuracy*100:.2f}%")
print(f"  Macro-精确率: {precision_macro:.4f}")
print(f"  Macro-召回率: {recall_macro:.4f}")
print(f"  Macro-F1: {f1_macro:.4f}")
print(f"  Weighted-F1: {f1_weighted:.4f}")

# 各类别性能
precision_per_class, recall_per_class, f1_per_class, support = \
    precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

print(f"\n各类别性能:")
print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
print("-"*55)
for i in range(len(precision_per_class)):
    class_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
    print(f"{class_name:<15} {precision_per_class[i]:<10.3f} {recall_per_class[i]:<10.3f} {f1_per_class[i]:<10.3f} {support[i]:<10}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

if class_names and len(class_names) <= 15:
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('归一化混淆矩阵', fontsize=14)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print(f"\n✅ 混淆矩阵已保存为 confusion_matrix.png")

# 保存结果
test_results = {
    'accuracy': accuracy,
    'precision_macro': precision_macro,
    'recall_macro': recall_macro,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted,
    'per_class_f1': f1_per_class.tolist(),
    'support': support.tolist()
}

with open('saved_data/final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('saved_data/test_results.pkl', 'wb') as f:
    pickle.dump(test_results, f)

print("\n✅ 结果已保存到 saved_data/ 文件夹:")
print("   - final_model.pkl")
print("   - test_results.pkl")

print("\n" + "="*70)
print("实验全部完成!")
print("="*70)
print(f"\n最终测试集 Macro-F1: {f1_macro:.4f}")
print(f"最终测试集准确率: {accuracy*100:.2f}%")