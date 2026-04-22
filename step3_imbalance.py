# =====================================================
# 步骤3: 不平衡处理方法对比（输出完整指标）
# =====================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

# 加载参数
with open('saved_data/params.pkl', 'rb') as f:
    params = pickle.load(f)
    RANDOM_STATE = params['RANDOM_STATE']
    SMOTE_RATIO = params['SMOTE_RATIO']

np.random.seed(RANDOM_STATE)

print("="*70)
print("步骤3: 不平衡处理方法对比")
print(f"SMOTE比例: {SMOTE_RATIO*100:.0f}%")
print("="*70)


# =====================================================
# 加载数据
# =====================================================

print("\n加载数据...")
X_train = np.load('saved_data/X_train.npy')
X_val = np.load('saved_data/X_val.npy')
y_train = np.load('saved_data/y_train.npy')
y_val = np.load('saved_data/y_val.npy')

with open('saved_data/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

print(f"训练集: {X_train.shape[0]:,} 样本")
print(f"验证集: {X_val.shape[0]:,} 样本")


# =====================================================
# SMOTE函数
# =====================================================

def apply_smote_controlled(X_train, y_train, target_ratio=SMOTE_RATIO):
    print(f"\n应用控制 SMOTE (少数类增加到多数类的 {target_ratio*100:.0f}%)...")
    start_time = time.time()
    
    unique, counts = np.unique(y_train, return_counts=True)
    majority_count = max(counts)
    target_count = int(majority_count * target_ratio)
    
    sampling_strategy = {}
    for u, c in zip(unique, counts):
        if c < target_count:
            sampling_strategy[u] = target_count
        else:
            sampling_strategy[u] = c
    
    min_class_size = min(counts[counts > 1]) if any(counts > 1) else 2
    k_neighbors = min(3, min_class_size - 1) if min_class_size > 1 else 1
    
    smote = SMOTE(sampling_strategy=sampling_strategy, 
                  random_state=RANDOM_STATE, 
                  k_neighbors=k_neighbors)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  数据量: {X_train.shape[0]:,} → {X_resampled.shape[0]:,}")
    print(f"  ✅ SMOTE 完成，耗时: {time.time() - start_time:.1f} 秒")
    
    return X_resampled, y_resampled


# =====================================================
# 对比4种方法
# =====================================================

print("\n" + "="*70)
print("开始对比4种不平衡处理方法")
print("="*70)

results = []
base_config = {
    'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6,
    'random_state': RANDOM_STATE, 'tree_method': 'hist',
    'eval_metric': 'mlogloss', 'use_label_encoder': False, 'n_jobs': -1
}

# 1. 无处理（基线）
print("\n1️⃣ 无处理（基线）")
start = time.time()
model = xgb.XGBClassifier(**base_config)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

acc_no = accuracy_score(y_val, y_pred)
prec_no = precision_score(y_val, y_pred, average='macro', zero_division=0)
rec_no = recall_score(y_val, y_pred, average='macro', zero_division=0)
f1_no = f1_score(y_val, y_pred, average='macro', zero_division=0)

results.append({
    'Method': 'No Handling',
    'Accuracy': acc_no,
    'Precision': prec_no,
    'Recall': rec_no,
    'Macro-F1': f1_no,
    'Time': time.time() - start
})
print(f"   准确率: {acc_no*100:.2f}%")
print(f"   Macro-精确率: {prec_no:.4f}")
print(f"   Macro-召回率: {rec_no:.4f}")
print(f"   Macro-F1: {f1_no:.4f}")

# 2. SMOTE Only
print(f"\n2️⃣ SMOTE Only ({SMOTE_RATIO*100:.0f}%)")
start = time.time()
X_smote, y_smote = apply_smote_controlled(X_train, y_train)
model = xgb.XGBClassifier(**base_config)
model.fit(X_smote, y_smote)
y_pred = model.predict(X_val)

acc_smote = accuracy_score(y_val, y_pred)
prec_smote = precision_score(y_val, y_pred, average='macro', zero_division=0)
rec_smote = recall_score(y_val, y_pred, average='macro', zero_division=0)
f1_smote = f1_score(y_val, y_pred, average='macro', zero_division=0)

results.append({
    'Method': 'SMOTE Only',
    'Accuracy': acc_smote,
    'Precision': prec_smote,
    'Recall': rec_smote,
    'Macro-F1': f1_smote,
    'Time': time.time() - start
})
print(f"   准确率: {acc_smote*100:.2f}%")
print(f"   Macro-精确率: {prec_smote:.4f}")
print(f"   Macro-召回率: {rec_smote:.4f}")
print(f"   Macro-F1: {f1_smote:.4f}")

# 3. Class Weighting
print("\n3️⃣ Class Weighting")
start = time.time()
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
sample_weights = np.array([class_weights[np.where(classes == c)[0][0]] for c in y_train])
model = xgb.XGBClassifier(**base_config)
model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred = model.predict(X_val)

acc_cw = accuracy_score(y_val, y_pred)
prec_cw = precision_score(y_val, y_pred, average='macro', zero_division=0)
rec_cw = recall_score(y_val, y_pred, average='macro', zero_division=0)
f1_cw = f1_score(y_val, y_pred, average='macro', zero_division=0)

results.append({
    'Method': 'Class Weighting',
    'Accuracy': acc_cw,
    'Precision': prec_cw,
    'Recall': rec_cw,
    'Macro-F1': f1_cw,
    'Time': time.time() - start
})
print(f"   准确率: {acc_cw*100:.2f}%")
print(f"   Macro-精确率: {prec_cw:.4f}")
print(f"   Macro-召回率: {rec_cw:.4f}")
print(f"   Macro-F1: {f1_cw:.4f}")

# 4. SMOTE + Class Weighting
print("\n4️⃣ SMOTE + Class Weighting")
start = time.time()
X_smote2, y_smote2 = apply_smote_controlled(X_train, y_train)
classes2 = np.unique(y_smote2)
class_weights2 = compute_class_weight('balanced', classes=classes2, y=y_smote2)
sample_weights2 = np.array([class_weights2[np.where(classes2 == c)[0][0]] for c in y_smote2])
model = xgb.XGBClassifier(**base_config)
model.fit(X_smote2, y_smote2, sample_weight=sample_weights2)
y_pred = model.predict(X_val)

acc_comb = accuracy_score(y_val, y_pred)
prec_comb = precision_score(y_val, y_pred, average='macro', zero_division=0)
rec_comb = recall_score(y_val, y_pred, average='macro', zero_division=0)
f1_comb = f1_score(y_val, y_pred, average='macro', zero_division=0)

results.append({
    'Method': 'SMOTE + Class Weighting',
    'Accuracy': acc_comb,
    'Precision': prec_comb,
    'Recall': rec_comb,
    'Macro-F1': f1_comb,
    'Time': time.time() - start
})
print(f"   准确率: {acc_comb*100:.2f}%")
print(f"   Macro-精确率: {prec_comb:.4f}")
print(f"   Macro-召回率: {rec_comb:.4f}")
print(f"   Macro-F1: {f1_comb:.4f}")


# =====================================================
# 结果汇总
# =====================================================

print("\n" + "="*70)
print("不平衡处理方法对比结果")
print("="*70)

results_df = pd.DataFrame(results)

# 显示完整表格
print(f"\n{'Method':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Macro-F1':<10} {'Time':<10}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['Method']:<25} {row['Accuracy']*100:>6.2f}%   {row['Precision']:.4f}    {row['Recall']:.4f}    {row['Macro-F1']:.4f}    {row['Time']:.1f}s")

# 保存原始数据
results_df.to_csv('saved_data/imbalance_results.csv', index=False)

# 找出最佳方法
best_idx = results_df['Macro-F1'].idxmax()
best_method = results_df.loc[best_idx, 'Method']
best_f1 = results_df.loc[best_idx, 'Macro-F1']

print(f"\n🏆 最佳方法: {best_method} (Macro-F1 = {best_f1:.4f})")

if f1_smote > f1_cw:
    print(f"✅ SMOTE ({f1_smote:.4f}) > Class Weight ({f1_cw:.4f})")
else:
    print(f"⚠️ SMOTE ({f1_smote:.4f}) <= Class Weight ({f1_cw:.4f})")

print("\n✅ 结果已保存到 saved_data/imbalance_results.csv")
# 创建 images 文件夹
os.makedirs('images', exist_ok=True)

# 方法1：从保存的 CSV 文件读取数据
csv_path = 'saved_data/imbalance_results.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    techniques = df['Method'].tolist()
    macro_f1 = df['Macro-F1'].tolist()
    recall = df['Recall'].tolist()
else:
    # 方法2：手动输入数据（根据你的步骤3输出）
    techniques = ['No Handling', 'SMOTE Only', 'Class Weighting', 'SMOTE + Class Weighting']
    macro_f1 = [0.5965, 0.6040, 0.5917, 0.5920]
    recall = [0.5856, 0.6641, 0.7135, 0.7046]

print("数据:")
for t, f1, r in zip(techniques, macro_f1, recall):
    print(f"  {t}: Macro-F1={f1:.4f}, Recall={r:.4f}")

# 设置图形
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱状图位置
x = np.arange(len(techniques))
width = 0.35

# 绘制柱状图
bars1 = ax.bar(x - width/2, macro_f1, width, label='Macro-F1', 
               color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, recall, width, label='Macro-Recall', 
               color='coral', edgecolor='black', linewidth=0.5)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# 设置标签和标题
ax.set_xlabel('Imbalance Handling Technique', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison of Imbalance Handling Techniques', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(techniques, fontsize=10, rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 保存图片
plt.savefig('images/imbalance_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n✅ 图片已保存为 images/imbalance_comparison.png")
print("\n" + "="*70)
print("步骤3完成! 可以运行 step4_optimize.py")
print("="*70)