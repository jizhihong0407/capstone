# =====================================================
# 步骤4: 超参数优化
# 使用Optuna优化XGBoost超参数
# =====================================================

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
import warnings
import time
import pickle
import os
import random

warnings.filterwarnings('ignore')

# 加载参数
with open('saved_data/params.pkl', 'rb') as f:
    params = pickle.load(f)
    RANDOM_STATE = params['RANDOM_STATE']
    SMOTE_RATIO = params['SMOTE_RATIO']
    OPTUNA_TRIALS = params['OPTUNA_TRIALS']

# 固定所有随机种子
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

print("="*70)
print("步骤4: 超参数优化")
print(f"随机种子: {RANDOM_STATE}")
print(f"Optuna试验次数: {OPTUNA_TRIALS}")
print("="*70)


# =====================================================
# 加载数据
# =====================================================

print("\n加载数据...")
X_train = np.load('saved_data/X_train.npy')
X_val = np.load('saved_data/X_val.npy')
y_train = np.load('saved_data/y_train.npy')
y_val = np.load('saved_data/y_val.npy')

print(f"训练集: {X_train.shape[0]:,} 样本")
print(f"验证集: {X_val.shape[0]:,} 样本")


# =====================================================
# 准备优化数据（应用SMOTE）
# =====================================================

print("\n准备优化数据（应用SMOTE）...")
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
X_train_opt, y_train_opt = smote.fit_resample(X_train, y_train)

print(f"优化数据量: {X_train_opt.shape[0]:,}")


# =====================================================
# Optuna优化
# =====================================================

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 3),
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    # 只使用SMOTE，不加类别权重
    model.fit(X_train_opt, y_train_opt)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='macro', zero_division=0)


print("\n" + "="*70)
print(f"开始超参数优化 (试验次数: {OPTUNA_TRIALS})")
print("="*70)

start_time = time.time()

# 固定Optuna种子以确保可重复性
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

elapsed = time.time() - start_time

best_params = study.best_params
best_score = study.best_value


# =====================================================
# 保存结果
# =====================================================

print("\n" + "="*70)
print("优化完成!")
print("="*70)
print(f"耗时: {elapsed/60:.1f} 分钟")
print(f"最佳 Macro-F1: {best_score:.4f}")
print("最佳超参数:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# 保存优化结果
with open('saved_data/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

with open('saved_data/best_score.pkl', 'wb') as f:
    pickle.dump(best_score, f)

# 保存完整的study结果（可选）
with open('saved_data/optuna_study.pkl', 'wb') as f:
    pickle.dump(study, f)

print("\n✅ 结果已保存到 saved_data/ 文件夹:")
print("   - best_params.pkl")
print("   - best_score.pkl")
print("   - best_params.pkl")


# =====================================================
# 生成 Optuna 收敛曲线图
# =====================================================

print("\n" + "="*70)
print("生成收敛曲线图")
print("="*70)

# 创建 images 文件夹（如果不存在）
os.makedirs('images', exist_ok=True)

# 获取所有 trial 的数据
trials_df = study.trials_dataframe()
trial_numbers = trials_df['number'].values
best_values = trials_df['value'].cummax().values
all_values = trials_df['value'].values

# 计算移动平均
window = 5
if len(all_values) >= window:
    moving_avg = np.convolve(all_values, np.ones(window)/window, mode='valid')
else:
    moving_avg = all_values

# 创建图形
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 子图1：所有 trials 和最佳值曲线
ax1 = axes[0]
ax1.plot(trial_numbers, all_values, 'o-', alpha=0.5, markersize=4, 
         label='Trial Macro-F1', color='gray', linewidth=1)
ax1.plot(trial_numbers, best_values, 's-', linewidth=2, markersize=5, 
         label='Best Macro-F1 So Far', color='blue')
ax1.axhline(y=best_score, color='red', linestyle='--', linewidth=1.5, 
            label=f'Final Best: {best_score:.4f}')
ax1.set_xlabel('Trial Number', fontsize=12)
ax1.set_ylabel('Macro-F1 Score', fontsize=12)
ax1.set_title('Bayesian Optimization Convergence', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 子图2：移动平均平滑曲线
ax2 = axes[1]
if len(moving_avg) >= window:
    ax2.plot(range(window-1, len(all_values)), moving_avg, 'o-', 
             color='green', linewidth=2, markersize=4)
else:
    ax2.plot(trial_numbers, moving_avg, 'o-', color='green', linewidth=2, markersize=4)
ax2.set_xlabel('Trial Number', fontsize=12)
ax2.set_ylabel(f'Moving Average (window={window})', fontsize=12)
ax2.set_title('Smoothed Convergence Curve', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
save_path = 'images/optuna_convergence.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ 收敛曲线已保存为 {save_path}")

# 也保存一份到 saved_data 文件夹
shutil_path = 'saved_data/optuna_convergence.png'
plt.savefig(shutil_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ 收敛曲线已保存为 {shutil_path}")

print("\n" + "="*70)
print("步骤4完成! 可以运行 step5_final.py")
print("="*70)