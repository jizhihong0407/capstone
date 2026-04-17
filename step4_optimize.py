# =====================================================
# 步骤4: 超参数优化
# 使用Optuna优化XGBoost超参数
# =====================================================

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
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
    OPTUNA_TRIALS = params['OPTUNA_TRIALS']

np.random.seed(RANDOM_STATE)

print("="*70)
print("步骤4: 超参数优化")
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

# 计算类别权重
classes = np.unique(y_train_opt)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_opt)
sample_weights = np.array([class_weights[np.where(classes == c)[0][0]] for c in y_train_opt])

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
    model.fit(X_train_opt, y_train_opt, sample_weight=sample_weights)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='macro', zero_division=0)


print("\n" + "="*70)
print(f"开始超参数优化 (试验次数: {OPTUNA_TRIALS})")
print("="*70)

start_time = time.time()

study = optuna.create_study(direction='maximize')
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

with open('saved_data/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

with open('saved_data/best_score.pkl', 'wb') as f:
    pickle.dump(best_score, f)

print("\n✅ 结果已保存到 saved_data/ 文件夹:")
print("   - best_params.pkl")
print("   - best_score.pkl")

print("\n" + "="*70)
print("步骤4完成! 可以运行 step5_final.py")
print("="*70)