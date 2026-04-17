# =====================================================
# 步骤2: 基线模型对比（显示完整指标）
# =====================================================

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

np.random.seed(RANDOM_STATE)

print("="*70)
print("步骤2: 基线模型对比")
print("="*70)


# =====================================================
# 加载数据
# =====================================================

print("\n加载预处理数据...")
X_train = np.load('saved_data/X_train.npy')
X_val = np.load('saved_data/X_val.npy')
y_train = np.load('saved_data/y_train.npy')
y_val = np.load('saved_data/y_val.npy')

with open('saved_data/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

print(f"训练集: {X_train.shape[0]:,} 样本")
print(f"验证集: {X_val.shape[0]:,} 样本")


# =====================================================
# 基线模型
# =====================================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                  random_state=RANDOM_STATE, tree_method='hist',
                                  eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1),
    'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
}

results = []
best_model_name = None
best_f1 = 0

print("\n" + "="*70)
print("开始训练基线模型")
print("="*70)

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"模型: {name}")
    print('='*50)
    start = time.time()
    
    try:
        if name == 'SVM' and X_train.shape[0] > 50000:
            print(f"  ⚠️ SVM使用50,000样本子集")
            sss = StratifiedShuffleSplit(n_splits=1, train_size=50000, random_state=RANDOM_STATE)
            for idx, _ in sss.split(X_train, y_train):
                X_sub, y_sub = X_train[idx], y_train[idx]
            model.fit(X_sub, y_sub)
            note = " (50k子集)"
        else:
            model.fit(X_train, y_train)
            note = ""
        
        train_time = time.time() - start
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        
        results.append({
            'Model': name + note,
            'Accuracy': accuracy,
            'Macro-Precision': precision_macro,
            'Macro-Recall': recall_macro,
            'Macro-F1': f1_macro,
            'Time': train_time
        })
        
        print(f"  准确率: {accuracy*100:.2f}%")
        print(f"  Macro-精确率: {precision_macro:.4f}")
        print(f"  Macro-召回率: {recall_macro:.4f}")
        print(f"  Macro-F1: {f1_macro:.4f}")
        print(f"  训练时间: {train_time:.1f}s")
        
        if f1_macro > best_f1 and name != 'SVM':
            best_f1 = f1_macro
            best_model_name = name
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        results.append({'Model': name, 'Accuracy': 0, 'Macro-Precision': 0, 
                       'Macro-Recall': 0, 'Macro-F1': 0, 'Time': 0})


# =====================================================
# 保存结果
# =====================================================

print("\n" + "="*70)
print("基线模型性能汇总")
print("="*70)

results_df = pd.DataFrame(results)

# 格式化输出
display_df = results_df.copy()
display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
display_df['Macro-Precision'] = display_df['Macro-Precision'].apply(lambda x: f"{x:.4f}")
display_df['Macro-Recall'] = display_df['Macro-Recall'].apply(lambda x: f"{x:.4f}")
display_df['Macro-F1'] = display_df['Macro-F1'].apply(lambda x: f"{x:.4f}")
display_df['Time'] = display_df['Time'].apply(lambda x: f"{x:.1f}s")

print(display_df.to_string(index=False))

print(f"\n🏆 最优基线模型: {best_model_name} (Macro-F1 = {best_f1:.4f})")

# 保存原始数据（用于论文）
results_df.to_csv('saved_data/baseline_results.csv', index=False)

with open('saved_data/best_model_name.pkl', 'wb') as f:
    pickle.dump(best_model_name, f)

print("\n✅ 结果已保存到 saved_data/ 文件夹")
print("\n" + "="*70)
print("步骤2完成! 可以运行 step3_imbalance.py")
print("="*70)