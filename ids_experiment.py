# =====================================================
# 网络入侵检测系统 - 全量数据 + 控制 SMOTE
# 包含 SVM（使用子集）
# 预计运行时间: 45-60 分钟
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import optuna
import warnings
import time
import pickle
import os
import glob

warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 设置图形样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("="*70)
print("网络入侵检测系统 - 全量数据 + 控制 SMOTE（含 SVM）")
print("预计运行时间: 45-60 分钟")
print("="*70)


# =====================================================
# 1. 数据加载函数
# =====================================================

def load_unsw_nb15_from_folder(folder_path='unsw_data'):
    """从文件夹加载UNSW-NB15数据集"""
    
    print(f"\n正在从文件夹 '{folder_path}' 加载全量数据...")
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return None
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    print(f"\n找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        size = os.path.getsize(f) / (1024 * 1024)
        if size > 0.5:
            print(f"  - {os.path.basename(f)} ({size:.1f} MB)")
    
    dfs = []
    
    for file in csv_files:
        filename = os.path.basename(file)
        file_size = os.path.getsize(file) / (1024 * 1024)
        
        # 跳过说明文件和小文件
        if 'feature' in filename.lower() and file_size < 1:
            continue
        if file_size < 0.5:
            continue
        
        print(f"\n加载: {filename} ({file_size:.1f} MB)")
        
        try:
            df = pd.read_csv(file)
            print(f"  ✅ {len(df):,} 条记录")
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    if not dfs:
        print("没有成功加载任何文件!")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n{'='*50}")
    print(f"合并完成! 总记录数: {len(combined_df):,}")
    print(f"{'='*50}")
    
    return combined_df


# =====================================================
# 2. 数据预处理类
# =====================================================

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.class_names = None
        
    def load_data_from_folder(self, folder_path='unsw_data'):
        """从文件夹加载数据"""
        df = load_unsw_nb15_from_folder(folder_path)
        if df is None:
            return None, None, None
        return self.preprocess(df)
    
    def preprocess(self, df):
        """执行数据预处理"""
        print("\n" + "="*60)
        print("开始数据预处理")
        print("="*60)
        
        df_processed = df.copy()
        
        # 1. 删除不必要的列
        columns_to_drop = ['srcip', 'dstip', 'Source IP', 'Destination IP', 
                           'Start time', 'Last time', 'No.', 'id']
        for col in columns_to_drop:
            for c in df_processed.columns:
                if col.lower() in c.lower():
                    df_processed = df_processed.drop(columns=[c], errors='ignore')
        
        # 2. 处理缺失值
        print("\n处理缺失值...")
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['float64', 'int64']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # 3. 编码分类特征
        print("\n编码分类特征...")
        categorical_features = ['proto', 'state', 'service']
        
        for col in categorical_features:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} 个类别")
        
        # 4. 处理攻击类别标签
        print("\n处理攻击类别标签...")
        if 'attack_cat' in df_processed.columns:
            le_attack = LabelEncoder()
            df_processed['attack_cat_encoded'] = le_attack.fit_transform(df_processed['attack_cat'].astype(str))
            self.label_encoders['attack_cat'] = le_attack
            self.class_names = le_attack.classes_.tolist()
            print(f"  攻击类别: {len(self.class_names)} 种")
        
        # 5. 确保 label 列是数值
        if 'label' in df_processed.columns:
            df_processed['label'] = pd.to_numeric(df_processed['label'], errors='coerce').fillna(0).astype(int)
        
        # 6. 分离特征和标签
        print("\n分离特征和标签...")
        exclude_cols = ['label', 'attack_cat', 'attack_cat_encoded']
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        X = df_processed[feature_cols].select_dtypes(include=[np.number])
        
        y_multiclass = df_processed['attack_cat_encoded'].values if 'attack_cat_encoded' in df_processed.columns else None
        y_binary = df_processed['label'].values if 'label' in df_processed.columns else None
        
        print(f"特征矩阵形状: {X.shape}")
        
        if y_multiclass is not None:
            unique, counts = np.unique(y_multiclass, return_counts=True)
            print(f"\n多分类标签分布:")
            for u, c in zip(unique, counts):
                class_name = self.class_names[u] if self.class_names and u < len(self.class_names) else str(u)
                print(f"  {class_name}: {c:,}")
        
        # 7. 特征缩放
        print("\n执行特征缩放...")
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        print("预处理完成!")
        
        return X_scaled, y_binary, y_multiclass
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15):
        """分割数据集"""
        print("\n" + "="*60)
        print("分割数据集")
        print("="*60)
        
        # 首次分割: 分离测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 第二次分割: 分离验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        print(f"\n训练集: {X_train.shape[0]:,} 样本")
        print(f"验证集: {X_val.shape[0]:,} 样本")
        print(f"测试集: {X_test.shape[0]:,} 样本")
        
        # 显示训练集类别分布
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n训练集类别分布:")
        for u, c in zip(unique, counts):
            class_name = self.class_names[u] if self.class_names and u < len(self.class_names) else str(u)
            print(f"  {class_name}: {c:,} ({c/len(y_train)*100:.2f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================
# 3. 基线模型类（包含 SVM）
# =====================================================

class BaselineModels:
    """基线模型训练类 - SVM 使用子集"""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        
        self.models_config = {
            'Logistic Regression': LogisticRegression(
                max_iter=200, random_state=random_state, n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, random_state=random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50, max_depth=10, n_jobs=-1, random_state=random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=random_state, tree_method='hist',
                eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', gamma='scale',
                probability=True, random_state=random_state
            )
        }
    
    def train_model(self, model_name, X_train, y_train):
        """训练单个模型"""
        print(f"训练 {model_name}...")
        start_time = time.time()
        
        model = self.models_config[model_name]
        
        # ========== SVM 特殊处理：使用子集 ==========
        if model_name == 'SVM':
            # SVM 使用 50,000 条子集（与论文表4.1一致）
            max_svm_samples = 50000
            
            if X_train.shape[0] > max_svm_samples:
                print(f"  ⚠️ SVM 使用 {max_svm_samples:,} 样本子集（全量 {X_train.shape[0]:,} 太大）")
                
                # 分层采样保持类别比例
                sss = StratifiedShuffleSplit(
                    n_splits=1, 
                    test_size=max_svm_samples / X_train.shape[0],
                    random_state=self.random_state
                )
                
                for _, idx in sss.split(X_train, y_train):
                    X_train_sub = X_train[idx]
                    y_train_sub = y_train[idx]
                
                print(f"  SVM 训练集: {X_train_sub.shape[0]:,} 样本")
                model.fit(X_train_sub, y_train_sub)
            else:
                model.fit(X_train, y_train)
        
        # ========== Logistic Regression 也使用子集加速 ==========
        elif model_name == 'Logistic Regression' and X_train.shape[0] > 500000:
            print(f"  ⚠️ Logistic Regression 使用 500,000 样本子集")
            indices = np.random.choice(X_train.shape[0], 500000, replace=False)
            X_train_sub = X_train[indices]
            y_train_sub = y_train[indices]
            model.fit(X_train_sub, y_train_sub)
        
        else:
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        return model, train_time
    
    def evaluate_model(self, model, X_val, y_val):
        """评估模型"""
        y_pred = model.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_val, y_pred)
        return {'f1_macro': f1_macro, 'accuracy': accuracy}
    
    def run_all_baselines(self, X_train, X_val, y_train, y_val):
        """运行所有基线模型"""
        print("\n" + "="*70)
        print("基线模型训练")
        print("="*70)
        
        results = []
        
        for model_name in self.models_config.keys():
            print(f"\n{'='*50}")
            print(f"模型: {model_name}")
            print('='*50)
            
            try:
                model, train_time = self.train_model(model_name, X_train, y_train)
                metrics = self.evaluate_model(model, X_val, y_val)
                
                # 添加说明
                note = ""
                if model_name == 'SVM' and X_train.shape[0] > 50000:
                    note = " (50k子集)"
                elif model_name == 'Logistic Regression' and X_train.shape[0] > 500000:
                    note = " (500k子集)"
                
                results.append({
                    'Model': model_name + note,
                    'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                    'Macro-Precision': "N/A",
                    'Macro-Recall': "N/A",
                    'Macro-F1': f"{metrics['f1_macro']:.4f}",
                    'Time': f"{train_time:.1f}s"
                })
                print(f"  ✅ 准确率: {metrics['accuracy']*100:.2f}%")
                print(f"  ✅ Macro-F1: {metrics['f1_macro']:.4f}")
                print(f"  ⏱️ 训练时间: {train_time:.1f}秒")
                
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                results.append({
                    'Model': model_name,
                    'Accuracy': 'Error',
                    'Macro-Precision': 'Error',
                    'Macro-Recall': 'Error',
                    'Macro-F1': 'Error',
                    'Time': 'Error'
                })
        
        print("\n" + "="*70)
        print("基线模型性能汇总")
        print("="*70)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        return results_df


# =====================================================
# 4. 不平衡处理类（控制 SMOTE）
# =====================================================

class ImbalanceHandler:
    """不平衡处理类 - 控制 SMOTE 生成数量"""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
    
    def apply_smote_controlled(self, X_train, y_train, target_ratio=0.3):
        """
        控制 SMOTE：将少数类增加到多数类的 target_ratio 倍
        
        Parameters:
        -----------
        target_ratio : float
            少数类目标数量 = 多数类数量 × target_ratio
            例如 0.3 表示少数类增加到多数类的 30%
        """
        print(f"\n应用控制 SMOTE (少数类增加到多数类的 {target_ratio*100:.0f}%)...")
        start_time = time.time()
        
        unique, counts = np.unique(y_train, return_counts=True)
        majority_count = max(counts)
        target_count = int(majority_count * target_ratio)
        
        print(f"\n原始类别分布:")
        for u, c in zip(unique, counts):
            print(f"  类别 {u}: {c:,} 条")
        
        print(f"\n多数类数量: {majority_count:,}")
        print(f"目标数量: {target_count:,} ({target_ratio*100:.0f}% of majority)")
        
        # 构建采样策略
        sampling_strategy = {}
        total_target = 0
        
        for u, c in zip(unique, counts):
            if c < target_count:
                sampling_strategy[u] = target_count
                total_target += target_count
            else:
                sampling_strategy[u] = c
                total_target += c
        
        print(f"\n采样策略:")
        for u, target in sampling_strategy.items():
            print(f"  类别 {u}: {target:,} 条")
        
        print(f"\n数据量变化: {X_train.shape[0]:,} → {total_target:,} (增加 {(total_target/X_train.shape[0]-1)*100:.1f}%)")
        
        # 应用 SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=3
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SMOTE 完成，耗时: {elapsed:.1f} 秒")
        
        return X_resampled, y_resampled
    
    def evaluate_handling_methods(self, X_train, X_val, y_train, y_val, class_names=None):
        """
        评估不平衡处理方法 - 使用控制 SMOTE
        预期结果符合论文：
        - No Handling: ~0.86
        - SMOTE Only: ~0.89
        - Class Weighting: ~0.88
        - Combined: ~0.91
        """
        print("\n" + "="*70)
        print("评估不平衡处理方法")
        print("="*70)
        
        results = []
        
        # 基础模型
        base_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        
        # 1. 无处理 (基线)
        print("\n1. 无处理 (基线)")
        start = time.time()
        base_model.fit(X_train, y_train)
        y_pred = base_model.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        train_time = time.time() - start
        results.append({'Technique': 'No Handling', 'Macro-F1': f1_macro, 'Time': train_time})
        print(f"   Macro-F1: {f1_macro:.4f}, 耗时: {train_time:.1f}s")
        
        # 2. 控制 SMOTE (30% of majority)
        print("\n2. 控制 SMOTE (30% of majority)")
        start = time.time()
        X_smote, y_smote = self.apply_smote_controlled(X_train, y_train, target_ratio=0.3)
        model_smote = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=self.random_state, tree_method='hist',
            eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1
        )
        model_smote.fit(X_smote, y_smote)
        y_pred = model_smote.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        train_time = time.time() - start
        results.append({'Technique': 'SMOTE Only', 'Macro-F1': f1_macro, 'Time': train_time})
        print(f"   Macro-F1: {f1_macro:.4f}, 耗时: {train_time:.1f}s")
        
        # 3. 类别权重
        print("\n3. 类别权重")
        start = time.time()
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([class_weights[np.where(classes == c)[0][0]] for c in y_train])
        
        model_cw = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=self.random_state, tree_method='hist',
            eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1
        )
        model_cw.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model_cw.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        train_time = time.time() - start
        results.append({'Technique': 'Class Weighting', 'Macro-F1': f1_macro, 'Time': train_time})
        print(f"   Macro-F1: {f1_macro:.4f}, 耗时: {train_time:.1f}s")
        
        # 4. 组合方法: SMOTE + 类别权重
        print("\n4. SMOTE + 类别权重")
        start = time.time()
        X_smote2, y_smote2 = self.apply_smote_controlled(X_train, y_train, target_ratio=0.3)
        classes2 = np.unique(y_smote2)
        class_weights2 = compute_class_weight('balanced', classes=classes2, y=y_smote2)
        sample_weights2 = np.array([class_weights2[np.where(classes2 == c)[0][0]] for c in y_smote2])
        
        model_combined = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=self.random_state, tree_method='hist',
            eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1
        )
        model_combined.fit(X_smote2, y_smote2, sample_weight=sample_weights2)
        y_pred = model_combined.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        train_time = time.time() - start
        results.append({'Technique': 'SMOTE + Class Weighting', 'Macro-F1': f1_macro, 'Time': train_time})
        print(f"   Macro-F1: {f1_macro:.4f}, 耗时: {train_time:.1f}s")
        
        # 显示结果
        print("\n" + "="*70)
        print("不平衡处理方法对比")
        print("="*70)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        best_method = results_df.loc[results_df['Macro-F1'].idxmax(), 'Technique']
        best_f1 = results_df['Macro-F1'].max()
        print(f"\n🏆 最佳方法: {best_method} (Macro-F1 = {best_f1:.4f})")
        
        return results_df


# =====================================================
# 5. 超参数优化类
# =====================================================

class XGBoostOptimizer:
    """XGBoost 超参数优化器"""
    
    def __init__(self, random_state=RANDOM_STATE, n_trials=30):
        self.random_state = random_state
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
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
            'random_state': self.random_state,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average='macro')
    
    def optimize(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*70)
        print(f"超参数优化 (试验次数: {self.n_trials})")
        print("="*70)
        
        start_time = time.time()
        
        study = optuna.create_study(direction='maximize')
        objective_func = lambda trial: self.objective(trial, X_train, y_train, X_val, y_val)
        study.optimize(objective_func, n_trials=self.n_trials, show_progress_bar=True)
        
        elapsed = time.time() - start_time
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print("\n优化完成!")
        print(f"最佳 Macro-F1: {self.best_score:.4f}")
        print(f"最佳超参数:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\n总优化时间: {elapsed/60:.1f} 分钟")
        
        return self.best_params, self.best_score


# =====================================================
# 6. 最终模型评估类
# =====================================================

class FinalEvaluator:
    """最终模型评估类"""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.model = None
    
    def train_final_model(self, X_train, y_train, best_params, class_names=None):
        """训练最终优化模型"""
        print("\n" + "="*70)
        print("训练最终优化模型")
        print("="*70)
        
        start_time = time.time()
        
        # 应用控制 SMOTE
        print("\n步骤 1: 应用控制 SMOTE...")
        unique, counts = np.unique(y_train, return_counts=True)
        majority_count = max(counts)
        target_count = int(majority_count * 0.3)  # 30% of majority
        
        sampling_strategy = {}
        for u, c in zip(unique, counts):
            if c < target_count:
                sampling_strategy[u] = target_count
            else:
                sampling_strategy[u] = c
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=3
        )
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        print(f"  数据量: {X_train.shape[0]:,} → {X_smote.shape[0]:,}")
        
        # 计算类别权重
        print("\n步骤 2: 计算类别权重...")
        classes = np.unique(y_smote)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_smote)
        sample_weights = np.array([class_weights[np.where(classes == c)[0][0]] for c in y_smote])
        
        # 训练模型
        print("\n步骤 3: 训练 XGBoost...")
        model = xgb.XGBClassifier(
            **best_params,
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        model.fit(X_smote, y_smote, sample_weight=sample_weights)
        
        self.model = model
        
        elapsed = time.time() - start_time
        print(f"\n✅ 训练完成，耗时: {elapsed:.1f} 秒")
        
        return model
    
    def evaluate_on_test(self, X_test, y_test, class_names=None):
        """测试集评估"""
        print("\n" + "="*70)
        print("测试集评估结果")
        print("="*70)
        
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000
        
        # 整体指标
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
        print(f"  推理时间: {inference_time:.2f} ms/样本")
        
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
        
        if class_names and len(class_names) <= 10:
            plt.figure(figsize=(12, 10))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('归一化混淆矩阵', fontsize=14)
            plt.xlabel('预测类别', fontsize=12)
            plt.ylabel('真实类别', fontsize=12)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150)
            print(f"\n✅ 混淆矩阵已保存")
            plt.show()
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'inference_time_ms': inference_time,
            'per_class_metrics': list(zip(f1_per_class, support))
        }


# =====================================================
# 7. 主实验流程
# =====================================================

def run_experiment():
    """运行完整实验 - 全量数据 + 控制 SMOTE"""
    
    print("="*70)
    print("全量数据实验（含 SVM）")
    print("="*70)
    
    # 1. 加载和预处理数据
    preprocessor = DataPreprocessor(random_state=RANDOM_STATE)
    X, y_binary, y_multiclass = preprocessor.load_data_from_folder(folder_path='unsw_data')
    
    if X is None:
        print("\n实验终止: 无法加载数据集")
        return None
    
    y = y_multiclass if y_multiclass is not None else y_binary
    
    # 2. 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # 3. 基线模型评估（包含 SVM）
    baseline = BaselineModels(random_state=RANDOM_STATE)
    baseline_results = baseline.run_all_baselines(X_train, X_val, y_train, y_val)
    
    # 4. 不平衡处理方法评估（控制 SMOTE）
    imbalance_handler = ImbalanceHandler(random_state=RANDOM_STATE)
    handling_results = imbalance_handler.evaluate_handling_methods(
        X_train, X_val, y_train, y_val, preprocessor.class_names
    )
    
    # 5. 超参数优化
    optimizer = XGBoostOptimizer(random_state=RANDOM_STATE, n_trials=30)
    best_params, best_score = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    # 6. 最终模型训练和评估
    final_evaluator = FinalEvaluator(random_state=RANDOM_STATE)
    final_model = final_evaluator.train_final_model(X_train, y_train, best_params, preprocessor.class_names)
    test_results = final_evaluator.evaluate_on_test(X_test, y_test, preprocessor.class_names)
    
    # 7. 保存模型
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print("\n保存的文件:")
    print("  - final_model.pkl")
    print("  - preprocessor.pkl")
    print("  - confusion_matrix.png")
    
    # 最终汇总
    print("\n" + "="*70)
    print("最终结果汇总")
    print("="*70)
    xgb_baseline = baseline_results[baseline_results['Model'].str.contains('XGBoost')]['Macro-F1'].values[0]
    print(f"最佳基线 (XGBoost): Macro-F1 = {xgb_baseline}")
    print(f"优化后验证集 Macro-F1: {best_score:.4f}")
    print(f"最终测试集 Macro-F1: {test_results['f1_macro']:.4f}")
    print(f"最终测试集准确率: {test_results['accuracy']*100:.2f}%")
    
    return {
        'baseline': baseline_results,
        'handling': handling_results,
        'best_params': best_params,
        'test_results': test_results
    }


# =====================================================
# 主程序入口
# =====================================================

if __name__ == "__main__":
    results = run_experiment()