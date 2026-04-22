# =====================================================
# 步骤1: 数据加载和预处理
# 使用官方分好的训练集和测试集
# =====================================================

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SMOTE_RATIO = 0.3
OPTUNA_TRIALS = 30

# 创建保存文件夹
os.makedirs('saved_data', exist_ok=True)

print("="*70)
print("步骤1: 数据加载和预处理")
print("使用官方训练集和测试集")
print("="*70)


# =====================================================
# 数据加载函数
# =====================================================

def load_official_split(folder_path='unsw_data'):
    """加载官方分好的训练集和测试集"""
    
    print(f"\n正在从文件夹 '{folder_path}' 加载数据...")
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return None, None
    
    train_file = os.path.join(folder_path, 'UNSW_NB15_training-set.csv')
    test_file = os.path.join(folder_path, 'UNSW_NB15_testing-set.csv')
    
    if not os.path.exists(train_file):
        print(f"错误: 训练集文件 '{train_file}' 不存在!")
        return None, None
    if not os.path.exists(test_file):
        print(f"错误: 测试集文件 '{test_file}' 不存在!")
        return None, None
    
    # 加载训练集
    train_size = os.path.getsize(train_file) / (1024 * 1024)
    print(f"\n加载训练集: UNSW_NB15_training-set.csv ({train_size:.1f} MB)")
    train_df = pd.read_csv(train_file, low_memory=False)
    print(f"  ✅ {len(train_df):,} 条记录")
    
    # 加载测试集
    test_size = os.path.getsize(test_file) / (1024 * 1024)
    print(f"\n加载测试集: UNSW_NB15_testing-set.csv ({test_size:.1f} MB)")
    test_df = pd.read_csv(test_file, low_memory=False)
    print(f"  ✅ {len(test_df):,} 条记录")
    
    print(f"\n{'='*50}")
    print(f"数据加载完成!")
    print(f"训练集: {len(train_df):,} 条")
    print(f"测试集: {len(test_df):,} 条")
    print(f"总计: {len(train_df) + len(test_df):,} 条")
    print(f"{'='*50}")
    
    return train_df, test_df


# =====================================================
# 数据预处理类
# =====================================================

class DataPreprocessor:
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.class_names = None
        
    def load_and_preprocess(self, folder_path='unsw_data', val_size=0.15):
        """加载官方训练集和测试集，并从训练集中分出验证集"""
        train_df, test_df = load_official_split(folder_path)
        if train_df is None:
            return None
        
        print("\n" + "="*60)
        print("预处理训练集")
        print("="*60)
        X_train_full, y_train_full = self.preprocess(train_df, fit_scaler=True)
        
        print("\n" + "="*60)
        print("预处理测试集")
        print("="*60)
        X_test, y_test = self.preprocess(test_df, fit_scaler=False)
        
        # 从训练集中分出验证集
        print("\n" + "="*60)
        print(f"从训练集分出验证集 ({val_size*100:.0f}%)")
        print("="*60)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=val_size, 
            random_state=self.random_state,
            stratify=y_train_full
        )
        
        print(f"\n训练集: {X_train.shape[0]:,} 样本")
        print(f"验证集: {X_val.shape[0]:,} 样本")
        print(f"测试集: {X_test.shape[0]:,} 样本")
        
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n训练集类别分布:")
        for u, c in zip(unique, counts):
            class_name = self.class_names[u] if self.class_names and u < len(self.class_names) else str(u)
            print(f"  {class_name}: {c:,} ({c/len(y_train)*100:.2f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(self, df, fit_scaler=True):
        print("\n开始数据预处理...")
        
        df_processed = df.copy()
        
        # 统一列名为小写
        df_processed.columns = df_processed.columns.str.strip().str.lower()
        
        # 删除不必要的列
        columns_to_drop = ['srcip', 'dstip', 'sport', 'dsport', 'stime', 'ltime',
                          'source ip', 'destination ip', 'start time', 'last time',
                          'no.', 'id']
        for col in columns_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
        
        # 处理缺失值
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['float64', 'int64']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    mode_val = df_processed[col].mode()
                    df_processed[col] = df_processed[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        
        # 编码分类特征
        categorical_features = ['proto', 'state', 'service']
        
        for col in categorical_features:
            if col in df_processed.columns:
                if fit_scaler:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df_processed[col] = df_processed[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else 'Unknown'
                    )
                    if 'Unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'Unknown')
                    df_processed[col] = le.transform(df_processed[col])
        
        # 处理攻击类别标签
        if 'attack_cat' in df_processed.columns:
            if fit_scaler:
                le_attack = LabelEncoder()
                df_processed['attack_cat_encoded'] = le_attack.fit_transform(df_processed['attack_cat'].astype(str))
                self.label_encoders['attack_cat'] = le_attack
                self.class_names = le_attack.classes_.tolist()
                print(f"  攻击类别: {len(self.class_names)} 种")
            else:
                le_attack = self.label_encoders['attack_cat']
                df_processed['attack_cat'] = df_processed['attack_cat'].astype(str).apply(
                    lambda x: x if x in le_attack.classes_ else 'Unknown'
                )
                df_processed['attack_cat_encoded'] = le_attack.transform(df_processed['attack_cat'])
        
        # 分离特征和标签
        exclude_cols = ['label', 'attack_cat', 'attack_cat_encoded']
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        X = df_processed[feature_cols].select_dtypes(include=[np.number])
        
        y = df_processed['attack_cat_encoded'].values if 'attack_cat_encoded' in df_processed.columns else None
        
        print(f"特征矩阵形状: {X.shape}")
        
        # 特征缩放
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        
        print("预处理完成!")
        
        return X_scaled, y


# =====================================================
# 主程序
# =====================================================

def main():
    # 加载和预处理
    preprocessor = DataPreprocessor()
    data = preprocessor.load_and_preprocess(folder_path='unsw_data', val_size=0.15)
    
    if data is None:
        print("数据加载失败!")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # 保存数据
    print("\n" + "="*60)
    print("保存预处理后的数据")
    print("="*60)
    
    np.save('saved_data/X_train.npy', X_train)
    np.save('saved_data/X_val.npy', X_val)
    np.save('saved_data/X_test.npy', X_test)
    np.save('saved_data/y_train.npy', y_train)
    np.save('saved_data/y_val.npy', y_val)
    np.save('saved_data/y_test.npy', y_test)
    
    with open('saved_data/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    with open('saved_data/class_names.pkl', 'wb') as f:
        pickle.dump(preprocessor.class_names, f)
    
    params = {'SMOTE_RATIO': SMOTE_RATIO, 'OPTUNA_TRIALS': OPTUNA_TRIALS, 'RANDOM_STATE': RANDOM_STATE}
    with open('saved_data/params.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    print("✅ 已保存到 saved_data/ 文件夹:")
    print("   - X_train.npy, X_val.npy, X_test.npy")
    print("   - y_train.npy, y_val.npy, y_test.npy")
    print("   - preprocessor.pkl")
    print("   - class_names.pkl")
    print("   - params.pkl")
    
    print("\n" + "="*70)
    print("步骤1完成! 可以运行 step2_baseline.py")
    print("="*70)


if __name__ == "__main__":
    main()