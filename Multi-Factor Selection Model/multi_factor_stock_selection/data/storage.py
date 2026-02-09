"""
数据存储模块 - 管理数据的保存和加载
"""

import pandas as pd
import os
import pickle
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.tushare_config import DATA_DIR, RESULTS_DIR


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, data_dir=None, results_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.results_dir = results_dir or RESULTS_DIR
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_dataframe(self, df, filename, subdir=""):
        """
        保存 DataFrame 到 CSV
        
        Parameters:
        -----------
        df : DataFrame
            数据
        filename : str
            文件名
        subdir : str
            子目录
        """
        if df is None or df.empty:
            print(f"⚠️ 数据为空，跳过保存 {filename}")
            return
        
        filepath = os.path.join(self.data_dir, subdir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"💾 数据已保存: {filepath} ({len(df)} 条记录)")
        return filepath
    
    def load_dataframe(self, filename, subdir=""):
        """
        从 CSV 加载 DataFrame
        
        Parameters:
        -----------
        filename : str
            文件名
        subdir : str
            子目录
            
        Returns:
        --------
        DataFrame : 数据
        """
        filepath = os.path.join(self.data_dir, subdir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ 文件不存在: {filepath}")
            return None
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"📂 数据已加载: {filepath} ({len(df)} 条记录)")
        return df
    
    def save_pickle(self, data, filename):
        """使用 pickle 保存数据"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 数据已保存 (pickle): {filepath}")
        return filepath
    
    def load_pickle(self, filename):
        """使用 pickle 加载数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 数据已加载 (pickle): {filepath}")
        return data
    
    def check_data_exists(self, filename, subdir=""):
        """检查数据文件是否存在"""
        filepath = os.path.join(self.data_dir, subdir, filename)
        return os.path.exists(filepath)


if __name__ == "__main__":
    storage = DataStorage()
    print(f"数据目录: {storage.data_dir}")
    print(f"结果目录: {storage.results_dir}")
