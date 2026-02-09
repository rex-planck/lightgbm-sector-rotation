
import sqlite3
import os

db_path = 'Tushare/Hybrid_Factor_Mining_GP/data/stock_data_optimized.db'
print(f"Checking DB: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    if ('daily_price',) in tables:
        cursor.execute('SELECT min(trade_date), max(trade_date) FROM daily_price')
        print("Date range:", cursor.fetchall())
    else:
        print("daily_price table not found")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")

db_path_2 = 'Tushare/Hybrid_Factor_Mining_GP/data/stock_data.db'
print(f"Checking DB: {db_path_2}")
try:
    conn = sqlite3.connect(db_path_2)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    if ('daily_price',) in tables:
        cursor.execute('SELECT min(trade_date), max(trade_date) FROM daily_price')
        print("Date range:", cursor.fetchall())
    else:
        print("daily_price table not found")

    conn.close()
except Exception as e:
    print(f"Error: {e}")
