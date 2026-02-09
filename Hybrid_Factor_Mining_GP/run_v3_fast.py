
import sys
import os
import pandas as pd

# Add the source directory to sys.path
sys.path.append(os.path.join(os.getcwd(), "Tushare/Hybrid_Factor_Mining_GP/src"))

# 1. Import config_v3
import config_v3

# 2. Patch config_v3
# Use existing data: stock_data.db which has data from 20240621 to 20260206
# stock_data_optimized.db is empty
config_v3.DB_PATH_V3 = os.path.join(config_v3.DATA_DIR, "stock_data.db")

# Modify GP config for speed
config_v3.GP_CONFIG_V3['population_size'] = 50
config_v3.GP_CONFIG_V3['generations'] = 2
config_v3.GP_CONFIG_V3['n_components'] = 10
config_v3.GP_CONFIG_V3['max_samples'] = 0.5

# Modify Transformer config for speed
config_v3.TRANSFORMER_CONFIG_V3['epochs'] = 2  # Just 2 epochs to demo
config_v3.TRANSFORMER_CONFIG_V3['batch_size'] = 64
config_v3.TRANSFORMER_CONFIG_V3['d_model'] = 64
config_v3.TRANSFORMER_CONFIG_V3['nhead'] = 4
config_v3.TRANSFORMER_CONFIG_V3['num_layers'] = 2
config_v3.TRANSFORMER_CONFIG_V3['dim_feedforward'] = 128

# Modify split dates to match available data (20240621 - 20260206)
# Train: 20240621 - 20250630
# Valid: 20250701 - 20250930
# Test:  20251001 - 20260206
def split_dates_fast():
    return {
        'train_end': '20250630',
        'valid_end': '20250930',
    }
config_v3.split_dates = split_dates_fast

# Disable sector neutral for now if data is missing industry info
# But let's keep it enabled and see if it works, or handle gracefully
config_v3.SECTOR_NEUTRAL_CONFIG['enabled'] = False # Disable to be safe/fast

# 3. Import run_v3_pipeline
import run_v3_pipeline

# 4. Run it
if __name__ == "__main__":
    print("Starting Fast V3 Pipeline...")
    run_v3_pipeline.run_v3_pipeline()
