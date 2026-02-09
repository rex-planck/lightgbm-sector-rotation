
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config_v3
    from run_v3_pipeline import run_v3_pipeline
    
    # Verify parameters before running
    print("=== Full Training Configuration Check ===")
    print(f"DB Path: {config_v3.DB_PATH_V3}")
    print(f"GP Population: {config_v3.GP_CONFIG_V3['population_size']}")
    print(f"GP Generations: {config_v3.GP_CONFIG_V3['generations']}")
    print(f"Transformer Epochs: {config_v3.TRANSFORMER_CONFIG_V3['epochs']}")
    print("=======================================")
    
    # Run pipeline
    run_v3_pipeline()
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you are running this script from the correct directory.")
except Exception as e:
    print(f"Runtime Error: {e}")
    import traceback
    traceback.print_exc()
