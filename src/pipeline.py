import os
import subprocess

def run_step(step_name, script_name):
    print(f"\n{'='*55}")
    print(f"🚀 RUNNING: {step_name}")
    print(f"{'='*55}")
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    subprocess.run(["python3", script_path], check=True)

if __name__ == "__main__":
    print("\n🌟 STARTING END-TO-END CUSTOMER LTV PREDICTION PIPELINE 🌟")
    
    run_step("Phase 1: Raw Data Cleaning (Nulls/Cancellations)", "data_prep.py")
    run_step("Phase 2: RFM Feature Engineering & LTV Targeting", "feature_engineering.py")
    run_step("Phase 3: XGBoost Predictive ML Modeling", "train_model.py")
    run_step("Phase 4: MySQL Database Export", "db_loader.py")
    
    print("\n✅ PIPELINE COMPLETE. Predictive structured data ready for Tableau BI.")
