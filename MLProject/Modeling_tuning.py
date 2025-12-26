import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys
import dagshub

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # --- 1. SET KONEKSI DAGSHUB (Wajib Tambah Ini) ---
    USERNAME = "juanwistasiregar"
    REPO_NAME = "Eksperimen_SML_Juan-Wistara"
    
    # Ambil token dari environment variable (GitHub Secrets)
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if token:
        dagshub.auth.add_app_token(token) # Menghindari "Authorization Required"
    
    dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")
    # ------------------------------------------------

    # 2. Load Data
    # Path otomatis mendeteksi folder tempat script berada
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_dir, "train_pca.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan di: {file_path}")
        sys.exit(1)
        
    data = pd.read_csv(file_path)

    # 3. Split Data
    X = data.drop("Credit_Score", axis=1)
    y = data["Credit_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    input_example = X_train.iloc[0:5] # Menggunakan .iloc agar lebih aman

    # 4. Ambil Parameter dari Command Line (atau default)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    # 5. Training & Logging
    with mlflow.start_run(run_name="RF_Credit_Score_Juan"):
        # Log Parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="RF_Credit_Model"
        )

        # Log Metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"✅ Training Selesai! Accuracy: {accuracy:.4f}")
