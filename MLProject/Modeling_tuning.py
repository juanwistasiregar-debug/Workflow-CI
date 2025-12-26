import os
import pandas as pd
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --- 1. SETUP AUTH & DAGSHUB ---
USERNAME = "juanwistasiregar"
REPO_NAME = "Eksperimen_SML_Juan-Wistara"

# Ambil token dari environment variable GitHub Secrets
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

if token:
    # Paksa autentikasi menggunakan token agar tidak muncul prompt login
    dagshub.auth.add_app_token(token)
    print("✅ Token DagsHub berhasil dimuat.")
else:
    print("⚠️ Peringatan: MLFLOW_TRACKING_PASSWORD tidak ditemukan!")

# Inisialisasi DagsHub
dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("RF_Tuning_Juan_Wistara")

# --- 2. MEMUAT DATASET (Dynamic Path) ---
# Menggunakan path relatif agar aman dijalankan di GitHub Actions
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'churn_preprocessing.csv')

if not os.path.exists(data_path):
    # Jika script dijalankan dari root, coba cari di folder Membangun_model
    data_path = 'Membangun_model/churn_preprocessing.csv'

df = pd.read_csv(data_path)
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. EKSPERIMEN MLFLOW ---
with mlflow.start_run(run_name="RF_Tuning_Juan_Wistara"):
    # Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Log Metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    })
    
    # Artefak 1: Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlOrBr')
    plt.title('Confusion Matrix - Juan Wistara')
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    # Artefak 2: File Model .pkl
    model_file = "model_churn.pkl"
    joblib.dump(model, model_file)
    mlflow.log_artifact(model_file)
    
    # Registrasi Model ke Model Registry
    mlflow.sklearn.log_model(model, "churn_model", registered_model_name="RF_Churn_Model_Juan")

print(f"✅ Berhasil! Cek tab MLflow di DagsHub: https://dagshub.com/{USERNAME}/{REPO_NAME}")
