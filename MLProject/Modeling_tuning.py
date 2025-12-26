import os
import dagshub
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --- 1. Setup DagsHub & MLflow ---
USERNAME = "juanwistasiregar"
REPO_NAME = "Eksperimen_SML_Juan-Wistara"

# Inisialisasi DagsHub secara eksplisit agar koneksi tidak 404
dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True)

mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("RF_Tuning_Juan_Wistara")

# --- 2. Memuat Dataset (Dynamic Path) ---
# Mengambil lokasi folder tempat script ini berada
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'churn_preprocessing.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File tidak ketemu di: {data_path}")

df = pd.read_csv(data_path)
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Jalankan Eksperimen ---
# Menggunakan 'with' agar run otomatis tertutup jika selesai/error
with mlflow.start_run(run_name="Random Forest Baseline"):
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
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)
    
    # Simpan Visualisasi
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlOrBr')
    plt.title('Confusion Matrix - Juan Wistara')
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    # Hapus file gambar lokal setelah diupload (opsional agar bersih)
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Simpan dan Registrasi Model
    # Input example membantu MLflow mendeteksi format data kamu
    input_example = X_train.iloc[[0]] 
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="churn_model", 
        registered_model_name="churn_model_juan",
        input_example=input_example
    )
    
print(f"✅ Berhasil! Cek eksperimen kamu di: https://dagshub.com/{USERNAME}/{REPO_NAME}/experiments")
