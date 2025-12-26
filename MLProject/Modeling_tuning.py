import os
import dagshub
import mlflow
import pandas as pd
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Setup DagsHub & MLflow
token = os.getenv("MLFLOW_TRACKING_PASSWORD")
USERNAME = "juanwistasiregar"
# PASTIKAN NAMA REPO DI BAWAH INI SAMA DENGAN DI DAGSHUB
REPO_NAME = "Eksperimen_SML_Juan-Wistara" 

from dagshub.auth import add_app_token
if token:
    add_app_token(token)

dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")

# 2. Memuat Dataset
# Pastikan path ini benar di struktur folder GitHub kamu
df = pd.read_csv('MLProject/churn_preprocessing.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Jalankan Eksperimen
with mlflow.start_run(run_name="RF_Tuning_Juan_Wistara"):
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
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    
    # Simpan Visualisasi
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlOrBr')
    plt.title('Confusion Matrix - Juan Wistara')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png") 

    # Simpan dan Registrasi Model ke Model Registry
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="churn_model", 
        registered_model_name="churn_model"
    )
    
print("✅ Berhasil! Model terdaftar sebagai 'churn_model' di DagsHub.")
