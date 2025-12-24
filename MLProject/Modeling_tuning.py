import pandas as pd
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Inisialisasi koneksi ke DagsHub 
USERNAME = "juanwistasiregar" 
REPO_NAME = "Eksperimen_SML_Juan-Wistara"

dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")

# 2. Memuat Dataset hasil preprocessing
df = pd.read_csv('churn_preprocessing.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Mulai pencatatan eksperimen dengan MLflow
with mlflow.start_run(run_name="RF_Tuning_Juan_Wistara"):
    # Hyperparameters (Bagian Tuning)
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    # MANUAL LOGGING PARAMS
    mlflow.log_params(params)
    
    # Pelatihan Model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Prediksi & Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # MANUAL LOGGING
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    
    # ARTEFAK TAMBAHAN 1: Plot Confusion Matrix 
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlOrBr')
    plt.title('Confusion Matrix - Juan Wistara')
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path) 

    # ARTEFAK TAMBAHAN 2: File Model .pkl
    model_file = "model_churn.pkl"
    joblib.dump(model, model_file)
    mlflow.log_artifact(model_file) 
    
    # Simpan model ke MLflow Model Registry
    mlflow.sklearn.log_model(model, "churn_model")

print("✅ Berhasil! Silakan cek tab MLflow di DagsHub kamu.")