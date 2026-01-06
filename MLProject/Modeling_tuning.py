import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, 
                             roc_curve, f1_score, precision_score, recall_score)

# Konfigurasi MLflow
mlflow.set_experiment("Churn_Prediction_Telco")

# Folder output sementara
os.makedirs('assets', exist_ok=True)

# Path Data (Pastikan file ini ada di folder dataset Anda)
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 

def run_experiment():
    print("🚀 Memulai Proses Training untuk Telco Churn...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        return

    df = pd.read_csv(DATA_PATH)

    # --- PREPROCESSING KHUSUS TELCO CHURN ---
    # Hapus ID karena tidak berguna untuk prediksi
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Ubah TotalCharges jadi numerik (karena kadang ada spasi kosong)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']) # Hapus baris yang kosong

    # Ubah Target (Churn) jadi numerik: Yes=1, No=0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Identifikasi kolom numerik dan kategorikal
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    print(f"✅ Data Siap. Ukuran: {X.shape}")

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Preprocessing Pipeline
    # Gabungkan scaler untuk angka dan encoder untuk teks
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Gabungkan ke Pipeline Utama
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Skenario Tuning
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20],
        'clf__min_samples_leaf': [2, 4]
    }

    active_run = mlflow.active_run()
    
    with mlflow.start_run(run_name="RF_Churn_Tuning", nested=(active_run is not None)):
        print("⚙️ Sedang melakukan GridSearch (Tuning)...")
        
        # Grid Search
        grid = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        
        print(f"✅ Tuning Selesai. Best Params: {best_params}")

        # 5. Evaluasi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        metrics = {
            "accuracy": acc,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": auc
        }
        
        print(f"📊 Hasil Akhir: Accuracy={acc:.4f}, F1-Score={metrics['f1_score']:.4f}")

        # Log Params & Metrics ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Log Model
        mlflow.sklearn.log_model(best_model, "model_churn_final")
        joblib.dump(best_model, "model_churn.pkl")

        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn')
        plt.title("Confusion Matrix - Telco Churn")
        plt.ylabel("Actual Churn"); plt.xlabel("Predicted Churn")
        cm_path = os.path.join("assets", "confusion_matrix_churn.png")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)

        # 2. ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title("ROC Curve - Telco Churn")
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        roc_path = os.path.join("assets", "roc_curve_churn.png")
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(roc_path)

        print("✅ Proses Selesai! Logs telah dikirim.")

if __name__ == "__main__":
    run_experiment()
