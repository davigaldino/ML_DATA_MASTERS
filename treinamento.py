import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import platform  # Para detectar o sistema operacional

# 📌 1. Configurar diretório do MLflow corretamente
mlflow_tracking_path = "MLflow_Logs"
os.makedirs(mlflow_tracking_path, exist_ok=True)  # Criar diretório se não existir

# ⚠️ CORREÇÃO: Ajustar o caminho da URI do MLflow
if platform.system() == "Windows":
    mlflow_tracking_uri = "file:///" + mlflow_tracking_path.replace("\\", "/")
else:
    mlflow_tracking_uri = "file://" + mlflow_tracking_path

mlflow.set_tracking_uri(mlflow_tracking_uri)

# 📌 2. Criar/Configurar o experimento
experiment_name = "Prouni_Prediction"
mlflow.set_experiment(experiment_name)  # Define o experimento (cria se não existir)

# 📌 3. Carregar os dados tratados
df = pd.read_csv("data/dataset_tratado.csv")

# 📌 4. Tratar valores ausentes, caso existam (opcional, pode ser ajustado conforme necessidade)
df.fillna(0, inplace=True)  # Preencher valores ausentes com 0, ou utilize df.dropna() para remover

# 📌 5. Verificar e converter colunas inteiras para float64, caso haja valores ausentes
df = df.astype({col: 'float64' for col in df.select_dtypes(include=['int64']).columns})

# 📌 6. Identificar a coluna correta para a variável alvo
target_column = [col for col in df.columns if "TIPO_BOLSA" in col]
if not target_column:
    raise KeyError("❌ Erro: Nenhuma coluna correspondente a 'TIPO_BOLSA' foi encontrada no dataset!")

print(f"🔍 Coluna alvo encontrada: {target_column[0]}")

# 📌 7. Definir Features (X) e Target (y)
X = df.drop(columns=target_column)
y = df[target_column[0]]  # Apenas a primeira coluna correspondente

# 📌 8. Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 9. Treinamento do Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 10. Avaliação do Modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 📌 11. Registrar o modelo no MLflow com exemplo de entrada
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    
    # Correção: Passando input_example como um DataFrame com uma linha de exemplo
    input_example = X_train.iloc[:1]  # Pega a primeira linha de X_train como exemplo
    mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

print(f'✅ Treinamento concluído! Acurácia do modelo: {accuracy:.4f}')












