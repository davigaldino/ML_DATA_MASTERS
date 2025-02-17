import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# 📌 1. Carregar os dados
file_path = r'D:\OneDrive\SANTANDER\ML Data Master\prouni-2019-2015.xlsm'
df = pd.read_excel(file_path, sheet_name="prouni-2019-2015")

# 📌 2. Criar a coluna de IDADE (caso ainda não esteja no dataset)
df['IDADE'] = 2025 - pd.to_datetime(df['DT_NASCIMENTO_BENEFICIARIO']).dt.year

# 📌 3. Remover colunas que não podem ser usadas diretamente
df.drop(columns=['DT_NASCIMENTO_BENEFICIARIO'], inplace=True)

# 📌 4. Separar colunas categóricas
categorical_columns = ['MODALIDADE_ENSINO_BOLSA', 'SEXO_BENEFICIARIO_BOLSA', 
                       'RACA_BENEFICIARIO_BOLSA', 'BENEFICIARIO_DEFICIENTE_FISICO', 
                       'REGIAO_BENEFICIARIO_BOLSA', 'SIGLA_UF_BENEFICIARIO_BOLSA', 
                       'NOME_IES_BOLSA', 'NOME_CURSO_BOLSA', 'NOME_TURNO_CURSO_BOLSA', 'TIPO_BOLSA']

# 📌 5. Aplicar Label Encoding para colunas com muitas categorias
le = LabelEncoder()
for col in categorical_columns:
    if df[col].nunique() > 10:  # Se houver muitas categorias, usa Label Encoding
        df[col] = le.fit_transform(df[col])
    else:  # Se houver poucas categorias, usa One-Hot Encoding
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# 📌 6. Preencher valores ausentes com a moda (valor mais frequente)
df.fillna(df.mode().iloc[0], inplace=True)

# 📌 7. Garantir que o diretório de saída existe
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
#df.to_csv("data/dataset_tratado.csv", index=False)

# 📌 8. Salvar o dataset tratado
df.to_csv(os.path.join(output_dir, "dataset_tratado.csv"), index=False)

print("✅ Pré-processamento concluído! Arquivo salvo.")

