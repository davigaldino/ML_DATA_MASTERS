# Use uma imagem base do Python
FROM python:3.11

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto para o container
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão para rodar sua aplicação #atua
CMD ["python", "treinamento.py"]
