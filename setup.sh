#!/bin/bash

echo "--- Iniciando configuración del entorno ---"

# 1. Actualizar pip
python -m pip install --upgrade pip

# 2. Instalar dependencias base
pip install -r requirements.txt

# Crear carpeta y descargar
mkdir -p data

#Verificar que las variables existan antes de empezar
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "❌ Error: Las variables KAGGLE_USERNAME o KAGGLE_KEY no están configuradas."
    echo "Asegurate de que los Secrets de Codespaces estén activos."
    exit 1
fi


echo "Descargando dataset desde Kaggle..."
kaggle competitions download -c higgs-boson -p data

# Entrar, descomprimir y limpiar
cd data
unzip -o higgs-boson.zip
unzip -o training.zip
unzip -o test.zip
unzip -o random_submission.zip

echo "Limpiando archivos temporales .zip..."
rm *.zip
cd ..

echo "--- Instalación completada con éxito ---"