#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "--- Iniciando configuración del entorno ---"

if [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Error: no se encontró python ni python3 en PATH."
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "❌ Error: falta el comando unzip."
    exit 1
fi

# 1. Actualizar pip
"$PYTHON_BIN" -m pip install --upgrade pip

# 2. Instalar dependencias base
"$PYTHON_BIN" -m pip install -r requirements.txt

PYTHON_DIR="$(dirname "$PYTHON_BIN")"

if [ -x "$PYTHON_DIR/kaggle" ]; then
    KAGGLE_BIN="$PYTHON_DIR/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
    KAGGLE_BIN="$(command -v kaggle)"
else
    echo "❌ Error: no se encontró el ejecutable de Kaggle."
    echo "Verificá que la dependencia 'kaggle' se haya instalado correctamente."
    exit 1
fi

# Crear carpeta y descargar
mkdir -p data

# Kaggle CLI usa KAGGLE_KEY; aceptamos también KAGGLE_API_TOKEN para no romper el README actual.
if [ -n "${KAGGLE_API_TOKEN:-}" ] && [ -z "${KAGGLE_KEY:-}" ]; then
    export KAGGLE_KEY="$KAGGLE_API_TOKEN"
fi

if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    echo "❌ Error: faltan KAGGLE_USERNAME y/o KAGGLE_KEY."
    echo "También podés usar KAGGLE_API_TOKEN en lugar de KAGGLE_KEY."
    echo "Asegurate de que los Secrets de Codespaces estén activos."
    exit 1
fi

echo "Descargando dataset desde Kaggle..."
if ! DOWNLOAD_OUTPUT="$("$KAGGLE_BIN" competitions download -c higgs-boson -p data 2>&1)"; then
    echo "$DOWNLOAD_OUTPUT"

    if echo "$DOWNLOAD_OUTPUT" | grep -qi "403"; then
        echo "❌ Error: Kaggle rechazó el acceso al dataset."
        echo "Entrá con tu cuenta a https://www.kaggle.com/c/higgs-boson/rules y aceptá las reglas de la competencia."
        echo "Si ya las aceptaste, verificá que la API key pertenezca a esa misma cuenta."
    fi

    exit 1
fi

echo "$DOWNLOAD_OUTPUT"

if [ ! -f data/higgs-boson.zip ]; then
    echo "❌ Error: Kaggle no descargó data/higgs-boson.zip."
    echo "Verificá tus credenciales y que hayas aceptado las reglas de la competencia en Kaggle."
    exit 1
fi

# Entrar, descomprimir y limpiar
cd data
unzip -o higgs-boson.zip

for zip_file in training.zip test.zip random_submission.zip; do
    if [ -f "$zip_file" ]; then
        unzip -o "$zip_file"
    fi
done

echo "Limpiando archivos temporales .zip..."
rm -f ./*.zip
cd ..

echo "--- Instalación completada con éxito ---"
