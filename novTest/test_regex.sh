#!/bin/bash
# Script para probar regexes con el modelo novTest
# Uso: ./test_regex.sh "TU_REGEX_AQUI"
# Ejemplo: ./test_regex.sh "A+B"

if [ -z "$1" ]; then
    echo "Uso: ./test_regex.sh \"TU_REGEX_AQUI\""
    echo "Ejemplo: ./test_regex.sh \"A+B\""
    exit 1
fi

python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex "$1"

