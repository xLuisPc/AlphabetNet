@echo off
REM Script para probar regexes con el modelo novTest
REM Uso: test_regex.bat "TU_REGEX_AQUI"
REM Ejemplo: test_regex.bat "A+B"

if "%1"=="" (
    echo Uso: test_regex.bat "TU_REGEX_AQUI"
    echo Ejemplo: test_regex.bat "A+B"
    exit /b 1
)

python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex %1

