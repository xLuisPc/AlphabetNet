import pandas as pd
import re
import os
from pathlib import Path

def validate_string(s):
    """
    Valida que el string solo contenga símbolos A..L (mayúsculas) o sea <EPS>.
    Retorna True si es válido, False si no.
    """
    if s == '<EPS>':
        return True
    # Solo debe contener caracteres A-L (mayúsculas)
    return bool(re.match(r'^[A-L]+$', s))

def validate_and_fix_types(df):
    """
    Valida y corrige los tipos de datos del DataFrame.
    Verifica que no haya nulos en columnas calculadas y tipos correctos.
    
    Args:
        df: DataFrame a validar
        
    Returns:
        DataFrame con tipos corregidos
        
    Raises:
        ValueError: Si hay nulos en columnas calculadas o tipos incorrectos
    """
    print("\nValidando tipos y nulos...")
    
    # Crear copia para no modificar el original
    df_validated = df.copy()
    
    # 1. Verificar y corregir tipos de columnas
    print("Verificando tipos de columnas...")
    
    # dfa_id: debe ser int
    if df_validated['dfa_id'].dtype != 'int64':
        print(f"  - Convirtiendo dfa_id a int (tipo actual: {df_validated['dfa_id'].dtype})")
        df_validated['dfa_id'] = df_validated['dfa_id'].astype('int64')
    
    # label: debe ser int (0 o 1)
    if df_validated['label'].dtype != 'int64':
        print(f"  - Convirtiendo label a int (tipo actual: {df_validated['label'].dtype})")
        df_validated['label'] = df_validated['label'].astype('int64')
    
    # len: debe ser int
    if df_validated['len'].dtype != 'int64':
        print(f"  - Convirtiendo len a int (tipo actual: {df_validated['len'].dtype})")
        df_validated['len'] = df_validated['len'].astype('int64')
    
    # string: debe ser str
    if df_validated['string'].dtype != 'object':
        print(f"  - Convirtiendo string a str (tipo actual: {df_validated['string'].dtype})")
        df_validated['string'] = df_validated['string'].astype('str')
    
    # regex: debe ser str
    if df_validated['regex'].dtype != 'object':
        print(f"  - Convirtiendo regex a str (tipo actual: {df_validated['regex'].dtype})")
        df_validated['regex'] = df_validated['regex'].astype('str')
    
    # alphabet_decl: debe ser str
    if df_validated['alphabet_decl'].dtype != 'object':
        print(f"  - Convirtiendo alphabet_decl a str (tipo actual: {df_validated['alphabet_decl'].dtype})")
        df_validated['alphabet_decl'] = df_validated['alphabet_decl'].astype('str')
    
    # 2. Verificar nulos en columnas calculadas
    print("\nVerificando nulos en columnas calculadas...")
    calculated_columns = ['dfa_id', 'string', 'label', 'len', 'regex', 'alphabet_decl']
    
    null_counts = df_validated[calculated_columns].isnull().sum()
    has_nulls = null_counts.sum() > 0
    
    if has_nulls:
        print("  ERROR: Se encontraron nulos en columnas calculadas:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"    - {col}: {count} nulos")
        raise ValueError("Hay nulos en columnas calculadas. No se puede continuar.")
    else:
        print("  ✓ No se encontraron nulos en columnas calculadas")
    
    # 3. Validar que len sea correcto (0 para <EPS>, longitud para otros)
    print("\nValidando cálculo de len...")
    
    # Calcular len esperado de forma vectorizada
    expected_len = df_validated['string'].apply(lambda x: 0 if x == '<EPS>' else len(x))
    
    # Verificar si hay diferencias
    len_mismatch = df_validated['len'] != expected_len
    len_errors_count = len_mismatch.sum()
    
    if len_errors_count > 0:
        print(f"  ERROR: Se encontraron {len_errors_count} errores en el cálculo de len")
        
        # Mostrar algunos ejemplos
        error_rows = df_validated[len_mismatch].head(10)
        print("  Ejemplos de errores:")
        for idx, row in error_rows.iterrows():
            string = row['string']
            len_val = row['len']
            expected = 0 if string == '<EPS>' else len(string)
            print(f"    - Fila {idx}: string='{string}', len={len_val}, esperado={expected}")
        
        if len_errors_count > 10:
            print(f"  ... y {len_errors_count - 10} errores más")
        
        # Corregir los errores
        print("\n  Corrigiendo valores de len...")
        df_validated['len'] = expected_len.astype('int64')
        print("  ✓ Valores de len corregidos")
    else:
        print("  ✓ Todos los valores de len son correctos")
    
    # 4. Validar valores de label (debe ser 0 o 1)
    print("\nValidando valores de label...")
    invalid_labels = df_validated[~df_validated['label'].isin([0, 1])]
    
    if len(invalid_labels) > 0:
        print(f"  ERROR: Se encontraron {len(invalid_labels)} filas con label inválido (debe ser 0 o 1)")
        raise ValueError("Hay valores de label inválidos. No se puede continuar.")
    else:
        print("  ✓ Todos los valores de label son válidos (0 o 1)")
    
    # 5. Validar valores de len (debe ser >= 0)
    print("\nValidando valores de len (debe ser >= 0)...")
    invalid_len = df_validated[df_validated['len'] < 0]
    
    if len(invalid_len) > 0:
        print(f"  ERROR: Se encontraron {len(invalid_len)} filas con len < 0")
        raise ValueError("Hay valores de len negativos. No se puede continuar.")
    else:
        print("  ✓ Todos los valores de len son >= 0")
    
    print("\n✓ Validaciones completadas exitosamente")
    
    return df_validated

def process_csv(input_file, output_file=None, quarantine_file=None, log_file=None, flat_file=None):
    """
    Procesa el CSV: valida strings, elimina duplicados y genera archivos de salida.
    
    Args:
        input_file: Archivo CSV de entrada
        output_file: Archivo CSV limpio de salida (si None, genera nombre con _vc)
        quarantine_file: Archivo CSV de cuarentena
        log_file: Archivo CSV de log de duplicados
        flat_file: Archivo CSV final validado (data/flat.csv)
    """
    # Generar nombre de archivo de salida si no se especifica
    if output_file is None:
        input_path = Path(input_file)
        # Agregar _vc antes de la extensión
        output_file = str(input_path.parent / f"{input_path.stem}_vc{input_path.suffix}")
    
    print(f"Leyendo {input_file}...")
    df = pd.read_csv(input_file)
    
    original_count = len(df)
    print(f"Total de filas originales: {original_count}")
    
    # 1. VALIDAR STRINGS
    print("\nValidando strings...")
    df['quarantine'] = df['string'].apply(lambda x: 0 if validate_string(x) else 1)
    invalid_count = df['quarantine'].sum()
    print(f"Filas con strings inválidos: {invalid_count}")
    
    # Separar filas válidas e inválidas
    valid_df = df[df['quarantine'] == 0].copy()
    invalid_df = df[df['quarantine'] == 1].copy()
    
    print(f"Filas válidas: {len(valid_df)}")
    print(f"Filas inválidas (cuarentena): {len(invalid_df)}")
    
    # 2. DETECTAR Y ELIMINAR DUPLICADOS (solo en filas válidas)
    print("\nDetectando duplicados...")
    
    # Identificar duplicados antes de eliminar
    duplicates_mask = valid_df.duplicated(subset=['dfa_id', 'string'], keep=False)
    duplicates_df = valid_df[duplicates_mask].copy()
    
    if len(duplicates_df) > 0:
        print(f"Duplicados encontrados: {len(duplicates_df)}")
        
        # Crear log de duplicados
        log_entries = []
        
        # Agrupar por (dfa_id, string)
        grouped = duplicates_df.groupby(['dfa_id', 'string'])
        
        for (dfa_id, string), group in grouped:
            indices = group.index.tolist()
            count = len(group)
            
            # Mantener la primera ocurrencia (keep='first')
            kept_index = indices[0]
            removed_indices = indices[1:]
            
            # Obtener información de la fila mantenida
            kept_row = group.iloc[0]
            
            log_entries.append({
                'dfa_id': dfa_id,
                'string': string,
                'total_ocurrencias': count,
                'indice_mantenido': kept_index,
                'indices_eliminados': ', '.join(map(str, removed_indices)),
                'regex': kept_row['regex'],
                'label_mantenido': kept_row['label']
            })
        
        # Crear DataFrame del log
        log_df = pd.DataFrame(log_entries)
        
        # Guardar log de duplicados
        if log_file:
            # Crear directorio data/ si no existe
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\nGuardando log de duplicados en {log_file}...")
            log_df.to_csv(log_file, index=False)
            print(f"Total de grupos duplicados: {len(log_df)}")
        
        # Eliminar duplicados (mantener la primera ocurrencia)
        valid_df = valid_df.drop_duplicates(subset=['dfa_id', 'string'], keep='first')
        print(f"Filas válidas después de eliminar duplicados: {len(valid_df)}")
    else:
        print("No se encontraron duplicados.")
        # Crear log vacío
        if log_file:
            # Crear directorio data/ si no existe
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_df = pd.DataFrame(columns=['dfa_id', 'string', 'total_ocurrencias', 
                                         'indice_mantenido', 'indices_eliminados', 
                                         'regex', 'label_mantenido'])
            log_df.to_csv(log_file, index=False)
            print(f"\nLog de duplicados vacío guardado en {log_file}...")
    
    # 3. PREPARAR CSV LIMPIO (sin columna quarantine, solo filas válidas)
    print("\nPreparando CSV limpio...")
    clean_df = valid_df.drop(columns=['quarantine']).copy()
    
    # Mantener el orden original (después de eliminar duplicados)
    # El orden ya se mantiene porque drop_duplicates mantiene 'first'
    
    # 4. VALIDAR TIPOS Y NULOS
    print("\n" + "="*50)
    print("VALIDACIÓN DE TIPOS Y NULOS")
    print("="*50)
    try:
        validated_df = validate_and_fix_types(clean_df)
    except ValueError as e:
        print(f"\nERROR en validaciones: {e}")
        raise
    
    # 5. PREPARAR CSV DE CUARENTENA (solo filas inválidas, con columna quarantine)
    if len(invalid_df) > 0:
        print(f"\nPreparando CSV de cuarentena...")
        quarantine_df = invalid_df.copy()  # Ya tiene la columna quarantine=1
        
        # Guardar archivo de cuarentena
        if quarantine_file:
            # Crear directorio data/ si no existe
            quarantine_path = Path(quarantine_file)
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Guardando archivo de cuarentena en {quarantine_file}...")
            quarantine_df.to_csv(quarantine_file, index=False)
            print(f"Total de filas en cuarentena: {len(quarantine_df)}")
    else:
        print("\nNo hay filas en cuarentena. No se creará archivo de cuarentena.")
    
    # 6. GUARDAR CSV LIMPIO
    print(f"\nGuardando CSV limpio en {output_file}...")
    # Crear directorio si no existe
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(output_file, index=False)
    
    # 7. GUARDAR ARCHIVO FINAL VALIDADO (data/flat.csv)
    if flat_file:
        # Crear directorio data/ si no existe
        flat_path = Path(flat_file)
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGuardando archivo final validado en {flat_file}...")
        validated_df.to_csv(flat_file, index=False)
        print(f"✓ Archivo final validado guardado: {flat_file}")
        
        # Mostrar resumen de tipos finales
        print("\nTipos de datos finales:")
        print(validated_df.dtypes)
        print("\nVerificación final de nulos:")
        null_counts = validated_df.isnull().sum()
        if null_counts.sum() > 0:
            print("  ADVERTENCIA: Se encontraron nulos:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"    - {col}: {count} nulos")
        else:
            print("  ✓ No se encontraron nulos en ninguna columna")
    
    # Resumen
    print("\n" + "="*50)
    print("RESUMEN")
    print("="*50)
    print(f"Filas originales: {original_count}")
    print(f"Filas válidas: {len(valid_df)}")
    print(f"Filas inválidas (cuarentena): {len(invalid_df)}")
    print(f"Duplicados eliminados: {original_count - len(valid_df) - len(invalid_df)}")
    print(f"Filas en CSV limpio: {len(validated_df)}")
    print(f"Archivo CSV limpio: {output_file}")
    if len(invalid_df) > 0:
        print(f"Filas en cuarentena: {len(invalid_df)}")
    if flat_file:
        print(f"Archivo final validado: {flat_file}")
    print("="*50)
    
    return validated_df, invalid_df if len(invalid_df) > 0 else None

if __name__ == '__main__':
    # Obtener el directorio raíz del proyecto (directorio padre de scripts/)
    project_root = Path(__file__).parent.parent
    
    input_file = project_root / 'data' / 'dataset3000_flat.csv'  # Leer desde data/
    # output_file será generado automáticamente como dataset3000_flat_vc.csv
    quarantine_file = project_root / 'data' / 'dataset3000_flat_quarantine_flat.csv'
    log_file = project_root / 'data' / 'duplicates_log.csv'
    flat_file = project_root / 'data' / 'flat.csv'  # Archivo final validado
    
    process_csv(str(input_file), None, str(quarantine_file), str(log_file), str(flat_file))

