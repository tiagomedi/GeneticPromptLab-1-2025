#!/usr/bin/env python3
"""
Script para diagnosticar el problema con el corpus
"""

import pandas as pd
import sys

def debug_corpus():
    """Diagnosticar estructura del corpus"""
    print("🔍 Diagnosticando corpus...")
    
    try:
        # Cargar corpus
        df = pd.read_csv('corpus.csv')
        
        print(f"📊 Información del corpus:")
        print(f"   Filas: {len(df)}")
        print(f"   Columnas: {df.columns.tolist()}")
        print(f"   Tipos de datos: {df.dtypes.to_dict()}")
        
        print(f"\n📝 Primeras 3 filas:")
        print(df.head(3))
        
        # Buscar columnas que podrían contener texto
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                text_columns.append(col)
        
        print(f"\n📄 Columnas de texto encontradas: {text_columns}")
        
        # Sugerir columna principal
        if text_columns:
            main_col = text_columns[0]
            print(f"\n💡 Sugerencia: usar columna '{main_col}' como texto principal")
            
            # Mostrar algunos ejemplos
            print(f"\n📋 Ejemplos de la columna '{main_col}':")
            for i, text in enumerate(df[main_col].head(3)):
                print(f"   {i+1}. {str(text)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_corpus() 