"""
generar_datos.py — Generador de datos sinteticos para el tutorial de Skills + LLM.

Este script genera un dataset sintetico de 500 registros que simula
una base de clientes/residentes con variables de todos los tipos
estadisticos (nominal, ordinal, intervalo, razon).

El dataset se genera con semilla fija (reproducible) y se guarda
como CSV limpio, listo para ser consumido por la skill de analisis.

Ejecutar:
    python datos/generar_datos.py
"""

import pandas as pd
import numpy as np
import os

def generar_dataset(n: int = 500, semilla: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintetico limpio con variables de todos los tipos.

    Columnas generadas:
    - id:                   Identificador unico (int)
    - zona:                 Zona geografica simulada (nominal, 10 zonas)
    - edad:                 Edad en anios (razon, distribucion normal)
    - genero:               Genero (nominal: Hombre, Mujer, No binario)
    - nivel_educativo:      Nivel educativo (ordinal: Basica, Media, Superior, Posgrado)
    - ingreso_mensual:      Ingreso en pesos (razon, distribucion lognormal)
    - satisfaccion:         Puntuacion de satisfaccion 1-10 (intervalo)
    - gasto_mensual:        Gasto mensual en pesos (razon, distribucion exponencial)
    - categoria_cliente:    Tipo de cliente (nominal: Nuevo, Regular, Premium)
    - anios_antiguedad:     Anios como cliente (razon)
    """
    np.random.seed(semilla)

    zonas = [f"Zona_{chr(65 + i)}" for i in range(10)]  # Zona_A ... Zona_J

    df = pd.DataFrame({
        "id": range(1, n + 1),

        # --- Variable nominal: zona geografica ---
        "zona": np.random.choice(zonas, n),

        # --- Variable de razon: edad (normal, truncada 18-75) ---
        "edad": np.random.normal(38, 12, n).astype(int).clip(18, 75),

        # --- Variable nominal: genero ---
        "genero": np.random.choice(
            ["Hombre", "Mujer", "No binario"], n, p=[0.47, 0.47, 0.06]
        ),

        # --- Variable ordinal: nivel educativo ---
        "nivel_educativo": np.random.choice(
            ["Basica", "Media", "Superior", "Posgrado"], n, p=[0.15, 0.35, 0.35, 0.15]
        ),

        # --- Variable de razon: ingreso mensual (lognormal) ---
        "ingreso_mensual": np.round(np.random.lognormal(9.8, 0.5, n), 2),

        # --- Variable de intervalo: satisfaccion 1-10 ---
        "satisfaccion": np.random.randint(1, 11, n),

        # --- Variable de razon: gasto mensual (exponencial) ---
        "gasto_mensual": np.round(np.random.exponential(3500, n), 2),

        # --- Variable nominal: categoria de cliente ---
        "categoria_cliente": np.random.choice(
            ["Nuevo", "Regular", "Premium"], n, p=[0.30, 0.50, 0.20]
        ),

        # --- Variable de razon: antiguedad como cliente ---
        "anios_antiguedad": np.random.randint(0, 16, n),
    })

    return df


def main():
    df = generar_dataset()

    # Guardar en el mismo directorio que este script
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_sinteticos.csv")
    df.to_csv(ruta, index=False)
    print(f"Dataset generado: {ruta}")
    print(f"Registros: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeras 5 filas:\n{df.head()}")
    print(f"\nResumen estadistico:\n{df.describe()}")


if __name__ == "__main__":
    main()
