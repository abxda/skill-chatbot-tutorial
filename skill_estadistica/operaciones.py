"""
operaciones.py — Motor de la Skill de Analisis Estadistico.

Este modulo contiene las funciones que el chatbot puede invocar como
herramientas (tools). Cada funcion:
  1. Carga el dataset sintetico.
  2. Ejecuta una operacion estadistica.
  3. Retorna el resultado como texto legible.

Las operaciones cubren estadistica descriptiva elemental:
  - Diccionario de datos (schema + escalas de medicion)
  - Validacion de operaciones segun escala de medicion
  - Estadisticos de resumen (media, mediana, moda, desviacion estandar)
  - Tabla de frecuencias para variables categoricas
  - Deteccion de outliers con metodo IQR
  - Tabla cruzada entre dos variables categoricas
  - Filtrado y agregacion por grupos

PRINCIPIO DE DISENO:
  La skill es la AUTORIDAD sobre los datos. El modelo LLM propone
  que operacion ejecutar, pero la skill VALIDA si esa operacion
  es estadisticamente valida segun la escala de medicion de la
  variable. Esto previene alucinaciones y educa al usuario.
"""

import pandas as pd
import numpy as np
import os
import json

# ============================================================
# CARGA DE DATOS
# ============================================================

def _ruta_datos() -> str:
    """Resuelve la ruta al CSV de datos sinteticos."""
    directorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(directorio_base, "datos", "datos_sinteticos.csv")


def _cargar_datos() -> pd.DataFrame:
    """Carga el dataset y aplica tipos correctos."""
    df = pd.read_csv(_ruta_datos())
    # Asegurar tipo ordinal para nivel_educativo
    orden_educativo = ["Basica", "Media", "Superior", "Posgrado"]
    df["nivel_educativo"] = pd.Categorical(
        df["nivel_educativo"], categories=orden_educativo, ordered=True
    )
    return df


# ============================================================
# DICCIONARIO DE DATOS
# ============================================================

DICCIONARIO = {
    "id": {
        "tipo": "int",
        "escala": "identificador",
        "descripcion": "Identificador unico del registro"
    },
    "zona": {
        "tipo": "str",
        "escala": "nominal",
        "descripcion": "Zona geografica (Zona_A a Zona_J)"
    },
    "edad": {
        "tipo": "int",
        "escala": "razon",
        "descripcion": "Edad en anios (18-75), distribucion normal"
    },
    "genero": {
        "tipo": "str",
        "escala": "nominal",
        "descripcion": "Genero: Hombre, Mujer, No binario"
    },
    "nivel_educativo": {
        "tipo": "str",
        "escala": "ordinal",
        "descripcion": "Nivel educativo: Basica < Media < Superior < Posgrado"
    },
    "ingreso_mensual": {
        "tipo": "float",
        "escala": "razon",
        "descripcion": "Ingreso mensual en pesos (distribucion lognormal)"
    },
    "satisfaccion": {
        "tipo": "int",
        "escala": "intervalo",
        "descripcion": "Puntuacion de satisfaccion del 1 al 10"
    },
    "gasto_mensual": {
        "tipo": "float",
        "escala": "razon",
        "descripcion": "Gasto mensual en pesos (distribucion exponencial)"
    },
    "categoria_cliente": {
        "tipo": "str",
        "escala": "nominal",
        "descripcion": "Categoria: Nuevo, Regular, Premium"
    },
    "anios_antiguedad": {
        "tipo": "int",
        "escala": "razon",
        "descripcion": "Anios de antiguedad como cliente (0-15)"
    }
}


# ============================================================
# REGLAS DE VALIDACION POR ESCALA DE MEDICION
# ============================================================
# Basadas en la clasificacion de Stevens (1946).
#
# Cada escala HEREDA las operaciones de las escalas inferiores:
#   nominal  → solo frecuencias y moda
#   ordinal  → + mediana, minimo, maximo, rango, percentiles
#   intervalo → + media, desviacion estandar, varianza, suma
#   razon    → + todas las anteriores + razones/proporciones
#
# Esta tabla es el CORAZON de la validacion. Es lo que impide
# que el modelo (o el usuario) pida operaciones sin sentido.

OPERACIONES_POR_ESCALA = {
    "nominal": {
        "permitidas": ["frecuencia", "moda", "conteo"],
        "prohibidas_ejemplo": "No se puede calcular el promedio de 'genero' porque es una variable nominal (categorias sin orden). Solo se pueden contar frecuencias y obtener la moda (categoria mas frecuente).",
        "que_puedes_hacer": "Usa 'tabla_frecuencias' para ver la distribucion de categorias, o 'tabla_cruzada' para cruzar con otra variable categorica."
    },
    "ordinal": {
        "permitidas": ["frecuencia", "moda", "conteo", "mediana", "minimo", "maximo", "rango", "percentiles"],
        "prohibidas_ejemplo": "No se puede calcular la media de 'nivel_educativo' porque es ordinal. Sabemos que Posgrado > Superior > Media > Basica, pero la distancia entre categorias no es uniforme. La mediana SI es valida: indica el nivel educativo 'central'.",
        "que_puedes_hacer": "Usa 'tabla_frecuencias' para frecuencias, o 'agrupar_y_calcular' con operacion 'mediana', 'conteo', 'minimo' o 'maximo'."
    },
    "intervalo": {
        "permitidas": ["frecuencia", "moda", "conteo", "mediana", "minimo", "maximo", "rango", "percentiles", "media", "desviacion", "varianza", "suma"],
        "prohibidas_ejemplo": "La escala de intervalo permite casi todas las operaciones. Las diferencias son significativas (la diferencia entre satisfaccion 8 y 6 es la misma que entre 4 y 2), pero el cero no es absoluto (satisfaccion 0 no significa 'sin satisfaccion').",
        "que_puedes_hacer": "Puedes usar 'estadisticos_resumen', 'detectar_outliers', o 'agrupar_y_calcular' con cualquier operacion."
    },
    "razon": {
        "permitidas": ["frecuencia", "moda", "conteo", "mediana", "minimo", "maximo", "rango", "percentiles", "media", "desviacion", "varianza", "suma", "proporcion", "coeficiente_variacion"],
        "prohibidas_ejemplo": "La escala de razon permite TODAS las operaciones. Tiene cero absoluto (0 pesos = nada de ingreso, 0 anios = recien nacido).",
        "que_puedes_hacer": "Todas las herramientas son validas: 'estadisticos_resumen', 'detectar_outliers', 'tabla_frecuencias' (si se discretiza), 'agrupar_y_calcular'."
    },
    "identificador": {
        "permitidas": ["conteo"],
        "prohibidas_ejemplo": "La columna 'id' es un identificador unico. No tiene sentido estadistico calcular promedios, frecuencias ni ninguna otra operacion sobre ella.",
        "que_puedes_hacer": "Solo puedes contar registros. Para analisis, usa las demas columnas."
    }
}

# Mapeo de operaciones estadisticas a que escalas las permiten
# (lectura inversa: dada una operacion, que escalas la soportan)
OPERACION_REQUIERE_ESCALA = {
    "media":       ["intervalo", "razon"],
    "mediana":     ["ordinal", "intervalo", "razon"],
    "moda":        ["nominal", "ordinal", "intervalo", "razon"],
    "desviacion":  ["intervalo", "razon"],
    "varianza":    ["intervalo", "razon"],
    "suma":        ["intervalo", "razon"],
    "conteo":      ["nominal", "ordinal", "intervalo", "razon", "identificador"],
    "minimo":      ["ordinal", "intervalo", "razon"],
    "maximo":      ["ordinal", "intervalo", "razon"],
    "rango":       ["ordinal", "intervalo", "razon"],
    "percentiles": ["ordinal", "intervalo", "razon"],
    "frecuencia":  ["nominal", "ordinal", "intervalo", "razon"],
    "outliers":    ["intervalo", "razon"],
}


# ============================================================
# FUNCION DE VALIDACION
# ============================================================

def validar_operacion(columna: str, operacion: str) -> str:
    """
    Verifica si una operacion estadistica es valida para una columna
    segun su escala de medicion. Retorna un dictamen claro con:

    - Si es VALIDA: confirma que puede procederse y explica por que.
    - Si es INVALIDA: explica por que no tiene sentido estadistico
      y sugiere que operaciones SI son validas para esa variable.

    Esta herramienta es el GUARDIAN estadistico del chatbot.
    SIEMPRE debe consultarse ANTES de ejecutar una operacion
    cuando haya duda sobre su validez.

    Parametros:
        columna:   nombre de cualquier columna del dataset
        operacion: operacion que se desea realizar
                   (media, mediana, moda, desviacion, varianza, suma,
                    conteo, minimo, maximo, rango, percentiles,
                    frecuencia, outliers)
    """
    todas_las_columnas = list(DICCIONARIO.keys())

    if columna not in todas_las_columnas:
        return (
            f"VALIDACION: ERROR\n"
            f"La columna '{columna}' no existe en el dataset.\n"
            f"Columnas disponibles: {todas_las_columnas}"
        )

    todas_las_operaciones = list(OPERACION_REQUIERE_ESCALA.keys())

    if operacion not in todas_las_operaciones:
        return (
            f"VALIDACION: ERROR\n"
            f"La operacion '{operacion}' no es reconocida.\n"
            f"Operaciones disponibles: {todas_las_operaciones}"
        )

    info_columna = DICCIONARIO[columna]
    escala = info_columna["escala"]
    escalas_validas = OPERACION_REQUIERE_ESCALA[operacion]
    info_escala = OPERACIONES_POR_ESCALA.get(escala, {})

    if escala in escalas_validas:
        return (
            f"VALIDACION: PERMITIDA\n"
            f"{'=' * 50}\n"
            f"  Columna: {columna}\n"
            f"  Escala de medicion: {escala}\n"
            f"  Operacion solicitada: {operacion}\n"
            f"  Resultado: VALIDA — puedes proceder.\n"
            f"\n"
            f"  Razon: la escala '{escala}' permite la operacion '{operacion}'.\n"
            f"  Operaciones permitidas para '{escala}': {info_escala.get('permitidas', [])}"
        )
    else:
        return (
            f"VALIDACION: NO PERMITIDA\n"
            f"{'=' * 50}\n"
            f"  Columna: {columna}\n"
            f"  Escala de medicion: {escala}\n"
            f"  Operacion solicitada: {operacion}\n"
            f"  Resultado: INVALIDA — no debe realizarse.\n"
            f"\n"
            f"  Por que no: {info_escala.get('prohibidas_ejemplo', 'Operacion no compatible con esta escala.')}\n"
            f"\n"
            f"  Que puedes hacer en su lugar: {info_escala.get('que_puedes_hacer', 'Consulta el diccionario de datos.')}\n"
            f"\n"
            f"  Detalle tecnico:\n"
            f"    La operacion '{operacion}' requiere escala: {escalas_validas}\n"
            f"    La columna '{columna}' tiene escala: {escala}\n"
            f"    Operaciones validas para '{escala}': {info_escala.get('permitidas', [])}"
        )


def diccionario_de_datos() -> str:
    """
    Retorna el diccionario de datos completo del dataset.
    Incluye: columnas, tipos, escalas de medicion y descripcion.

    Tambien incluye la TABLA DE OPERACIONES VALIDAS POR ESCALA
    para que el modelo sepa que puede y que no puede hacer con
    cada variable ANTES de intentar una operacion.
    """
    lineas = ["DICCIONARIO DE DATOS", "=" * 50]
    lineas.append(f"Total de registros: 500")
    lineas.append(f"Total de columnas: {len(DICCIONARIO)}\n")

    for col, info in DICCIONARIO.items():
        lineas.append(f"  {col}")
        lineas.append(f"    Tipo: {info['tipo']} | Escala: {info['escala']}")
        lineas.append(f"    {info['descripcion']}")
        lineas.append("")

    # Incluir la tabla de reglas por escala
    lineas.append("REGLAS DE OPERACIONES POR ESCALA DE MEDICION")
    lineas.append("=" * 50)
    lineas.append("(Las escalas SUPERIORES heredan las operaciones de las INFERIORES)")
    lineas.append("")
    lineas.append("  NOMINAL (zona, genero, categoria_cliente):")
    lineas.append("    Permitido: frecuencia, moda, conteo")
    lineas.append("    NO permitido: media, mediana, desviacion, suma, outliers")
    lineas.append("")
    lineas.append("  ORDINAL (nivel_educativo):")
    lineas.append("    Permitido: todo lo nominal + mediana, minimo, maximo, rango, percentiles")
    lineas.append("    NO permitido: media, desviacion, varianza, suma")
    lineas.append("")
    lineas.append("  INTERVALO (satisfaccion):")
    lineas.append("    Permitido: todo lo ordinal + media, desviacion, varianza, suma, outliers")
    lineas.append("")
    lineas.append("  RAZON (edad, ingreso_mensual, gasto_mensual, anios_antiguedad):")
    lineas.append("    Permitido: TODAS las operaciones")
    lineas.append("")
    lineas.append("IMPORTANTE: Ante la duda, usa 'validar_operacion' ANTES de calcular.")

    return "\n".join(lineas)


# ============================================================
# ESTADISTICOS DE RESUMEN
# ============================================================

def estadisticos_resumen(columna: str) -> str:
    """
    Calcula estadisticos descriptivos para una columna.
    Adapta los estadisticos calculados segun la escala de medicion:

    - Nominal: solo moda y conteo
    - Ordinal: moda, mediana, conteo, min, max
    - Intervalo/Razon: todos los estadisticos

    Parametros:
        columna: nombre de cualquier columna del dataset
    """
    if columna not in DICCIONARIO:
        return f"Error: '{columna}' no existe. Columnas: {list(DICCIONARIO.keys())}"

    if columna == "id":
        return "La columna 'id' es un identificador. No tiene sentido estadistico analizarla."

    df = _cargar_datos()
    info = DICCIONARIO[columna]
    escala = info["escala"]

    # --- NOMINAL: solo frecuencias y moda ---
    if escala == "nominal":
        serie = df[columna]
        moda = serie.mode().iloc[0] if not serie.mode().empty else "N/A"
        conteo = int(serie.count())
        n_categorias = serie.nunique()

        lineas = [
            f"ESTADISTICOS DE RESUMEN: {columna} (escala: NOMINAL)",
            "=" * 50,
            f"  conteo: {conteo}",
            f"  categorias unicas: {n_categorias}",
            f"  moda (categoria mas frecuente): {moda}",
            f"  frecuencia de la moda: {int(serie.value_counts().iloc[0])}",
            f"",
            f"  NOTA: La media, mediana y desviacion estandar NO aplican",
            f"  para variables nominales. Las categorias no tienen orden",
            f"  ni distancia numerica entre ellas.",
            f"  Para ver todas las frecuencias usa 'tabla_frecuencias'."
        ]
        return "\n".join(lineas)

    # --- ORDINAL: + mediana, min, max ---
    if escala == "ordinal":
        serie = df[columna]
        moda = serie.mode().iloc[0] if not serie.mode().empty else "N/A"
        conteo = int(serie.count())
        n_categorias = serie.nunique()

        # Para ordinal, la mediana tiene sentido porque hay orden
        codigos = serie.cat.codes
        mediana_codigo = int(codigos.median())
        categorias_orden = serie.cat.categories.tolist()
        mediana_valor = categorias_orden[min(mediana_codigo, len(categorias_orden) - 1)]

        lineas = [
            f"ESTADISTICOS DE RESUMEN: {columna} (escala: ORDINAL)",
            "=" * 50,
            f"  conteo: {conteo}",
            f"  categorias unicas: {n_categorias}",
            f"  orden de categorias: {categorias_orden}",
            f"  moda (categoria mas frecuente): {moda}",
            f"  frecuencia de la moda: {int(serie.value_counts().iloc[0])}",
            f"  mediana (categoria central): {mediana_valor}",
            f"  minimo (categoria mas baja): {categorias_orden[0]}",
            f"  maximo (categoria mas alta): {categorias_orden[-1]}",
            f"",
            f"  NOTA: La media y desviacion estandar NO aplican para",
            f"  variables ordinales. Aunque hay orden (Basica < Media <",
            f"  Superior < Posgrado), la distancia entre categorias no",
            f"  es uniforme ni cuantificable."
        ]
        return "\n".join(lineas)

    # --- INTERVALO y RAZON: todos los estadisticos ---
    serie = df[columna].dropna()

    resultado = {
        "columna": columna,
        "escala": escala,
        "conteo": int(serie.count()),
        "media": round(float(serie.mean()), 2),
        "mediana": round(float(serie.median()), 2),
        "moda": round(float(serie.mode().iloc[0]), 2) if not serie.mode().empty else None,
        "desviacion_estandar": round(float(serie.std()), 2),
        "varianza": round(float(serie.var()), 2),
        "minimo": round(float(serie.min()), 2),
        "maximo": round(float(serie.max()), 2),
        "Q1 (percentil 25)": round(float(serie.quantile(0.25)), 2),
        "Q2 (mediana)": round(float(serie.quantile(0.50)), 2),
        "Q3 (percentil 75)": round(float(serie.quantile(0.75)), 2),
        "rango": round(float(serie.max() - serie.min()), 2),
        "rango_intercuartilico (IQR)": round(float(serie.quantile(0.75) - serie.quantile(0.25)), 2),
    }

    lineas = [f"ESTADISTICOS DE RESUMEN: {columna} (escala: {escala.upper()})", "=" * 50]
    for clave, valor in resultado.items():
        lineas.append(f"  {clave}: {valor}")

    if escala == "intervalo":
        lineas.append(f"\n  NOTA SOBRE ESCALA INTERVALO:")
        lineas.append(f"  Las diferencias son significativas (8-6 = 4-2 = 2 puntos),")
        lineas.append(f"  pero el cero no es absoluto. Satisfaccion '0' no existe en")
        lineas.append(f"  esta escala (va de 1 a 10).")

    return "\n".join(lineas)


# ============================================================
# TABLA DE FRECUENCIAS
# ============================================================

def tabla_frecuencias(columna: str) -> str:
    """
    Genera tabla de frecuencias absolutas y relativas (%) para
    cualquier variable. Es valida para TODAS las escalas, ya que
    contar ocurrencias siempre tiene sentido.

    Para variables numericas, primero discretiza en rangos.

    Parametros:
        columna: nombre de cualquier columna del dataset
                 (excepto 'id')
    """
    if columna not in DICCIONARIO or columna == "id":
        columnas_validas = [c for c in DICCIONARIO.keys() if c != "id"]
        return f"Error: '{columna}' no es valida. Opciones: {columnas_validas}"

    df = _cargar_datos()
    escala = DICCIONARIO[columna]["escala"]

    # Para categoricas: frecuencia directa
    if escala in ("nominal", "ordinal"):
        if escala == "ordinal":
            frecuencias = df[columna].value_counts().reindex(df[columna].cat.categories)
        else:
            frecuencias = df[columna].value_counts()
        porcentajes = (frecuencias / frecuencias.sum() * 100).round(1)
    else:
        # Para numericas: discretizar en 5 rangos
        serie = df[columna].dropna()
        rangos = pd.cut(serie, bins=5)
        frecuencias = rangos.value_counts().sort_index()
        porcentajes = (frecuencias / frecuencias.sum() * 100).round(1)

    tabla = pd.DataFrame({
        "frecuencia": frecuencias,
        "porcentaje": porcentajes
    })

    lineas = [f"TABLA DE FRECUENCIAS: {columna} (escala: {escala})", "=" * 50]
    for categoria, fila in tabla.iterrows():
        lineas.append(f"  {categoria}: {int(fila['frecuencia'])} ({fila['porcentaje']}%)")

    lineas.append(f"\n  Total: {int(frecuencias.sum())}")

    if escala in ("intervalo", "razon"):
        lineas.append(f"\n  NOTA: Variable numerica discretizada en 5 rangos para la tabla.")
        lineas.append(f"  Para estadisticos continuos, usa 'estadisticos_resumen'.")

    return "\n".join(lineas)


# ============================================================
# DETECCION DE OUTLIERS (IQR)
# ============================================================

def detectar_outliers(columna: str) -> str:
    """
    Detecta valores atipicos (outliers) usando el metodo del
    Rango Intercuartilico (IQR).

    Un valor es outlier si:
      valor < Q1 - 1.5 * IQR   o   valor > Q3 + 1.5 * IQR

    Solo es valido para escalas de INTERVALO y RAZON.
    Para nominales y ordinales, retorna explicacion de por que no aplica.

    Parametros:
        columna: nombre de la columna a analizar
    """
    if columna not in DICCIONARIO or columna == "id":
        columnas_validas = [c for c in DICCIONARIO.keys() if c != "id"]
        return f"Error: '{columna}' no es valida. Opciones: {columnas_validas}"

    escala = DICCIONARIO[columna]["escala"]

    if escala in ("nominal", "ordinal"):
        info_escala = OPERACIONES_POR_ESCALA[escala]
        return (
            f"DETECCION DE OUTLIERS: {columna} — NO APLICABLE\n"
            f"{'=' * 50}\n"
            f"  La columna '{columna}' tiene escala '{escala}'.\n"
            f"  La deteccion de outliers requiere escala de intervalo o razon\n"
            f"  porque necesita calcular Q1, Q3 y distancias numericas.\n"
            f"\n"
            f"  {info_escala['prohibidas_ejemplo']}\n"
            f"\n"
            f"  {info_escala['que_puedes_hacer']}"
        )

    df = _cargar_datos()
    serie = df[columna].dropna()
    Q1 = float(serie.quantile(0.25))
    Q3 = float(serie.quantile(0.75))
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]

    lineas = [f"DETECCION DE OUTLIERS (IQR): {columna} (escala: {escala})", "=" * 50]
    lineas.append(f"  Q1: {round(Q1, 2)}")
    lineas.append(f"  Q3: {round(Q3, 2)}")
    lineas.append(f"  IQR: {round(IQR, 2)}")
    lineas.append(f"  Limite inferior: {round(limite_inferior, 2)}")
    lineas.append(f"  Limite superior: {round(limite_superior, 2)}")
    lineas.append(f"  Outliers encontrados: {len(outliers)}")

    if len(outliers) > 0:
        lineas.append(f"  Valores atipicos: {sorted(outliers.values.tolist())[:20]}")
        if len(outliers) > 20:
            lineas.append(f"  (mostrando primeros 20 de {len(outliers)})")

    return "\n".join(lineas)


# ============================================================
# TABLA CRUZADA
# ============================================================

def tabla_cruzada(columna_fila: str, columna_col: str) -> str:
    """
    Genera una tabla de contingencia (tabla cruzada) entre dos
    variables categoricas. Muestra frecuencias absolutas.

    Solo valida para variables nominales u ordinales.

    Parametros:
        columna_fila: variable categorica para filas
        columna_col:  variable categorica para columnas
    """
    categoricas = [c for c, info in DICCIONARIO.items()
                   if info["escala"] in ("nominal", "ordinal")]

    for col in [columna_fila, columna_col]:
        if col not in categoricas:
            return (
                f"Error: '{col}' no es una variable categorica (nominal/ordinal).\n"
                f"Opciones: {categoricas}\n"
                f"Las tablas cruzadas requieren variables con categorias discretas."
            )

    df = _cargar_datos()
    tabla = pd.crosstab(df[columna_fila], df[columna_col])

    lineas = [f"TABLA CRUZADA: {columna_fila} x {columna_col}", "=" * 50]
    lineas.append(tabla.to_string())

    return "\n".join(lineas)


# ============================================================
# AGRUPACION Y AGREGACION
# ============================================================

def agrupar_y_calcular(columna_grupo: str, columna_valor: str, operacion: str) -> str:
    """
    Agrupa el dataset por una columna categorica y calcula una
    operacion estadistica sobre otra columna.

    VALIDA la operacion segun la escala de medicion de columna_valor:
    - No permite media/desviacion sobre nominales/ordinales
    - Si la operacion no es valida, explica por que y sugiere alternativas

    Parametros:
        columna_grupo: columna categorica para agrupar
                       (zona, genero, nivel_educativo, categoria_cliente)
        columna_valor: columna sobre la cual calcular
        operacion: operacion a realizar
                   (media, mediana, suma, conteo, minimo, maximo, desviacion)
    """
    categoricas = [c for c, info in DICCIONARIO.items()
                   if info["escala"] in ("nominal", "ordinal")]

    if columna_grupo not in categoricas:
        return f"Error: '{columna_grupo}' no es categorica. Opciones: {categoricas}"

    if columna_valor not in DICCIONARIO or columna_valor == "id":
        columnas_validas = [c for c in DICCIONARIO.keys() if c != "id"]
        return f"Error: '{columna_valor}' no es valida. Opciones: {columnas_validas}"

    ops = {
        "media": "mean",
        "mediana": "median",
        "suma": "sum",
        "conteo": "count",
        "minimo": "min",
        "maximo": "max",
        "desviacion": "std"
    }

    if operacion not in ops:
        return f"Error: operacion '{operacion}' no valida. Opciones: {list(ops.keys())}"

    # --- VALIDACION POR ESCALA ---
    escala_valor = DICCIONARIO[columna_valor]["escala"]

    if operacion in OPERACION_REQUIERE_ESCALA:
        escalas_permitidas = OPERACION_REQUIERE_ESCALA[operacion]
        if escala_valor not in escalas_permitidas:
            info_escala = OPERACIONES_POR_ESCALA.get(escala_valor, {})
            ops_validas = [op for op, escalas in OPERACION_REQUIERE_ESCALA.items()
                          if escala_valor in escalas and op in ops]
            return (
                f"OPERACION NO VALIDA\n"
                f"{'=' * 50}\n"
                f"  No se puede calcular '{operacion}' de '{columna_valor}'.\n"
                f"  La columna '{columna_valor}' tiene escala '{escala_valor}'.\n"
                f"  La operacion '{operacion}' requiere escala: {escalas_permitidas}.\n"
                f"\n"
                f"  {info_escala.get('prohibidas_ejemplo', '')}\n"
                f"\n"
                f"  Operaciones validas para '{columna_valor}' (escala {escala_valor}):\n"
                f"    {ops_validas}\n"
                f"\n"
                f"  {info_escala.get('que_puedes_hacer', '')}"
            )

    df = _cargar_datos()

    # Para ordinales, convertir a codigos numericos para poder calcular mediana/min/max
    if escala_valor == "ordinal" and operacion in ("mediana", "minimo", "maximo"):
        serie_codigos = df[columna_valor].cat.codes
        grupo = df.groupby(columna_grupo).apply(
            lambda g: df[columna_valor].cat.categories[
                int(getattr(g[columna_valor].cat.codes, ops[operacion])())
            ],
            include_groups=False
        )
        lineas = [
            f"AGRUPACION: {operacion} de {columna_valor} por {columna_grupo} (escala: ordinal)",
            "=" * 50
        ]
        for nombre, valor in grupo.items():
            lineas.append(f"  {nombre}: {valor}")
        return "\n".join(lineas)

    # Para categoricas con conteo
    if escala_valor in ("nominal", "ordinal") and operacion == "conteo":
        grupo = df.groupby(columna_grupo)[columna_valor].count()
    else:
        grupo = df.groupby(columna_grupo)[columna_valor].agg(ops[operacion])

    lineas = [
        f"AGRUPACION: {operacion} de {columna_valor} por {columna_grupo} (escala: {escala_valor})",
        "=" * 50
    ]
    for nombre, valor in grupo.items():
        lineas.append(f"  {nombre}: {round(float(valor), 2)}")

    return "\n".join(lineas)


# ============================================================
# REGISTRO DE HERRAMIENTAS (para el chatbot)
# ============================================================

# Mapa nombre -> funcion (usado por el chatbot para despachar llamadas)
HERRAMIENTAS = {
    "diccionario_de_datos": diccionario_de_datos,
    "validar_operacion": validar_operacion,
    "estadisticos_resumen": estadisticos_resumen,
    "tabla_frecuencias": tabla_frecuencias,
    "detectar_outliers": detectar_outliers,
    "tabla_cruzada": tabla_cruzada,
    "agrupar_y_calcular": agrupar_y_calcular,
}


def ejecutar(nombre_herramienta: str, argumentos: dict) -> str:
    """
    Punto de entrada unico para ejecutar cualquier herramienta.
    El chatbot llama a esta funcion con el nombre y los argumentos
    que el modelo decidio usar.
    """
    if nombre_herramienta not in HERRAMIENTAS:
        return f"Error: herramienta '{nombre_herramienta}' no existe. Disponibles: {list(HERRAMIENTAS.keys())}"

    funcion = HERRAMIENTAS[nombre_herramienta]

    try:
        return funcion(**argumentos)
    except TypeError as e:
        return f"Error en argumentos de '{nombre_herramienta}': {e}"
    except Exception as e:
        return f"Error al ejecutar '{nombre_herramienta}': {e}"
