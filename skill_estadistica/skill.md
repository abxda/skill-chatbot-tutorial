# Skill: Analisis Estadistico Basico con Validacion por Escala

## Descripcion

Esta skill permite analizar un dataset de 500 registros de clientes/residentes
mediante operaciones estadisticas elementales. Incorpora un sistema de
**validacion por escala de medicion** (Stevens, 1946) que impide operaciones
estadisticamente invalidas y educa al usuario sobre por que no tienen sentido.

## Capacidades

1. **Diccionario de datos** — Estructura completa + reglas por escala
2. **Validar operacion** — Guardian que verifica validez estadistica
3. **Estadisticos de resumen** — Adaptativos segun escala (nominal/ordinal/intervalo/razon)
4. **Tabla de frecuencias** — Para cualquier variable (discretiza numericas)
5. **Deteccion de outliers** — Metodo IQR (solo intervalo/razon, rechaza con explicacion)
6. **Tabla cruzada** — Contingencia entre dos categoricas
7. **Agrupacion y calculo** — Agregar por grupo con validacion integrada

## Reglas de validacion por escala

| Escala | Variables | Operaciones validas |
|--------|-----------|-------------------|
| Nominal | zona, genero, categoria_cliente | frecuencia, moda, conteo |
| Ordinal | nivel_educativo | + mediana, min, max, rango |
| Intervalo | satisfaccion | + media, desviacion, varianza, suma, outliers |
| Razon | edad, ingreso_mensual, gasto_mensual, anios_antiguedad | TODAS |

## Principio anti-alucinacion

La skill es la AUTORIDAD sobre los datos. El modelo propone operaciones,
la skill valida si son estadisticamente correctas. Si no lo son, retorna
una explicacion pedagogica con la razon y sugiere alternativas validas.
El modelo NUNCA calcula por si mismo: siempre delega en la skill.
