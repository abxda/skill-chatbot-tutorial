"""
chatbot.py — Chatbot con capacidad de ejecutar una Skill de Analisis Estadistico.

Este chatbot se conecta a Groq (API compatible con OpenAI) y usa
tool calling (llamada a herramientas) para ejecutar operaciones
estadisticas sobre un dataset sintetico.

Arquitectura:
    Usuario -> Chatbot -> Modelo LLM (Groq) -> Tool Call -> Skill -> Resultado -> LLM -> Respuesta

La skill incluye un sistema de VALIDACION POR ESCALA DE MEDICION
que impide que el modelo realice operaciones estadisticamente
invalidas y educa al usuario sobre por que no tienen sentido.

Uso:
    python chatbot.py
"""

import os
import json
from openai import OpenAI

# Importamos el motor de la skill
from skill_estadistica.operaciones import ejecutar, DICCIONARIO

# ============================================================
# CONFIGURACION
# ============================================================

def cargar_api_key() -> str:
    """Carga la API key de Groq desde variable de entorno o .env"""
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key

    # Intentar leer desde .env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for linea in f:
                linea = linea.strip()
                if linea.startswith("GROQ_API_KEY="):
                    return linea.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: No se encontro GROQ_API_KEY.")
    print("Opciones:")
    print("  1. Crear archivo .env con: GROQ_API_KEY=tu_clave_aqui")
    print("  2. Exportar variable: export GROQ_API_KEY=tu_clave_aqui")
    exit(1)


# Cliente OpenAI apuntando a Groq
cliente = OpenAI(
    api_key=cargar_api_key(),
    base_url="https://api.groq.com/openai/v1"
)

MODELO = "openai/gpt-oss-120b"

# ============================================================
# DEFINICION DE HERRAMIENTAS (TOOLS)
# ============================================================
# Estas definiciones le dicen al modelo QUE herramientas existen,
# que parametros aceptan y para que sirven.
# El modelo NO ejecuta codigo: solo decide CUAL herramienta llamar
# y con que argumentos. Nosotros ejecutamos la herramienta.
#
# NOTA DE DISENO: Las herramientas NO usan "enum" restrictivo
# en las columnas. Esto es intencional: permitimos que el modelo
# envie CUALQUIER columna, y la SKILL valida si la operacion es
# estadisticamente valida segun la escala de medicion.
# Asi, el modelo puede aprender del feedback de la skill.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "diccionario_de_datos",
            "description": "Consulta la estructura completa del dataset: columnas, tipos, escalas de medicion, descripcion de cada variable, y la TABLA DE REGLAS que indica que operaciones son validas para cada escala. SIEMPRE usar esto al inicio de la conversacion.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validar_operacion",
            "description": "GUARDIAN ESTADISTICO. Verifica si una operacion es valida para una columna segun su escala de medicion (nominal, ordinal, intervalo, razon). Retorna PERMITIDA o NO PERMITIDA con explicacion detallada y sugerencias. SIEMPRE llamar esta herramienta ANTES de ejecutar una operacion cuando haya cualquier duda sobre su validez estadistica. Esto previene errores conceptuales y educa al usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columna": {
                        "type": "string",
                        "description": "Nombre de la columna del dataset a validar"
                    },
                    "operacion": {
                        "type": "string",
                        "description": "Operacion estadistica a verificar. Opciones: media, mediana, moda, desviacion, varianza, suma, conteo, minimo, maximo, rango, percentiles, frecuencia, outliers"
                    }
                },
                "required": ["columna", "operacion"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estadisticos_resumen",
            "description": "Calcula estadisticos descriptivos de una columna. Se ADAPTA automaticamente segun la escala de medicion: para nominales solo muestra moda y conteo, para ordinales agrega mediana, para intervalo/razon muestra todos (media, mediana, moda, desviacion, cuartiles, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columna": {
                        "type": "string",
                        "description": "Nombre de la columna a analizar. Puede ser cualquier columna del dataset."
                    }
                },
                "required": ["columna"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tabla_frecuencias",
            "description": "Genera tabla de frecuencias absolutas y relativas (porcentaje). Valida para TODAS las escalas: para categoricas muestra cada categoria, para numericas discretiza en rangos automaticamente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columna": {
                        "type": "string",
                        "description": "Nombre de cualquier columna del dataset (excepto id)"
                    }
                },
                "required": ["columna"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detectar_outliers",
            "description": "Detecta valores atipicos (outliers) usando el metodo IQR. Solo valido para escalas de INTERVALO y RAZON. Si se intenta con nominal/ordinal, retorna explicacion educativa de por que no aplica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columna": {
                        "type": "string",
                        "description": "Nombre de la columna a analizar"
                    }
                },
                "required": ["columna"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tabla_cruzada",
            "description": "Genera tabla de contingencia (tabla cruzada) entre dos variables categoricas (nominales u ordinales). Muestra frecuencias absolutas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columna_fila": {
                        "type": "string",
                        "description": "Variable categorica (nominal/ordinal) para las filas"
                    },
                    "columna_col": {
                        "type": "string",
                        "description": "Variable categorica (nominal/ordinal) para las columnas"
                    }
                },
                "required": ["columna_fila", "columna_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agrupar_y_calcular",
            "description": "Agrupa el dataset por una variable categorica y calcula una operacion sobre otra columna. VALIDA internamente que la operacion sea compatible con la escala de medicion de la columna de valor. Si no es valida, explica por que y sugiere alternativas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columna_grupo": {
                        "type": "string",
                        "description": "Variable categorica para agrupar (nominal u ordinal)"
                    },
                    "columna_valor": {
                        "type": "string",
                        "description": "Variable sobre la cual calcular la operacion"
                    },
                    "operacion": {
                        "type": "string",
                        "description": "Operacion a realizar: media, mediana, suma, conteo, minimo, maximo, desviacion",
                        "enum": ["media", "mediana", "suma", "conteo", "minimo", "maximo", "desviacion"]
                    }
                },
                "required": ["columna_grupo", "columna_valor", "operacion"]
            }
        }
    }
]

# ============================================================
# MENSAJE DE SISTEMA (System Prompt)
# ============================================================
# Le dice al modelo QUIEN es, COMO debe comportarse, y las
# REGLAS ESTADISTICAS que debe respetar.

SYSTEM_PROMPT = """Eres un asistente de analisis estadistico riguroso y educativo.
Tu trabajo es responder preguntas sobre un dataset de 500 registros de clientes
usando las herramientas disponibles.

REGLAS CRITICAS:

1. SIEMPRE usa 'diccionario_de_datos' al inicio de la conversacion para
   conocer la estructura del dataset y las REGLAS POR ESCALA DE MEDICION.

2. ANTES de ejecutar cualquier operacion estadistica, verifica mentalmente
   si es valida para la escala de medicion de la variable. Si tienes DUDA,
   usa 'validar_operacion' para confirmar.

3. NO inventes datos. Solo responde con informacion de las herramientas.

4. Si el usuario pide una operacion que NO es valida para la escala de la
   variable (ej: "promedio de genero"), NO la ejecutes. En su lugar:
   a) Explica POR QUE no tiene sentido estadistico.
   b) Explica QUE escala tiene la variable y que implica.
   c) Sugiere QUE operaciones SI son validas.

5. Explica los resultados de forma clara. Cuando reportes un estadistico,
   menciona la escala de medicion para que el usuario entienda el contexto.

RESUMEN DE ESCALAS DE MEDICION (de menor a mayor capacidad):

  NOMINAL (zona, genero, categoria_cliente):
    → Solo: frecuencia, moda, conteo
    → NO: media, mediana, desviacion, outliers, suma

  ORDINAL (nivel_educativo):
    → Lo nominal + mediana, min, max, rango, percentiles
    → NO: media, desviacion, varianza, suma

  INTERVALO (satisfaccion):
    → Lo ordinal + media, desviacion, varianza, suma, outliers

  RAZON (edad, ingreso_mensual, gasto_mensual, anios_antiguedad):
    → TODAS las operaciones son validas

HERRAMIENTAS DISPONIBLES:
- diccionario_de_datos: estructura y reglas del dataset
- validar_operacion: verificar si operacion + columna es valida (GUARDIAN)
- estadisticos_resumen: descriptivos adaptados a la escala
- tabla_frecuencias: frecuencias para cualquier variable
- detectar_outliers: valores atipicos (solo intervalo/razon)
- tabla_cruzada: contingencia entre dos categoricas
- agrupar_y_calcular: estadisticos por grupo con validacion
"""

# ============================================================
# BUCLE PRINCIPAL DEL CHATBOT
# ============================================================

def procesar_tool_calls(respuesta, mensajes):
    """
    Procesa las llamadas a herramientas que el modelo solicita.

    Flujo:
    1. El modelo responde con tool_calls (no con texto).
    2. Extraemos el nombre de la herramienta y sus argumentos.
    3. Ejecutamos la herramienta (nuestra skill).
    4. Enviamos el resultado de vuelta al modelo.
    5. El modelo genera la respuesta final en lenguaje natural.
    """
    mensaje_asistente = respuesta.choices[0].message
    mensajes.append(mensaje_asistente)

    for tool_call in mensaje_asistente.tool_calls:
        nombre = tool_call.function.name
        argumentos = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

        print(f"  [Skill] Ejecutando: {nombre}({argumentos})")

        # === AQUI ES DONDE LA SKILL SE EJECUTA ===
        resultado = ejecutar(nombre, argumentos)

        # Enviamos el resultado al modelo como mensaje de tipo "tool"
        mensajes.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": resultado
        })

    # Segunda llamada al modelo: ahora tiene los resultados
    respuesta_final = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        tools=TOOLS,
        tool_choice="auto"
    )

    # Si el modelo pide MAS herramientas, procesamos recursivamente
    if respuesta_final.choices[0].message.tool_calls:
        return procesar_tool_calls(respuesta_final, mensajes)

    return respuesta_final.choices[0].message.content


def main():
    print("=" * 60)
    print("  CHATBOT DE ANALISIS ESTADISTICO")
    print("  Modelo: {} (via Groq)".format(MODELO))
    print("  con validacion por escala de medicion")
    print("  Escribe 'salir' para terminar")
    print("=" * 60)
    print()

    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        pregunta = input("Tu: ").strip()

        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("Hasta luego!")
            break

        mensajes.append({"role": "user", "content": pregunta})

        try:
            respuesta = cliente.chat.completions.create(
                model=MODELO,
                messages=mensajes,
                tools=TOOLS,
                tool_choice="auto"
            )

            mensaje = respuesta.choices[0].message

            # Si el modelo quiere usar herramientas
            if mensaje.tool_calls:
                texto = procesar_tool_calls(respuesta, mensajes)
            else:
                texto = mensaje.content
                mensajes.append(mensaje)

            print(f"\nBot: {texto}\n")

        except Exception as e:
            print(f"\n[Error]: {e}\n")
            mensajes.pop()


if __name__ == "__main__":
    main()
