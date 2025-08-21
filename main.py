# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from docxtpl import DocxTemplate
import google.generativeai as genai
import os
import re
import time
import zipfile
from io import BytesIO

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Ensamblador de Fichas Técnicas con IA",
    page_icon="🤖",
    layout="wide"
)

# --- FUNCIONES DE LÓGICA ---

def limpiar_html(texto_html):
    """Limpia etiquetas HTML de un texto."""
    if not isinstance(texto_html, str):
        return texto_html
    cleanr = re.compile('<.*?>')
    texto_limpio = re.sub(cleanr, '', texto_html)
    return texto_limpio

def setup_model(api_key):
    """Configura y retorna el cliente para el modelo Gemini."""
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 8192
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"Error al configurar la API de Google: {e}")
        return None

# --- PROMPTS INTERNOS ---

PROMPT_ANALISIS = """
Actúa como un experto en psicometría y pedagogía. Tu misión es deconstruir un ítem de evaluación.

--- INSUMOS DEL ÍTEM ---
- Grado: {ItemGradoId}
- Competencia: {CompetenciaNombre}
- Evidencia: {EvidenciaNombre}
- Contexto: {ItemContexto}
- Enunciado: {ItemEnunciado}
- Opción A: {OpcionA}
- Opción B: {OpcionB}
- Opción C: {OpcionC}
- Opción D: {OpcionD}
- Respuesta Clave: {AlternativaClave}

--- INSTRUCCIONES ---

FASE 1: RUTA COGNITIVA
Describe, en un párrafo continuo y de forma impersonal, el procedimiento mental que un estudiante debe ejecutar para llegar a la respuesta correcta.
1.  Genera la Ruta Cognitiva: Describe el paso a paso mental y lógico que un estudiante debe seguir para llegar a la respuesta correcta. Usa verbos que representen procesos cognitivos.
2.  Auto-Verificación: Revisa que la ruta se alinee con la Competencia ('{CompetenciaNombre}') y la Evidencia ('{EvidenciaNombre}').
3.  Justificación Final: El último paso debe justificar la elección de la respuesta correcta.

FASE 2: ANÁLISIS DE OPCIONES NO VÁLIDAS
- Para cada opción incorrecta, identifica la naturaleza del error y explica el razonamiento fallido.
- Luego, explica el posible razonamiento que lleva al estudiante a cometer ese error.
- Finalmente, clarifica por qué esa opción es incorrecta en el contexto de la tarea evaluativa.

--- FORMATO DE SALIDA (REGLA CRÍTICA)---
Responde únicamente con los dos títulos siguientes, en este orden y sin añadir texto adicional.

Ruta Cognitiva Correcta:
[Párrafo continuo y detallado. Ejemplo: Para resolver correctamente este ítem, el estudiante primero debe [verbo cognitivo 1]... Luego, necesita [verbo cognitivo 2]... Este proceso le permite [verbo cognitivo 3]..., lo que finalmente lo lleva a concluir que la opción {AlternativaClave} es la correcta porque [justificación final].]

Análisis de Opciones No Válidas:
- **Opción [Letra del distractor]:** El estudiante podría escoger esta opción si comete un error de [naturaleza de la confusión u error], lo que lo lleva a pensar que [razonamiento erróneo]. Sin embargo, esto es incorrecto porque [razón clara y concisa].
"""

PROMPT_SINTESIS = """
Actúa como un experto en evaluación. Basado en la siguiente Ruta Cognitiva, redacta una única frase (máximo 2 renglones) que resuma la habilidad principal que se está evaluando.
Reglas:
1. Comienza obligatoriamente con "Este ítem evalúa la capacidad del estudiante para...".
2. Describe procesos cognitivos genéricos, no menciones detalles específicos del ítem.
3. Usa la taxonomía de referencia para que el lenguaje sea preciso.

RUTA COGNITIVA:
---
{ruta_cognitiva_texto}
---

TAXONOMÍA DE REFERENCIA:
- Competencia: {CompetenciaNombre}
- Evidencia de Aprendizaje: {EvidenciaNombre}

RESPUESTA:
"""

PROMPT_RECOMENDACIONES = """
Actúa como un diseñador instruccional experto. Basado en la información del ítem, genera tres recomendaciones distintas.

--- FORMATO DE SALIDA (REGLA CRÍTICA) ---
Usa obligatoriamente la siguiente estructura de encabezados.

RECOMENDACIÓN PARA FORTALECER
[Describe una actividad de aprendizaje creativa y no tradicional para un estudiante que respondió incorrectamente, enfocada en remediar los errores conceptuales.]

RECOMENDACIÓN PARA AVANZAR
[Describe una actividad de profundización o un desafío para un estudiante que respondió correctamente, para llevar su habilidad al siguiente nivel.]

OPORTUNIDAD DE MEJORA
[Actúa como un tutor dirigiéndote al estudiante. Proporciona un consejo práctico y directo (máximo dos párrafos) enfocado en el error más común revelado en el análisis de distractores. No uses introducciones formales como "Se recomienda que...". Sé accionable.]

--- INFORMACIÓN DEL ÍTEM Y ANÁLISIS ---
- Qué Evalúa: {que_evalua_sintetizado}
- Análisis completo: {analisis_central_generado}
- Competencia: {CompetenciaNombre}
- Grado: {ItemGradoId}
"""

# --- FUNCIÓN PARA CONSTRUIR PROMPTS ---
def construir_prompt(fila, plantilla, **kwargs):
    fila = fila.fillna('')
    campos = {k: fila.get(k, '') for k in [
        'ItemContexto', 'ItemEnunciado', 'ComponenteNombre', 'CompetenciaNombre',
        'AfirmacionNombre', 'EvidenciaNombre', 'Tipologia Textual', 'ItemGradoId',
        'Analisis_Errores', 'AlternativaClave', 'OpcionA', 'OpcionB', 'OpcionC', 'OpcionD'
    ]}
    campos.update(kwargs)
    return plantilla.format(**campos)

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---
st.title("🤖 Ensamblador de Fichas Técnicas con IA")
st.markdown("Una aplicación para enriquecer datos pedagógicos y generar fichas personalizadas.")

if 'df_enriquecido' not in st.session_state: st.session_state.df_enriquecido = None
if 'zip_buffer' not in st.session_state: st.session_state.zip_buffer = None

st.sidebar.header("🔑 Configuración Obligatoria")
api_key = st.sidebar.text_input("Ingresa tu Clave API de Google AI (Gemini)", type="password")

st.header("Paso 1: Carga tus Archivos")
col1, col2 = st.columns(2)
with col1: archivo_excel = st.file_uploader("Sube tu Excel con los datos base", type=["xlsx"])
with col2: archivo_plantilla = st.file_uploader("Sube tu Plantilla de Word", type=["docx"])

st.header("Paso 2: Enriquece tus Datos con IA")
if st.button("🤖 Iniciar Análisis y Generación", disabled=(not api_key or not archivo_excel)):
    model = setup_model(api_key)
    if model:
        with st.spinner("Procesando archivo Excel y preparando datos..."):
            df = pd.read_excel(archivo_excel)
            for col in df.columns:
                if df[col].dtype == 'object': df[col] = df[col].apply(limpiar_html)
            
            columnas_nuevas = ["Que_Evalua", "Justificacion_Correcta", "Analisis_Distractores",
                               "Justificacion_A", "Justificacion_B", "Justificacion_C", "Justificacion_D",
                               "Recomendacion_Fortalecer", "Recomendacion_Avanzar", "oportunidad_de_mejora"]
            for col in columnas_nuevas:
                if col not in df.columns: df[col] = ""
            st.success("Datos limpios y listos.")

        progress_bar_main = st.progress(0, text="Iniciando Proceso...")
        total_filas = len(df)

        for i, fila in df.iterrows():
            item_id = fila.get('ItemId', i + 1)
            st.markdown(f"--- \n ### Procesando Ítem: **{item_id}**")
            progress_bar_main.progress((i + 1) / total_filas, text=f"Procesando ítem {i+1}/{total_filas}")

            with st.container(border=True):
                try:
                    # --- PASO 1: ANÁLISIS CENTRAL ---
                    st.write(f"**Paso 1/3:** Realizando análisis central...")
                    prompt_paso1 = construir_prompt(fila, PROMPT_ANALISIS)
                    response_paso1 = model.generate_content(prompt_paso1)
                    analisis_central = response_paso1.text.strip()
                    time.sleep(1.5)

                    header_correcta = "Ruta Cognitiva Correcta:"
                    header_distractores = "Análisis de Opciones No Válidas:"
                    idx_distractores = analisis_central.find(header_distractores)

                    if idx_distractores != -1:
                        ruta_cognitiva = analisis_central[len(header_correcta):idx_distractores].strip()
                        analisis_distractores_bloque = analisis_central[idx_distractores + len(header_distractores):].strip()
                        
                        df.loc[i, "Justificacion_Correcta"] = ruta_cognitiva
                        df.loc[i, "Analisis_Distractores"] = analisis_distractores_bloque

                        clave_correcta = str(fila.get('AlternativaClave', '')).strip().upper()
                        opciones = ['A', 'B', 'C', 'D']
                        
                        for opt in opciones:
                            if opt == clave_correcta:
                                df.loc[i, f"Justificacion_{opt}"] = ruta_cognitiva
                            else:
                                # Lógica de separación mejorada
                                pattern = re.compile(rf"-\s*\*\*\s*Opción\s*{opt}:\s*\*\*(.*?)(?=-\s*\*\*Opción|\Z)", re.DOTALL | re.IGNORECASE)
                                match = pattern.search(analisis_distractores_bloque)
                                df.loc[i, f"Justificacion_{opt}"] = match.group(1).strip() if match else "Análisis del distractor no encontrado."
                    else:
                        df.loc[i, "Justificacion_Correcta"] = analisis_central
                        df.loc[i, "Analisis_Distractores"] = "Error al parsear distractores"

                    # --- PASO 2: SÍNTESIS DEL "QUÉ EVALÚA" ---
                    st.write(f"**Paso 2/3:** Sintetizando 'Qué Evalúa'...")
                    prompt_paso2 = construir_prompt(fila, PROMPT_SINTESIS, ruta_cognitiva_texto=df.loc[i, "Justificacion_Correcta"])
                    response_paso2 = model.generate_content(prompt_paso2)
                    df.loc[i, "Que_Evalua"] = response_paso2.text.strip()
                    time.sleep(1.5)
                    
                    # --- PASO 3: GENERACIÓN DE RECOMENDACIONES (3 en 1) ---
                    st.write(f"**Paso 3/3:** Generando recomendaciones pedagógicas...")
                    prompt_paso3 = construir_prompt(fila, PROMPT_RECOMENDACIONES, que_evalua_sintetizado=df.loc[i, "Que_Evalua"], analisis_central_generado=analisis_central)
                    response_paso3 = model.generate_content(prompt_paso3)
                    recomendaciones = response_paso3.text.strip()
                    
                    # Lógica para separar las 3 recomendaciones
                    fortalecer_match = re.search(r'RECOMENDACIÓN PARA FORTALECER(.*?)RECOMENDACIÓN PARA AVANZAR', recomendaciones, re.DOTALL | re.IGNORECASE)
                    avanzar_match = re.search(r'RECOMENDACIÓN PARA AVANZAR(.*?)OPORTUNIDAD DE MEJORA', recomendaciones, re.DOTALL | re.IGNORECASE)
                    oportunidad_match = re.search(r'OPORTUNIDAD DE MEJORA(.*?)$', recomendaciones, re.DOTALL | re.IGNORECASE)

                    df.loc[i, "Recomendacion_Fortalecer"] = fortalecer_match.group(1).strip() if fortalecer_match else "No generada."
                    df.loc[i, "Recomendacion_Avanzar"] = avanzar_match.group(1).strip() if avanzar_match else "No generada."
                    df.loc[i, "oportunidad_de_mejora"] = oportunidad_match.group(1).strip() if oportunidad_match else "No generada."

                    st.success(f"Ítem {item_id} procesado con éxito.")

                except Exception as e:
                    st.error(f"Ocurrió un error procesando el ítem {item_id}: {e}")
                    for col in columnas_nuevas: df.loc[i, col] = f"ERROR: {e}"
        
        progress_bar_main.progress(1.0, text="¡Proceso completado!")
        st.session_state.df_enriquecido = df
        st.balloons()

# --- PASO 3: Vista Previa y Descarga de Excel ---
if st.session_state.df_enriquecido is not None:
    st.header("Paso 3: Verifica y Descarga los Datos Enriquecidos")
    st.dataframe(st.session_state.df_enriquecido.head())
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.df_enriquecido.to_excel(writer, index=False, sheet_name='Datos Enriquecidos')
    output_excel.seek(0)
    st.download_button(
        label="📥 Descargar Excel Enriquecido", data=output_excel,
        file_name="excel_enriquecido_con_ia.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- PASO 4: Ensamblaje de Fichas ---
if st.session_state.df_enriquecido is not None and archivo_plantilla is not None:
    st.header("Paso 4: Ensambla las Fichas Técnicas")
    columna_nombre_archivo = st.text_input("Escribe el nombre de la columna para nombrar los archivos (ej. ItemId)", value="ItemId")
    if st.button("📄 Ensamblar Fichas Técnicas", type="primary"):
        df_final = st.session_state.df_enriquecido
        if columna_nombre_archivo not in df_final.columns:
            st.error(f"La columna '{columna_nombre_archivo}' no existe en el Excel. Elige una de: {', '.join(df_final.columns)}")
        else:
            with st.spinner("Ensamblando todas las fichas en un archivo .zip..."):
                plantilla_bytes = BytesIO(archivo_plantilla.getvalue())
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    total_docs = len(df_final)
                    progress_bar_zip = st.progress(0, text="Iniciando ensamblaje...")
                    for i, fila in df_final.iterrows():
                        plantilla_bytes.seek(0)
                        doc = DocxTemplate(plantilla_bytes)
                        contexto = fila.to_dict()
                        contexto_limpio = {k: (v if pd.notna(v) else "") for k, v in contexto.items()}
                        doc.render(contexto_limpio)
                        doc_buffer = BytesIO()
                        doc.save(doc_buffer)
                        doc_buffer.seek(0)
                        nombre_base = str(fila.get(columna_nombre_archivo, f"ficha_{i+1}")).replace('/', '_').replace('\\', '_')
                        zip_file.writestr(f"{nombre_base}.docx", doc_buffer.getvalue())
                        progress_bar_zip.progress((i + 1) / total_docs, text=f"Añadiendo ficha {i+1}/{total_docs} al .zip")
                st.session_state.zip_buffer = zip_buffer
                st.success("¡Ensamblaje completado!")

# --- PASO 5: Descarga Final del ZIP ---
if st.session_state.zip_buffer:
    st.header("Paso 5: Descarga el Resultado Final")
    st.download_button(
        label="📥 Descargar TODAS las fichas (.zip)",
        data=st.session_state.zip_buffer.getvalue(),
        file_name="fichas_tecnicas_generadas.zip",
        mime="application/zip"
    )
