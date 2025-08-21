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

# --- PROMPTS Y EJEMPLOS INTERNOS (FEW-SHOT PROMPTING) ---
# Ya no se leen de archivos externos

PROMPT_ANALISIS = """
Actúa como un experto en psicometría y pedagogía. Tu tarea es analizar un ítem de evaluación de opción múltiple.

Tu respuesta DEBE seguir estrictamente la siguiente estructura de encabezados, sin añadir texto adicional fuera de ellos:

Ruta Cognitiva Correcta:
[Describe el proceso mental paso a paso que un estudiante debe seguir para llegar a la respuesta correcta ({AlternativaClave}). Sé claro, lógico y detalla cada etapa del razonamiento necesario, basándote en el contexto y el enunciado del ítem.]

Análisis de Opciones No Válidas:
[Para CADA UNA de las opciones incorrectas, explica por qué no es válida. Inicia cada explicación con "Opción A:", "Opción B:", etc. Identifica el tipo de error conceptual o procedimental que comete el estudiante al elegirla.]

--- DATOS DEL ÍTEM ---
- Grado: {ItemGradoId}
- Competencia: {CompetenciaNombre}
- Contexto: {ItemContexto}
- Enunciado: {ItemEnunciado}
- Opción A: {OpcionA}
- Opción B: {OpcionB}
- Opción C: {OpcionC}
- Opción D: {OpcionD}
- Respuesta Clave: {AlternativaClave}
"""

PROMPT_SINTESIS = """
Actúa como un experto en evaluación que sintetiza análisis complejos en una sola frase concisa. Basándote exclusivamente en el siguiente ANÁLISIS DE LA RUTA COGNITIVA, redacta una única frase (máximo 2 renglones) que resuma la habilidad principal que se está evaluando.

Reglas:
1. La frase debe comenzar obligatoriamente con "Este ítem evalúa la capacidad del estudiante para...".
2. La frase debe describir procesos cognitivos genéricos, sin mencionar elementos específicos del texto o del ítem.
3. Utiliza la taxonomía de referencia para asegurar que el lenguaje sea preciso.

ANÁLISIS DE LA RUTA COGNITIVA:
---
{ruta_cognitiva_texto}
---

TAXONOMÍA DE REFERENCIA:
- Competencia: {CompetenciaNombre}
- Evidencia de Aprendizaje: {EvidenciaNombre}

FORMATO DE SALIDA:
Responde únicamente con la frase solicitada.
"""

PROMPT_RECOMENDACIONES = """
Actúa como un diseñador instruccional experto, especializado en crear actividades de lectura novedosas. Basado en la información del ítem, genera dos recomendaciones.

RECOMENDACIÓN PARA FORTALECER
Describe una actividad o estrategia de aprendizaje para un estudiante que respondió incorrectamente. La recomendación debe enfocarse en remediar los errores conceptuales identificados en el análisis de los distractores. Sé creativo y evita ejercicios típicos.

RECOMENDACIÓN PARA AVANZAR
Describe una actividad de profundización o un desafío para un estudiante que respondió correctamente. El objetivo es llevar su habilidad al siguiente nivel, conectándola con temas más complejos o aplicaciones prácticas.

--- INFORMACIÓN DEL ÍTEM Y ANÁLISIS ---
{analisis_central_generado}
- Qué Evalúa: {que_evalua_sintetizado}
- Competencia: {CompetenciaNombre}
- Grado: {ItemGradoId}
"""

PROMPT_PARAFRASEO = """
Actúa como un redactor pedagógico conciso. Convierte la siguiente recomendación en una "Oportunidad de Mejora".
La "Oportunidad de Mejora" debe ser una sugerencia práctica y directa (máximo dos frases) que el estudiante puede aplicar de inmediato.
NO uses frases introductorias como "Se recomienda que..." o "Para mejorar...". Sé directo y accionable.

Recomendación original:
"{recomendacion_fortalecer}"

Oportunidad de Mejora:
"""

# --- FUNCIONES PARA CONSTRUIR PROMPTS ---
def construir_prompt(fila, plantilla):
    fila = fila.fillna('')
    # Usamos un diccionario con todos los posibles campos
    # para evitar errores si una columna no existe.
    campos = {
        'ItemContexto': fila.get('ItemContexto', ''), 'ItemEnunciado': fila.get('ItemEnunciado', ''),
        'ComponenteNombre': fila.get('ComponenteNombre', ''), 'CompetenciaNombre': fila.get('CompetenciaNombre', ''),
        'AfirmacionNombre': fila.get('AfirmacionNombre', ''), 'EvidenciaNombre': fila.get('EvidenciaNombre', ''),
        'Tipologia Textual': fila.get('Tipologia Textual', ''), 'ItemGradoId': fila.get('ItemGradoId', ''),
        'Analisis_Errores': fila.get('Analisis_Errores', ''), 'AlternativaClave': fila.get('AlternativaClave', ''),
        'OpcionA': fila.get('OpcionA', ''), 'OpcionB': fila.get('OpcionB', ''),
        'OpcionC': fila.get('OpcionC', ''), 'OpcionD': fila.get('OpcionD', '')
    }
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
                    st.write(f"**Paso 1/4:** Realizando análisis central...")
                    prompt_paso1 = construir_prompt(fila, PROMPT_ANALISIS)
                    response_paso1 = model.generate_content(prompt_paso1)
                    analisis_central = response_paso1.text.strip()
                    time.sleep(1)

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
                            if opt != clave_correcta:
                                pattern = re.compile(rf"Opción\s*{opt}:\s*(.*?)(?=\s*-\s*Opción\s*[A-D]:|$)", re.DOTALL | re.IGNORECASE)
                                match = pattern.search(analisis_distractores_bloque)
                                if match:
                                    df.loc[i, f"Justificacion_{opt}"] = match.group(1).strip()
                    else:
                        df.loc[i, "Justificacion_Correcta"] = analisis_central
                        df.loc[i, "Analisis_Distractores"] = "Error al parsear distractores"

                    # --- PASO 2: SÍNTESIS DEL "QUÉ EVALÚA" ---
                    st.write(f"**Paso 2/4:** Sintetizando 'Qué Evalúa'...")
                    prompt_paso2 = construir_prompt_paso2_sintesis_que_evalua(analisis_central, fila)
                    response_paso2 = model.generate_content(prompt_paso2)
                    df.loc[i, "Que_Evalua"] = response_paso2.text.strip()
                    time.sleep(1)
                    
                    # --- PASO 3: GENERACIÓN DE RECOMENDACIONES ---
                    st.write(f"**Paso 3/4:** Generando recomendaciones...")
                    prompt_paso3 = construir_prompt_paso3_recomendaciones(df.loc[i, "Que_Evalua"], analisis_central, fila)
                    response_paso3 = model.generate_content(prompt_paso3)
                    recomendaciones = response_paso3.text.strip()
                    
                    idx_avanzar = recomendaciones.upper().find("RECOMENDACIÓN PARA AVANZAR")
                    if idx_avanzar != -1:
                        fortalecer = recomendaciones[:idx_avanzar].replace("RECOMENDACIÓN PARA FORTALECER", "").strip()
                        avanzar = recomendaciones[idx_avanzar:].replace("RECOMENDACIÓN PARA AVANZAR", "").strip()
                    else:
                        fortalecer = recomendaciones.replace("RECOMENDACIÓN PARA FORTALECER", "").strip()
                        avanzar = "No generada"
                    
                    df.loc[i, "Recomendacion_Fortalecer"] = fortalecer
                    df.loc[i, "Recomendacion_Avanzar"] = avanzar

                    # --- PASO 4: PARAFRASEO PARA OPORTUNIDAD DE MEJORA ---
                    if fortalecer != "No generada" and fortalecer.strip() != "":
                        st.write(f"**Paso 4/4:** Creando oportunidad de mejora...")
                        prompt_parafraseo = PROMPT_PARAFRASEO.format(recomendacion_fortalecer=fortalecer)
                        response_parafraseo = model.generate_content(prompt_parafraseo)
                        df.loc[i, "oportunidad_de_mejora"] = response_parafraseo.text.strip()
                    else:
                        df.loc[i, "oportunidad_de_mejora"] = "No se generó recomendación para fortalecer."

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
                        plantilla_bytes.seek(0) # Reiniciar el buffer de la plantilla para cada documento
                        doc = DocxTemplate(plantilla_bytes)
                        contexto = fila.to_dict()
                        contexto_limpio = {k: (v if pd.notna(v) else "") for k, v in contexto.items()}
                        doc.render(contexto_limpio)
                        doc_buffer = BytesIO()
                        doc.save(doc_buffer)
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
