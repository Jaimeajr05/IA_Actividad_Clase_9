import os

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="Laboratorio de lenguaje natural",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Laboratorio de lenguaje natural")

st.write(
    "Explora tokenización, embeddings, similitud entre frases "
    "y generación de texto."
)


# ==========================================================
# CARGA DE MODELOS
# Se guardan en caché para no cargarlos en cada interacción.
# ==========================================================

@st.cache_resource
def cargar_tokenizador(nombre):
    return AutoTokenizer.from_pretrained(nombre)


@st.cache_resource
def cargar_embeddings(nombre):
    return SentenceTransformer(nombre, device="cpu")


# ==========================================================
# BARRA LATERAL
# ==========================================================

with st.sidebar:
    st.header("Configuración de Groq")

    api_key = st.text_input(
        "API key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Solo se necesita para generar texto.",
    ).strip()

    st.markdown(
        "[Obtener una API key de Groq](https://console.groq.com/keys)"
    )

    st.caption(
        "La primera vez, los tokenizadores y modelos de embeddings "
        "se descargan de Internet. La descarga puede tardar."
    )

    st.info(
        "Los tokenizadores de esta práctica son independientes "
        "del modelo elegido para generar texto."
    )


tab_tokens, tab_embeddings, tab_generacion = st.tabs(
    [
        "1. Tokenización",
        "2. Embeddings y similitud",
        "3. Generación de texto",
    ]
)


# ==========================================================
# 1. TOKENIZACIÓN
# ==========================================================

with tab_tokens:
    st.header("Comparar tokenizadores")

    st.write(
        "Un token puede representar una palabra, parte de una palabra "
        "o un signo. Su ID depende del vocabulario de cada modelo."
    )

    texto = st.text_area(
        "Texto para tokenizar",
        value="La ingeniería matemática combina modelos y datos.",
        height=120,
    )

    tokenizadores = st.multiselect(
        "Selecciona uno o varios tokenizadores",
        options=[
            "gpt2",
            "bert-base-multilingual-cased",
            "distilbert-base-uncased",
        ],
        default=[
            "gpt2",
            "bert-base-multilingual-cased",
        ],
    )

    especiales = st.checkbox(
        "Agregar tokens especiales del modelo",
        value=False,
    )

    if st.button("Tokenizar", type="primary"):
        if not texto.strip():
            st.warning("Escribe un texto.")
        elif not tokenizadores:
            st.warning("Selecciona al menos un tokenizador.")
        else:
            for nombre in tokenizadores:
                st.subheader(nombre)

                try:
                    with st.spinner("Cargando tokenizador..."):
                        tokenizador = cargar_tokenizador(nombre)

                        ids = tokenizador.encode(
                            texto,
                            add_special_tokens=especiales,
                        )

                        tokens = tokenizador.convert_ids_to_tokens(ids)

                    tabla = pd.DataFrame(
                        {
                            "Posición": range(len(ids)),
                            "Token": tokens,
                            "Token ID": ids,
                        }
                    )

                    st.metric("Cantidad de tokens", len(ids))

                    st.dataframe(
                        tabla,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.write("Texto reconstruido:")

                    st.code(
                        tokenizador.decode(
                            ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        language=None,
                    )

                except Exception:
                    st.error(
                        f"No se pudo cargar o ejecutar {nombre}. "
                        "Revisa la conexión y la instalación de dependencias."
                    )


# ==========================================================
# 2. EMBEDDINGS Y SIMILITUD DE COSENO
# ==========================================================

with tab_embeddings:
    st.header("Representación vectorial de frases")

    st.write(
        "Un embedding representa una frase mediante un vector numérico. "
        "La similitud de coseno compara la dirección de dos vectores."
    )

    st.latex(
        r"\operatorname{sim}(u,v)="
        r"\frac{u^\top v}{\|u\|\|v\|}"
    )

    frases_texto = st.text_area(
        "Escribe una frase por línea",
        value=(
            "El perro juega en el parque.\n"
            "Un cachorro corre por el jardín.\n"
            "Estoy estudiando álgebra lineal.\n"
            "Me gustan las matemáticas."
        ),
        height=160,
    )

    modelos_embeddings = st.multiselect(
        "Modelos de embeddings para comparar",
        options=[
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        default=[
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ],
    )

    st.caption(
        "Los dos primeros modelos son multilingües. "
        "all-MiniLM-L6-v2 está orientado principalmente al inglés."
    )

    if st.button("Calcular embeddings y similitud", type="primary"):
        frases = [
            frase.strip()
            for frase in frases_texto.splitlines()
            if frase.strip()
        ]

        if len(frases) < 2:
            st.warning("Escribe al menos dos frases.")
        elif len(frases) > 30:
            st.warning("Utiliza un máximo de 30 frases por comparación.")
        elif not modelos_embeddings:
            st.warning("Selecciona al menos un modelo.")
        else:
            etiquetas = [
                f"Frase {i + 1}"
                for i in range(len(frases))
            ]

            st.dataframe(
                pd.DataFrame(
                    {
                        "Etiqueta": etiquetas,
                        "Texto": frases,
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            for nombre in modelos_embeddings:
                st.subheader(nombre)

                try:
                    with st.spinner("Cargando modelo y calculando..."):
                        modelo = cargar_embeddings(nombre)

                        # Detectar frases que superan el límite del modelo.
                        longitudes = [
                            len(
                                modelo.tokenizer.encode(
                                    frase,
                                    truncation=False,
                                )
                            )
                            for frase in frases
                        ]

                        if any(
                            longitud > modelo.max_seq_length
                            for longitud in longitudes
                        ):
                            st.warning(
                                "Alguna frase supera el límite de "
                                f"{modelo.max_seq_length} tokens "
                                "y será truncada por este modelo."
                            )

                        vectores = modelo.encode(
                            frases,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )

                        # Con vectores normalizados, el producto
                        # escalar equivale a la similitud de coseno.
                        similitudes = vectores @ vectores.T
                        similitudes = np.clip(
                            similitudes,
                            -1.0,
                            1.0,
                        )

                    st.metric(
                        "Dimensión de cada embedding",
                        vectores.shape[1],
                    )

                    matriz = pd.DataFrame(
                        similitudes,
                        index=etiquetas,
                        columns=etiquetas,
                    )

                    st.write("**Matriz de similitud de coseno**")

                    st.dataframe(
                        matriz.round(4),
                        use_container_width=True,
                    )

                    st.caption(
                        "Valores cercanos a 1 indican direcciones similares; "
                        "cercanos a 0, poca alineación; y negativos, "
                        "direcciones opuestas. No son probabilidades "
                        "ni garantizan que dos frases sean sinónimas "
                        "o antónimas."
                    )

                    # Buscar el par más similar sin incluir la diagonal.
                    filas, columnas = np.triu_indices(
                        len(frases),
                        k=1,
                    )

                    mejor = np.argmax(
                        similitudes[filas, columnas]
                    )

                    i = int(filas[mejor])
                    j = int(columnas[mejor])

                    st.success(
                        f"Par más similar: Frase {i + 1} y Frase {j + 1}. "
                        f"Similitud: {similitudes[i, j]:.4f}"
                    )

                    with st.expander("Ver los vectores completos"):
                        st.dataframe(
                            pd.DataFrame(
                                vectores,
                                index=etiquetas,
                                columns=[
                                    f"d_{k + 1}"
                                    for k in range(vectores.shape[1])
                                ],
                            ),
                            use_container_width=True,
                        )

                except Exception:
                    st.error(
                        f"No se pudo ejecutar {nombre}. "
                        "Revisa la conexión, las dependencias "
                        "y la memoria disponible."
                    )


# ==========================================================
# 3. GENERACIÓN DE TEXTO CON GROQ
# ==========================================================

with tab_generacion:
    st.header("Generar texto con Groq")

    st.write(
        "Consulta los modelos disponibles en tu cuenta y elige "
        "uno compatible con generación de texto."
    )

    # Evita conservar la lista si cambia la credencial.
    if st.session_state.get("_clave_actual") != api_key:
        st.session_state["_clave_actual"] = api_key
        st.session_state.pop("modelos_groq", None)

    if st.button("Consultar modelos disponibles"):
        if not api_key:
            st.warning("Ingresa tu API key en la barra lateral.")
        else:
            try:
                with Groq(
                    api_key=api_key,
                    timeout=60.0,
                    max_retries=1,
                ) as cliente:
                    respuesta = cliente.models.list()

                st.session_state["modelos_groq"] = sorted(
                    modelo.id
                    for modelo in respuesta.data
                )

                st.success("Lista de modelos actualizada.")

            except Exception:
                st.error(
                    "No se pudo consultar Groq. "
                    "Verifica tu API key y la conexión."
                )

    disponibles = st.session_state.get("modelos_groq", [])

    if disponibles:
        modelo_lista = st.selectbox(
            "Modelo disponible",
            disponibles,
        )
    else:
        modelo_lista = ""

    modelo_manual = st.text_input(
        "ID del modelo — opcional si ya seleccionaste uno",
        placeholder="Pega aquí el ID de un modelo de texto de Groq",
    ).strip()

    modelo_elegido = modelo_manual or modelo_lista

    st.caption(
        "La lista puede incluir modelos de audio u otras tareas. "
        "Selecciona uno compatible con Chat Completions."
    )

    instrucciones = st.text_area(
        "Instrucciones del sistema",
        value=(
            "Eres un asistente educativo. "
            "Responde en español con claridad y ejemplos."
        ),
    )

    prompt = st.text_area(
        "¿Qué quieres generar?",
        value=(
            "Explica qué es una red neuronal "
            "con un ejemplo sencillo."
        ),
        height=140,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        temperatura = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controla la aleatoriedad de la generación.",
        )

    with col2:
        top_p = st.slider(
            "Top-p",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="Controla la masa de probabilidad usada al muestrear.",
        )

    with col3:
        max_tokens = st.number_input(
            "Máximo de tokens de salida",
            min_value=64,
            max_value=8192,
            value=512,
            step=64,
        )

    st.caption(
        "Para observar el efecto de cada parámetro, cambia uno "
        "a la vez. Algunos modelos restringen los valores admitidos."
    )

    if st.button("Generar texto", type="primary"):
        if not api_key:
            st.warning("Ingresa tu API key.")
        elif not modelo_elegido:
            st.warning("Selecciona o escribe el ID de un modelo.")
        elif not prompt.strip():
            st.warning("Escribe una solicitud.")
        else:
            try:
                mensajes = []

                if instrucciones.strip():
                    mensajes.append(
                        {
                            "role": "system",
                            "content": instrucciones.strip(),
                        }
                    )

                mensajes.append(
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                )

                with st.spinner("Generando respuesta..."):
                    with Groq(
                        api_key=api_key,
                        timeout=60.0,
                        max_retries=1,
                    ) as cliente:
                        respuesta = cliente.chat.completions.create(
                            model=modelo_elegido,
                            messages=mensajes,
                            temperature=float(temperatura),
                            top_p=float(top_p),
                            max_completion_tokens=int(max_tokens),
                        )

                st.subheader("Resultado")

                st.markdown(
                    respuesta.choices[0].message.content
                    or "El modelo no devolvió texto."
                )

                if respuesta.usage:
                    a, b, c = st.columns(3)

                    a.metric(
                        "Tokens de entrada",
                        respuesta.usage.prompt_tokens,
                    )

                    b.metric(
                        "Tokens de salida",
                        respuesta.usage.completion_tokens,
                    )

                    c.metric(
                        "Tokens totales",
                        respuesta.usage.total_tokens,
                    )

                if respuesta.choices[0].finish_reason == "length":
                    st.warning(
                        "Se alcanzó el límite de generación. "
                        "Puedes aumentar el máximo de tokens."
                    )

            except Exception as error:
                estado = getattr(error, "status_code", None)

                if estado == 401:
                    st.error("API key inválida.")
                elif estado == 429:
                    st.error(
                        "Se alcanzó un límite de uso de Groq. "
                        "Espera un momento o revisa tu cuota."
                    )
                elif estado in (400, 404):
                    st.error(
                        "Revisa que el modelo admita Chat Completions "
                        "y los parámetros seleccionados."
                    )
                else:
                    st.error(
                        "No se pudo generar la respuesta. "
                        "Revisa la conexión y el acceso al modelo."
                    )
