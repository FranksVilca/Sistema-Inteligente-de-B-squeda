import streamlit as st
import pandas as pd
from vectoria import MultiSiteVectorDB
import time
from datetime import datetime
import json
import unicodedata
import re
import html
import urllib.parse
import logging
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Buscador de Documentación Técnica",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def normalizar_texto_utf8(texto):
    """Normaliza texto UTF-8 para mejor manejo y búsqueda"""
    if not texto:
        return ""
    
    try:
        texto = unicodedata.normalize('NFKC', str(texto))
        texto = html.unescape(texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    except Exception as e:
        logger.error(f"Error al normalizar texto: {e}")
        return str(texto)

def validar_url(url):
    """Valida y limpia URLs con caracteres especiales"""
    if not url:
        return None
    
    try:
        url = normalizar_texto_utf8(url)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            return None
            
        return urllib.parse.urlunparse(parsed)
    except Exception as e:
        logger.error(f"Error al validar URL: {e}")
        return None

def inicializar_estado():
    """Inicializa el estado de la sesión con valores por defecto"""
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'ultima_busqueda' not in st.session_state:
        st.session_state.ultima_busqueda = ""
    if 'resultados_cache' not in st.session_state:
        st.session_state.resultados_cache = {}
    if 'fuentes_cargadas' not in st.session_state:
        st.session_state.fuentes_cargadas = False


@st.cache_resource
def inicializar_vector_db():
    """Inicializa la base de datos vectorial con manejo de errores robusto"""
    try:
        db = MultiSiteVectorDB()
        logger.info("Base de datos vectorial inicializada correctamente")
        return db
    except Exception as e:
        error_msg = f"Error al inicializar la base de datos: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise Exception(error_msg)

def realizar_busqueda(vector_db, query, n_resultados=5):
    """
    Realiza búsqueda adaptada a los métodos disponibles en el backend
    """
    try:
        # Verificar qué método de búsqueda está disponible
        if hasattr(vector_db, 'buscar_en_todos_sitios'):
            return vector_db.buscar_en_todos_sitios(consulta=query, n_resultados=n_resultados)
        elif hasattr(vector_db, 'buscar'):
            return vector_db.buscar(query=query, n_results=n_resultados)
        elif hasattr(vector_db.collection, 'query'):
            # Búsqueda directa en ChromaDB
            results = vector_db.collection.query(
                query_texts=[query],
                n_results=n_resultados
            )
            # Convertir resultados a formato esperado
            return convertir_resultados_chromadb(results)
        else:
            raise Exception("No se encontró método de búsqueda válido")
    except Exception as e:
        logger.error(f"Error en búsqueda: {str(e)}")
        raise

def convertir_resultados_chromadb(results):
    """
    Convierte resultados de ChromaDB al formato esperado por el frontend
    """
    resultados_formateados = []
    
    try:
        if not results or not results.get('documents'):
            return []
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results.get('metadatas') else []
        distances = results['distances'][0] if results.get('distances') else []
        
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            
            # Convertir distancia a puntuación (menor distancia = mayor puntuación)
            puntuacion = max(0, 1 - distance)
            
            resultado = {
                'contenido': doc,
                'titulo': metadata.get('titulo', 'Sin título'),
                'url': metadata.get('url', ''),
                'fuente': metadata.get('fuente', 'Desconocida'),
                'idioma': metadata.get('idioma', 'es'),
                'categoria': metadata.get('categoria', 'General'),
                'descripcion': metadata.get('descripcion', ''),
                'ultima_actualizacion': metadata.get('ultima_actualizacion', 'Desconocido'),
                'puntuacion': puntuacion
            }
            resultados_formateados.append(resultado)
            
    except Exception as e:
        logger.error(f"Error al convertir resultados: {str(e)}")
    
    return resultados_formateados


def obtener_fuentes(vector_db):
    """Obtiene fuentes con manejo de errores"""
    try:
        if hasattr(vector_db, 'obtener_fuentes'):
            return vector_db.obtener_fuentes()
        elif hasattr(vector_db, 'sources'):
            return vector_db.sources
        else:
            return []
    except Exception as e:
        logger.error(f"Error al obtener fuentes: {str(e)}")
        return []

def agregar_fuente_segura(vector_db, datos_fuente):
    """Agrega fuente con validación robusta"""
    try:
        if hasattr(vector_db, 'agregar_fuente'):
            return vector_db.agregar_fuente(**datos_fuente)
        else:
            # Implementación de respaldo
            if not hasattr(vector_db, 'sources'):
                vector_db.sources = []
            
            # Validar que no existe
            for fuente in vector_db.sources:
                if fuente.get('url') == datos_fuente['url']:
                    return {"exito": False, "mensaje": "La fuente ya existe"}
            
            # Agregar nueva fuente
            nueva_fuente = {
                **datos_fuente,
                "fecha_agregada": datetime.now().strftime("%Y-%m-%d"),
                "estado": "activo"
            }
            vector_db.sources.append(nueva_fuente)
            
            return {"exito": True, "mensaje": "Fuente agregada correctamente"}
    except Exception as e:
        logger.error(f"Error al agregar fuente: {str(e)}")
        return {"exito": False, "mensaje": f"Error: {str(e)}"}

def indexar_fuente_segura(vector_db, url, max_paginas=0):
    """Indexa fuente con progreso y manejo de errores"""
    try:
        if hasattr(vector_db, 'indexar_fuente'):
            return vector_db.indexar_fuente(url, max_paginas)
        else:
            return {"exito": False, "mensaje": "Método de indexación no disponible"}
    except Exception as e:
        logger.error(f"Error al indexar fuente: {str(e)}")
        return {"exito": False, "mensaje": f"Error en indexación: {str(e)}"}

# ===== CORRECCIÓN 6: CSS mejorado =====
st.markdown("""
<style>
    @charset "UTF-8";
    
    .resultado-container {
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
    
    .resultado-container:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .puntuacion-alta { color: #28a745; font-weight: bold; }
    .puntuacion-media { color: #ffc107; font-weight: bold; }
    .puntuacion-baja { color: #dc3545; font-weight: bold; }
    
    .fuente-tag, .idioma-tag, .categoria-tag {
        color: white;
        padding: 2px 8px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
        margin-right: 5px;
    }
    
    .fuente-tag { background-color: #007bff; }
    .idioma-tag { background-color: #6f42c1; }
    .categoria-tag { background-color: #17a2b8; }
    
    .contenido-preview {
        background-color: #ffffff;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        white-space: pre-wrap;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        line-height: 1.4;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .error-container {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success-container {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def mostrar_resultado(i, resultado):
    """Renderiza un resultado con manejo robusto de datos"""
    try:
        # Extraer datos con valores por defecto seguros
        titulo = normalizar_texto_utf8(resultado.get('titulo', 'Sin título'))
        url = resultado.get('url', '')
        fuente = normalizar_texto_utf8(resultado.get('fuente', 'Desconocida'))
        idioma = normalizar_texto_utf8(resultado.get('idioma', 'es')).upper()
        categoria = normalizar_texto_utf8(resultado.get('categoria', 'General'))
        contenido = resultado.get('contenido', 'Sin contenido')
        ultima_actualizacion = resultado.get('ultima_actualizacion', 'Desconocido')
        
        # Manejo seguro de puntuación
        try:
            puntuacion = float(resultado.get('puntuacion', 0))
        except (ValueError, TypeError):
            puntuacion = 0.0
        
        # Truncar contenido para preview
        contenido_preview = contenido[:300] + "..." if len(contenido) > 300 else contenido
        
        # Determinar clase de puntuación
        if puntuacion >= 0.8:
            clase_puntuacion = "puntuacion-alta"
        elif puntuacion >= 0.5:
            clase_puntuacion = "puntuacion-media"
        else:
            clase_puntuacion = "puntuacion-baja"
        
        # Validar URL
        url_segura = validar_url(url) if url else "#"
        
        # Renderizar resultado
        st.markdown(f"""
        <div class="resultado-container">
            <h3>🔗 {i}. {html.escape(titulo)}</h3>
            <p><strong>URL:</strong> <a href="{url_segura}" target="_blank" class="url-display">{html.escape(url)}</a></p>
            <p>
                <span class="fuente-tag">{html.escape(fuente)}</span>
                <span class="idioma-tag">{html.escape(idioma)}</span>
                <span class="categoria-tag">{html.escape(categoria)}</span>
            </p>
            <p><strong>📄 Contenido:</strong></p>
            <div class="contenido-preview">{html.escape(contenido_preview)}</div>
            <p><small>📅 Última actualización: {html.escape(str(ultima_actualizacion))}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error al mostrar resultado {i}: {str(e)}")
        st.error(f"Error al mostrar resultado {i}")


def main():
    """Función principal con inicialización robusta"""
    
    # Inicializar estado
    inicializar_estado()
    
    # Título principal
    st.title("🔍 Buscador de Documentación Técnica")
    st.markdown("### Encuentra información técnica en múltiples fuentes especializadas")

    # Inicialización de la base de datos con manejo de errores
    try:
        if st.session_state.vector_db is None:
            with st.spinner("🔧 Inicializando sistema..."):
                st.session_state.vector_db = inicializar_vector_db()
        
        vector_db = st.session_state.vector_db
        
        if vector_db:
            st.success("✅ Sistema inicializado correctamente")
        else:
            st.error("❌ Error al inicializar el sistema")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error crítico al inicializar: {str(e)}")
        st.info("💡 Intenta recargar la página o verifica la configuración de la base de datos")
        st.stop()

    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Buscar", "➕ Agregar Fuentes", "📚 Gestionar Fuentes", "📊 Estadísticas"])

    # ===== PESTAÑA 1: BÚSQUEDA MEJORADA =====
    with tab1:
        st.header("🔍 Buscar en la Base de Conocimientos")
        
        # Ejemplos de búsqueda
        with st.expander("💡 Ejemplos de búsqueda"):
            st.markdown("""
            - `configuración de APIs Python`
            - `función de búsqueda JavaScript`
            - `diseño responsivo CSS`
            - `autenticación OAuth JWT`
            - `machine learning scikit-learn`
            """)
        
        # Entrada de búsqueda
        query = st.text_input(
            "¿Qué deseas buscar?",
            value=st.session_state.ultima_busqueda,
            placeholder="Ej: 'configuración Python', 'función JavaScript', 'diseño CSS'",
            help="Ingresa términos de búsqueda. Soporta acentos y caracteres especiales."
        )
        
        # Configuración de búsqueda
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            num_resultados = st.slider(
                "📄 Número de resultados",
                min_value=1,
                max_value=20,
                value=5
            )
        
        with col2:
            mostrar_estadisticas = st.checkbox("📊 Estadísticas", value=True)
        
        with col3:
            usar_cache = st.checkbox("💾 Usar caché", value=True)
        
        # Botones
        col1, col2 = st.columns([1, 1])
        with col1:
            buscar_button = st.button("🔍 Buscar", type="primary", use_container_width=True)
        with col2:
            limpiar_button = st.button("🗑️ Limpiar", use_container_width=True)
        
        if limpiar_button:
            st.session_state.ultima_busqueda = ""
            st.session_state.resultados_cache = {}
            st.rerun()
        
        # Realizar búsqueda
        if buscar_button and query.strip():
            query_normalizada = normalizar_texto_utf8(query)
            st.session_state.ultima_busqueda = query_normalizada
            
            # Verificar caché
            cache_key = f"{query_normalizada}_{num_resultados}"
            if usar_cache and cache_key in st.session_state.resultados_cache:
                st.info("📦 Resultados obtenidos del caché")
                resultados = st.session_state.resultados_cache[cache_key]
            else:
                with st.spinner("🔍 Buscando en la base de conocimientos..."):
                    try:
                        start_time = time.time()
                        resultados = realizar_busqueda(vector_db, query_normalizada, num_resultados)
                        execution_time = time.time() - start_time
                        
                        # Guardar en caché
                        if usar_cache:
                            st.session_state.resultados_cache[cache_key] = resultados
                        
                        st.success(f"✅ Búsqueda completada en {execution_time:.2f} segundos")
                        
                    except Exception as e:
                        st.error(f"❌ Error en la búsqueda: {str(e)}")
                        st.info("💡 Verifica que la base de datos esté correctamente indexada")
                        resultados = []
            
            # Mostrar resultados
            if resultados:
                st.success(f"✅ Se encontraron {len(resultados)} resultados")
                
                # Estadísticas
                if mostrar_estadisticas and resultados:
                    with st.expander("📊 Estadísticas de búsqueda"):
                        try:
                            puntuaciones = [float(r.get('puntuacion', 0)) for r in resultados]
                            fuentes = [r.get('fuente', 'Desconocida') for r in resultados]
                            idiomas = [r.get('idioma', 'es') for r in resultados]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Puntuación media", f"{sum(puntuaciones)/len(puntuaciones):.1%}")
                            with col2:
                                st.metric("Mejor resultado", f"{max(puntuaciones):.1%}")
                            with col3:
                                st.metric("Fuentes únicas", len(set(fuentes)))
                            
                            # Distribución por idioma
                            idiomas_count = Counter(idiomas)
                            st.write("**Distribución por idioma:**")
                            for idioma, count in idiomas_count.items():
                                st.write(f"  - {idioma}: {count}")
                                
                        except Exception as e:
                            st.warning("No se pudieron calcular estadísticas completas")
                
                # Mostrar resultados
                for i, resultado in enumerate(resultados, 1):
                    mostrar_resultado(i, resultado)
            else:
                st.warning("❌ No se encontraron resultados")
                st.info("💡 Prueba con términos diferentes o agrega más fuentes")

    # ===== PESTAÑA 2: AGREGAR FUENTES MEJORADA =====
    with tab2:
        st.header("➕ Agregar Nueva Fuente de Documentación")
        
        with st.form("agregar_fuente_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                nueva_url = st.text_input(
                    "URL de la fuente*",
                    placeholder="https://docs.ejemplo.com",
                    help="URL base de la documentación a indexar"
                )
                
                nombre_fuente = st.text_input(
                    "Nombre descriptivo*",
                    placeholder="Documentación oficial de...",
                    help="Nombre identificativo de la fuente"
                )
            
            with col2:
                categoria = st.text_input(
                    "Categoría*",
                    placeholder="Categoria de la documentacion",
                    help="Categoría principal del contenido"
                )
                
                descripcion = st.text_area(
                    "Descripción",
                    placeholder="Descripción breve del contenido...",
                    height=100
                )
            
            # Configuración avanzada
            with st.expander("⚙️ Configuración de indexación"):
                col1, col2 = st.columns(2)
                
                with col1:
                    max_paginas = st.number_input(
                        "Máximo de páginas (0 = sin límite)",
                        min_value=0,
                        max_value=1000,
                        value=50,
                        help="Límite de páginas a indexar"
                    )
                
                with col2:
                    idioma = st.selectbox(
                        "Idioma principal",
                        options=["es", "en", "fr", "de", "pt"],
                        index=0
                    )
                
                indexar_inmediatamente = st.checkbox(
                    "Indexar inmediatamente",
                    value=True,
                    help="Iniciar indexación automáticamente después de agregar"
                )
            
            submitted = st.form_submit_button("➕ Agregar Fuente", type="primary")
            
            if submitted:
                if not nueva_url or not nombre_fuente:
                    st.error("⚠️ URL y nombre son obligatorios")
                else:
                    url_validada = validar_url(nueva_url)
                    if not url_validada:
                        st.error("⚠️ URL inválida")
                    else:
                        try:
                            # Preparar datos
                            datos_fuente = {
                                'url': url_validada,
                                'nombre': normalizar_texto_utf8(nombre_fuente),
                                'descripcion': normalizar_texto_utf8(descripcion),
                                'categoria': categoria,
                                'idioma': idioma
                            }
                            
                            # Agregar fuente
                            resultado = agregar_fuente_segura(vector_db, datos_fuente)
                            
                            if resultado.get("exito"):
                                st.success("✅ Fuente agregada correctamente")
                                
                                # Indexar si se solicitó
                                if indexar_inmediatamente:
                                    with st.status("🚀 Indexando contenido...", expanded=True) as status:
                                        try:
                                            resultado_indexacion = indexar_fuente_segura(
                                                vector_db, url_validada, max_paginas
                                            )
                                            
                                            if resultado_indexacion.get("exito"):
                                                docs_indexados = resultado_indexacion.get("documentos_indexados", 0)
                                                tiempo = resultado_indexacion.get("tiempo_ejecucion", 0)
                                                
                                                status.update(
                                                    label=f"✅ Indexación completada: {docs_indexados} documentos en {tiempo}s",
                                                    state="complete"
                                                )
                                                
                                                st.balloons()
                                            else:
                                                status.update(
                                                    label="❌ Error en la indexación",
                                                    state="error"
                                                )
                                                st.error(f"Error: {resultado_indexacion.get('mensaje', 'Error desconocido')}")
                                        
                                        except Exception as e:
                                            status.update(label="❌ Error en indexación", state="error")
                                            st.error(f"Error durante indexación: {str(e)}")
                            else:
                                st.error(f"❌ {resultado.get('mensaje', 'Error desconocido')}")
                                
                        except Exception as e:
                            st.error(f"❌ Error al agregar fuente: {str(e)}")

    # ===== PESTAÑA 3: GESTIONAR FUENTES =====
    with tab3:
        st.header("📚 Gestión de Fuentes")
        
        try:
            fuentes = obtener_fuentes(vector_db)
            
            if fuentes:
                st.success(f"📊 Total de fuentes: {len(fuentes)}")
                
                # Filtros
                col1, col2 = st.columns(2)
                with col1:
                    filtro_categoria = st.selectbox(
                        "Filtrar por categoría",
                        options=["Todas"] + list(set([f.get('categoria', 'Sin categoría') for f in fuentes]))
                    )
                
                with col2:
                    filtro_estado = st.selectbox(
                        "Filtrar por estado",
                        options=["Todos", "activo", "inactivo", "error"]
                    )
                
                # Aplicar filtros
                fuentes_filtradas = fuentes
                if filtro_categoria != "Todas":
                    fuentes_filtradas = [f for f in fuentes_filtradas if f.get('categoria') == filtro_categoria]
                if filtro_estado != "Todos":
                    fuentes_filtradas = [f for f in fuentes_filtradas if f.get('estado') == filtro_estado]
                
                # Mostrar fuentes
                for i, fuente in enumerate(fuentes_filtradas):
                    with st.expander(f"📄 {fuente.get('nombre', 'Sin nombre')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**URL:** {fuente.get('url', 'N/A')}")
                            st.write(f"**Categoría:** {fuente.get('categoria', 'N/A')}")
                            st.write(f"**Idioma:** {fuente.get('idioma', 'N/A')}")
                        
                        with col2:
                            st.write(f"**Estado:** {fuente.get('estado', 'N/A')}")
                            st.write(f"**Agregada:** {fuente.get('fecha_agregada', 'N/A')}")
                        
                        if fuente.get('descripcion'):
                            st.write(f"**Descripción:** {fuente['descripcion']}")
                        
                        # Botones de acción
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"🔄 Re-indexar", key=f"reindex_{i}"):
                                with st.spinner("Re-indexando..."):
                                    try:
                                        resultado = indexar_fuente_segura(vector_db, fuente['url'])
                                        if resultado.get("exito"):
                                            st.success("✅ Re-indexación completada")
                                        else:
                                            st.error(f"❌ Error: {resultado.get('mensaje')}")
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("📝 No hay fuentes configuradas. Agrega algunas en la pestaña anterior.")
                
        except Exception as e:
            st.error(f"❌ Error al cargar fuentes: {str(e)}")

    # ===== PESTAÑA 4: ESTADÍSTICAS =====
    with tab4:
        st.header("📊 Estadísticas del Sistema")
        
        try:
            # Estadísticas de la base de datos
            if hasattr(vector_db.collection, 'count'):
                total_docs = vector_db.collection.count()
            else:
                total_docs = "No disponible"
            
            fuentes = obtener_fuentes(vector_db)
            total_fuentes = len(fuentes)
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📚 Total Documentos", total_docs)
            
            with col2:
                st.metric("🌐 Total Fuentes", total_fuentes)
            
            with col3:
                fuentes_activas = len([f for f in fuentes if f.get('estado') == 'activo'])
                st.metric("✅ Fuentes Activas", fuentes_activas)
            
            with col4:
                cache_size = len(st.session_state.resultados_cache)
                st.metric("💾 Caché Búsquedas", cache_size)
            
            # Gráficos de estadísticas
            if fuentes:
                st.subheader("📈 Distribución por Categorías")
                
                # Contar por categorías
                categorias = [f.get('categoria', 'Sin categoría') for f in fuentes]
                categoria_counts = Counter(categorias)
                
                if categoria_counts:
                    # Gráfico de barras
                    fig_bar = px.bar(
                        x=list(categoria_counts.keys()),
                        y=list(categoria_counts.values()),
                        title="Fuentes por Categoría",
                        labels={'x': 'Categoría', 'y': 'Número de Fuentes'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Gráfico de pastel
                    fig_pie = px.pie(
                        values=list(categoria_counts.values()),
                        names=list(categoria_counts.keys()),
                        title="Distribución Porcentual por Categoría"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Distribución por idiomas
                st.subheader("🌍 Distribución por Idiomas")
                idiomas = [f.get('idioma', 'es') for f in fuentes]
                idioma_counts = Counter(idiomas)
                
                if idioma_counts:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Idiomas disponibles:**")
                        for idioma, count in idioma_counts.items():
                            st.write(f"- {idioma.upper()}: {count} fuentes")
                    
                    with col2:
                        fig_idiomas = px.pie(
                            values=list(idioma_counts.values()),
                            names=[lang.upper() for lang in idioma_counts.keys()],
                            title="Distribución por Idioma"
                        )
                        st.plotly_chart(fig_idiomas, use_container_width=True)
                
                # Línea de tiempo de fuentes agregadas
                st.subheader("📅 Línea de Tiempo - Fuentes Agregadas")
                
                fechas_agregadas = []
                for fuente in fuentes:
                    fecha = fuente.get('fecha_agregada')
                    if fecha:
                        try:
                            fechas_agregadas.append(datetime.strptime(fecha, "%Y-%m-%d"))
                        except:
                            continue
                
                if fechas_agregadas:
                    # Contar por mes
                    fechas_por_mes = {}
                    for fecha in fechas_agregadas:
                        mes_key = fecha.strftime("%Y-%m")
                        fechas_por_mes[mes_key] = fechas_por_mes.get(mes_key, 0) + 1
                    
                    if fechas_por_mes:
                        meses = sorted(fechas_por_mes.keys())
                        cantidades = [fechas_por_mes[mes] for mes in meses]
                        
                        fig_timeline = px.line(
                            x=meses,
                            y=cantidades,
                            title="Fuentes Agregadas por Mes",
                            labels={'x': 'Mes', 'y': 'Fuentes Agregadas'}
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Estado de las fuentes
                st.subheader("🔄 Estado de las Fuentes")
                estados = [f.get('estado', 'desconocido') for f in fuentes]
                estado_counts = Counter(estados)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for estado, count in estado_counts.items():
                        emoji = "✅" if estado == "activo" else "❌" if estado == "inactivo" else "⚠️"
                        st.write(f"{emoji} **{estado.title()}**: {count}")
                
                with col2:
                    if len(estado_counts) > 1:
                        fig_estados = px.pie(
                            values=list(estado_counts.values()),
                            names=[estado.title() for estado in estado_counts.keys()],
                            title="Estado de las Fuentes"
                        )
                        st.plotly_chart(fig_estados, use_container_width=True)
            
            # Información del sistema
            st.subheader("🔧 Información del Sistema")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configuración:**")
                st.write(f"- Base de datos: ChromaDB")
                st.write(f"- Modelo embedding: all-MiniLM-L6-v2")
                st.write(f"- Encoding: UTF-8")
                st.write(f"- Caché habilitado: ✅")
            
            with col2:
                st.write("**Rendimiento:**")
                if hasattr(st.session_state, 'ultima_busqueda'):
                    st.write(f"- Última búsqueda: {st.session_state.ultima_busqueda or 'Ninguna'}")
                st.write(f"- Búsquedas en caché: {len(st.session_state.resultados_cache)}")
                
                # Verificar salud del sistema
                try:
                    test_query = vector_db.collection.query(query_texts=["test"], n_results=1)
                    st.write("- Estado del sistema: ✅ Operativo")
                except:
                    st.write("- Estado del sistema: ⚠️ Verificar configuración")
            
            # Botones de mantenimiento
            st.subheader("🛠️ Herramientas de Mantenimiento")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Limpiar Caché", use_container_width=True):
                    st.session_state.resultados_cache = {}
                    st.success("✅ Caché limpiado")
            
            with col2:
                if st.button("🔄 Recargar Fuentes", use_container_width=True):
                    st.session_state.fuentes_cargadas = False
                    st.rerun()
            
            with col3:
                if st.button("📊 Actualizar Stats", use_container_width=True):
                    st.rerun()
            
            # Indexación masiva
            with st.expander("🚀 Indexación Masiva"):
                st.write("Re-indexar todas las fuentes activas:")
                
                if st.button("🔄 Re-indexar Todas las Fuentes", type="secondary"):
                    fuentes_activas = [f for f in fuentes if f.get('estado') == 'activo']
                    
                    if not fuentes_activas:
                        st.warning("No hay fuentes activas para indexar")
                    else:
                        with st.status(f"🚀 Indexando {len(fuentes_activas)} fuentes...", expanded=True) as status:
                            resultados_indexacion = []
                            
                            for i, fuente in enumerate(fuentes_activas):
                                try:
                                    st.write(f"📄 Indexando: {fuente['nombre']}")
                                    
                                    resultado = indexar_fuente_segura(vector_db, fuente['url'])
                                    resultados_indexacion.append({
                                        'fuente': fuente['nombre'],
                                        'exito': resultado.get('exito', False),
                                        'documentos': resultado.get('documentos_indexados', 0),
                                        'tiempo': resultado.get('tiempo_ejecucion', 0)
                                    })
                                    
                                    # Progreso
                                    progreso = (i + 1) / len(fuentes_activas)
                                    status.update(
                                        label=f"🚀 Progreso: {i+1}/{len(fuentes_activas)} ({progreso:.1%})",
                                        state="running"
                                    )
                                    
                                except Exception as e:
                                    logger.error(f"Error indexando {fuente['nombre']}: {str(e)}")
                                    resultados_indexacion.append({
                                        'fuente': fuente['nombre'],
                                        'exito': False,
                                        'error': str(e)
                                    })
                            
                            # Mostrar resultados
                            exitosas = sum(1 for r in resultados_indexacion if r['exito'])
                            fallidas = len(resultados_indexacion) - exitosas
                            total_docs = sum(r.get('documentos', 0) for r in resultados_indexacion)
                            
                            status.update(
                                label=f"✅ Indexación completada: {exitosas} exitosas, {fallidas} fallidas, {total_docs} documentos",
                                state="complete"
                            )
                            
                            # Tabla de resultados
                            if resultados_indexacion:
                                df_resultados = pd.DataFrame(resultados_indexacion)
                                st.dataframe(df_resultados, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error al cargar estadísticas: {str(e)}")
            logger.error(f"Error en estadísticas: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🔍 <strong>Buscador de Documentación Técnica</strong></p>
        <p>Sistema de búsqueda vectorial con soporte completo UTF-8</p>
        <p><small>Potenciado por ChromaDB + Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error crítico en la aplicación: {str(e)}")
        logger.error(f"Error crítico: {str(e)}")
        logger.error(traceback.format_exc())