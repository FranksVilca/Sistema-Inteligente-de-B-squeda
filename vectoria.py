import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Optional, Set
import time
import json
import os
from datetime import datetime
import chardet
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import locale
import sys
import unicodedata

# Configurar el sistema para usar UTF-8
if sys.platform.startswith('win'):
    try:
        locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'Spanish_Spain.1252')
else:
    try:
        locale.setlocale(locale.LC_ALL, 'es_ES.utf8')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# Configurar stdout para UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSiteVectorDB:
    def __init__(self, collection_name="documentacion_tecnica"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_model
        )
        
        # State management
        self.visited_urls = set()
        self.user_agent = "Mozilla/5.0 (compatible; BuscadorDocumentacion/1.0)"
        
        # Configuration files
        self.sources_file = "fuentes.json"
        self.indexados_file = "indexados.json"
        
        # Load initial data
        self.sources = self._cargar_fuentes()
        self.indexados = self._cargar_indexados()
        
        # Performance settings
        self.max_workers = 5  # For concurrent requests
        self.request_timeout = 15
        self.min_sleep_time = 0.5  # Be polite to servers

    def _detectar_codificacion(self, response_content: bytes) -> str:
        """Improved encoding detection with multiple fallbacks"""
        try:
            # Try chardet first
            detection = chardet.detect(response_content)
            if detection and detection['confidence'] > 0.8:
                return detection['encoding']
            
            # Common encodings to try
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
            
            for encoding in encodings:
                try:
                    response_content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
                    
            return 'utf-8'  # Final fallback
        except Exception:
            return 'utf-8'

    def _limpiar_texto(self, texto: str) -> str:
        """Corrige problemas de codificación y normaliza el texto"""
        if not texto:
            return ""
            
        # 1. Corregir Mojibake común
        correcciones = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã±': 'ñ', 'Ã¼': 'ü', 'Ã£': 'ã', 'Ã¶': 'ö', 'Ã¤': 'ä',
            'Ã¢': 'â', 'Ã»': 'û', 'Ã®': 'î', 'Ã¨': 'è',
            'Ãª': 'ê', 'Ã§': 'ç', 'Ã ': 'à', 'Ã´': 'ô', 'Ã¹': 'ù',
            'Ã«': 'ë', 'Ã¯': 'ï', 'Ã½': 'ý', 'Ã¾': 'þ', 'Ã¿': 'ÿ',
            'Ã': 'Á', 'Ã‰': 'É', 'Ã': 'Í', 'Ã"': 'Ó', 'Ãš': 'Ú',
            'Ã': 'Ñ', 'Ãœ': 'Ü', 'Â¿': '¿', 'Â¡': '¡', 'â‚¬': '€',
            'Ã‚': 'Â', 'Ãƒ': 'Ã', 'Ã„': 'Ä', 'Ã…': 'Å', 'Ã†': 'Æ',
            'Ã‡': 'Ç', 'Ãˆ': 'È', 'ÃŠ': 'Ê', 'Ã‹': 'Ë', 'ÃŒ': 'Ì',
            'ÃŽ': 'Î', 'Ã': 'Ò', 'Ã"': 'Ô', 'Ã•': 'Õ', 'Ã–': 'Ö',
            'Ã—': '×', 'Ã˜': 'Ø', 'Ã™': 'Ù', 'Ã›': 'Û', 'Ãž': 'Þ',
            'ÃŸ': 'ß', 'Ã¢': 'â', 'Ã¥': 'å', 'Ã¦': 'æ', 'Ã¨': 'è',
            'Ã¬': 'ì', 'Ã°': 'ð', 'Ã²': 'ò', 'Ãµ': 'õ', 'Ã·': '÷',
            'Ã¸': 'ø', 'Ã»': 'û'
        }
        
        for mal, bien in correcciones.items():
            texto = texto.replace(mal, bien)
        
        # 2. Normalización Unicode
        try:
            texto = unicodedata.normalize('NFKC', texto)
        except Exception:
            pass
        
        # 3. Eliminar caracteres problemáticos
        texto = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', texto)  # Control chars
        texto = re.sub(r'�', '', texto)  # Carácter de reemplazo Unicode
        
        # 4. Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto

    def _obtener_contenido(self, url: str) -> Optional[BeautifulSoup]:
        """Obtiene el contenido HTML de una URL"""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            response = requests.get(
                url, 
                headers=headers, 
                timeout=self.request_timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Intentar UTF-8 primero
            try:
                content = response.content.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback a detección automática
                encoding = self._detectar_codificacion(response.content)
                content = response.content.decode(encoding, errors='replace')
            
            return BeautifulSoup(content, 'html.parser')
            
        except Exception as e:
            logger.error(f"Error obteniendo {url}: {str(e)}")
            return None

    def _cargar_indexados(self) -> Dict[str, str]:
        """Load indexed sites with better error handling"""
        try:
            if os.path.exists(self.indexados_file):
                with open(self.indexados_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading indexados: {str(e)}")
        return {}

    def _guardar_indexados(self):
        """Save indexed sites with atomic write"""
        try:
            temp_file = f"{self.indexados_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.indexados, f, indent=2, ensure_ascii=False)
            
            # Atomic replace
            if os.path.exists(temp_file):
                os.replace(temp_file, self.indexados_file)
        except Exception as e:
            logger.error(f"Error saving indexados: {str(e)}")

    def _cargar_fuentes(self) -> List[Dict]:
        """Load sources with better default handling"""
        try:
            if os.path.exists(self.sources_file):
                with open(self.sources_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading sources: {str(e)}")
        
        # Default sources
        return [
            {
                "url": "https://docs.python.org/es/3/",
                "nombre": "Documentación Python (Español)",
                "descripcion": "Documentación oficial de Python en español",
                "categoria": "Python",
                "fecha_agregada": datetime.now().strftime("%Y-%m-%d"),
                "estado": "activo",
                "idioma": "es"
            }
        ]

    def _guardar_fuentes(self, fuentes=None):
        """Save sources with atomic write"""
        try:
            temp_file = f"{self.sources_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(fuentes or self.sources, f, indent=2, ensure_ascii=False)
            
            # Atomic replace
            if os.path.exists(temp_file):
                os.replace(temp_file, self.sources_file)
        except Exception as e:
            logger.error(f"Error saving sources: {str(e)}")

    def _es_url_valida(self, url: str, dominio_base: str) -> bool:
        """Improved URL validation"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
                
            # Check same domain
            if parsed.netloc != urlparse(dominio_base).netloc:
                return False
                
            # Check file extensions
            if any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.exe', '.dmg']):
                return False
                
            return True
        except:
            return False

    def _detectar_idioma(self, url: str) -> str:
        """Detecta el idioma basado en la URL"""
        if '/es/' in url or url.endswith('.es'):
            return 'es'
        elif '/en/' in url or url.endswith('.com'):
            return 'en'
        else:
            return 'es'  # Default

    def _extraer_contenido_principal(self, soup: BeautifulSoup) -> Optional[str]:
        """Improved main content extraction"""
        # Priority selectors for documentation content
        selectors = [
            'main', 'article', '.content', '.documentation', 
            '.doc-content', '.markdown-body', '.page-content',
            '#content', '#main-content', '.body'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove unwanted elements
                for tag in element(['script', 'style', 'nav', 'footer', 'aside', 'header', 'form']):
                    tag.decompose()
                
                text = element.get_text()
                return self._limpiar_texto(text)
        
        # Fallback to body if no specific content found
        body = soup.find('body')
        if body:
            for tag in body(['script', 'style', 'nav', 'footer', 'aside', 'header', 'form']):
                tag.decompose()
            return self._limpiar_texto(body.get_text())
        
        return None

    def extraer_pagina_individual(self, url: str) -> Optional[Dict]:
        """Improved single page extraction"""
        try:
            soup = self._obtener_contenido(url)
            if not soup:
                return None
                
            # Extract title
            title = ""
            if soup.title:
                title = self._limpiar_texto(soup.title.string or "")
            if not title:
                title = urlparse(url).path.split('/')[-1] or url
                
            # Extract main content
            content = self._extraer_contenido_principal(soup)
            if not content or len(content.split()) < 20:  # Minimum word count
                return None
                
            # Extract description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = self._limpiar_texto(meta_desc.get('content', ''))
                
            return {
                'url': url,
                'titulo': title,
                'contenido': content,
                'descripcion': description,
                'ultima_actualizacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.warning(f"Error processing {url}: {str(e)}")
            return None

    def _procesar_enlaces(self, url: str, soup: BeautifulSoup, dominio_base: str) -> Set[str]:
        """Extract and validate links from a page"""
        links = set()
        
        for link in soup.find_all('a', href=True):
            try:
                href = link['href'].split('#')[0]  # Remove fragments
                full_url = urljoin(url, href)
                
                if self._es_url_valida(full_url, dominio_base):
                    links.add(full_url)
            except:
                continue
                
        return links

    def rastrear_sitio_web(self, url_inicio: str, max_paginas: int = 0) -> List[Dict]:
        """Improved web crawler with concurrent processing"""
        dominio_base = urlparse(url_inicio).netloc
        to_visit = {url_inicio}
        visited = set()
        results = []
        
        logger.info(f"Starting crawl of {url_inicio} (max pages: {'unlimited' if max_paginas == 0 else max_paginas})")
        
        while to_visit and (max_paginas == 0 or len(results) < max_paginas):
            current_url = to_visit.pop()
            
            if current_url in visited:
                continue
                
            visited.add(current_url)
            
            # Process page
            page_data = self.extraer_pagina_individual(current_url)
            if page_data:
                results.append(page_data)
                
                # Get links if we need more pages
                if max_paginas == 0 or len(results) < max_paginas:
                    soup = self._obtener_contenido(current_url)
                    if soup:
                        new_links = self._procesar_enlaces(current_url, soup, dominio_base)
                        to_visit.update(new_links - visited)
            
            # Be polite
            time.sleep(self.min_sleep_time)
        
        logger.info(f"Crawl completed. Found {len(results)} pages.")
        return results

    def agregar_a_base_vectorial(self, documentos: List[Dict]):
        """Add documents to vector database with batching"""
        if not documentos:
            return
            
        batch_size = 100  # Chroma's recommended batch size
        for i in range(0, len(documentos), batch_size):
            batch = documentos[i:i+batch_size]
            
            ids = []
            docs = []
            metadatas = []
            
            for doc in batch:
                doc_id = f"doc_{abs(hash(doc['url']))}"
                ids.append(doc_id)
                docs.append(doc['contenido'])
                metadatas.append({
                    'url': doc['url'],
                    'titulo': doc['titulo'],
                    'fuente': urlparse(doc['url']).netloc,
                    'descripcion': doc.get('descripcion', ''),
                    'ultima_actualizacion': doc['ultima_actualizacion'],
                    'idioma': self._detectar_idioma(doc['url'])
                })
            
            try:
                self.collection.add(
                    documents=docs,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added batch of {len(batch)} documents to vector DB")
            except Exception as e:
                logger.error(f"Error adding batch to vector DB: {str(e)}")

    def indexar_fuente(self, url: str, max_paginas: int = 0, profundidad: int = None) -> Dict:
        """Index a single source with all possible pages"""
        # Si se proporciona profundidad, usar ese valor
        if profundidad is not None:
            max_paginas = profundidad
    
        start_time = time.time()
        documentos = self.rastrear_sitio_web(url, max_paginas)
    
        if documentos:
            self.agregar_a_base_vectorial(documentos)
            self.indexados[url] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._guardar_indexados()
        
            return {
                "exito": True,
                "documentos_indexados": len(documentos),
                "paginas_procesadas": len(documentos),
                "tiempo_ejecucion": round(time.time() - start_time, 2),
                "url": url
            }
        else:
            return {
                "exito": False,
                "mensaje": "No se pudo indexar el sitio",
                "url": url
            }
        
    def indexar_todas_fuentes(self, max_paginas_por_sitio: int = 0) -> Dict:
        """Index all sources with all possible pages"""
        resultados = {
            "total_fuentes": 0,
            "exitosas": 0,
            "fallidas": 0,
            "total_documentos": 0,
            "tiempo_total": 0,
            "detalles": []
        }
        
        fuentes_activas = [f for f in self.sources if f.get("estado") == "activo"]
        resultados["total_fuentes"] = len(fuentes_activas)
        start_time = time.time()
        
        logger.info(f"Starting full index of {len(fuentes_activas)} active sources")
        
        for fuente in fuentes_activas:
            try:
                result = self.indexar_fuente(fuente['url'], max_paginas_por_sitio,profundidad=10 )
                if result.get("exito"):
                    resultados["exitosas"] += 1
                    resultados["total_documentos"] += result.get("documentos_indexados", 0)
                    resultados["detalles"].append({
                        "fuente": fuente['nombre'],
                        "documentos": result.get("documentos_indexados", 0),
                        "estado": "exitoso",
                        "tiempo": result.get("tiempo_ejecucion", 0)
                    })
                    logger.info(f"Successfully indexed {fuente['nombre']}")
                else:
                    resultados["fallidas"] += 1
                    resultados["detalles"].append({
                        "fuente": fuente['nombre'],
                        "documentos": 0,
                        "estado": "fallido",
                        "error": result.get("mensaje", "Unknown error")
                    })
                    logger.warning(f"Failed to index {fuente['nombre']}")
            except Exception as e:
                resultados["fallidas"] += 1
                resultados["detalles"].append({
                    "fuente": fuente['nombre'],
                    "documentos": 0,
                    "estado": "error",
                    "error": str(e)
                })
                logger.error(f"Error indexing {fuente['nombre']}: {str(e)}")
            
            # Be polite between sources
            time.sleep(1)
        
        resultados["tiempo_total"] = round(time.time() - start_time, 2)
        logger.info(f"Index completed in {resultados['tiempo_total']} seconds")
        return resultados

    def obtener_fuentes(self) -> List[Dict]:
        """Obtiene la lista de fuentes configuradas"""
        return self.sources

    def agregar_fuente(self, url: str, nombre: str, descripcion: str = "", categoria: str = "General", idioma: str = "es") -> Dict:
        """Agrega una nueva fuente a la configuración"""
        try:
            # Validar URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "exito": False,
                    "mensaje": "URL inválida"
                }
            
            # Verificar si ya existe
            for fuente in self.sources:
                if fuente['url'] == url:
                    return {
                        "exito": False,
                        "mensaje": "La fuente ya existe"
                    }
            
            # Crear nueva fuente
            nueva_fuente = {
                "url": url,
                "nombre": nombre,
                "descripcion": descripcion,
                "categoria": categoria,
                "fecha_agregada": datetime.now().strftime("%Y-%m-%d"),
                "estado": "activo",
                "idioma": idioma
            }
            
            # Agregar a la lista
            self.sources.append(nueva_fuente)
            
            # Guardar en archivo
            self._guardar_fuentes()
            
            logger.info(f"Fuente agregada: {nombre} ({url})")
            
            return {
                "exito": True,
                "mensaje": "Fuente agregada exitosamente",
                "fuente": nueva_fuente
            }
            
        except Exception as e:
            logger.error(f"Error agregando fuente: {str(e)}")
            return {
                "exito": False,
                "mensaje": f"Error: {str(e)}"
            }

    def eliminar_fuente(self, url: str) -> Dict:
        """Elimina una fuente de la configuración"""
        try:
            # Buscar la fuente
            for i, fuente in enumerate(self.sources):
                if fuente['url'] == url:
                    # Eliminar de la lista
                    fuente_eliminada = self.sources.pop(i)
                    
                    # Guardar cambios
                    self._guardar_fuentes()
                    
                    # Eliminar de indexados si existe
                    if url in self.indexados:
                        del self.indexados[url]
                        self._guardar_indexados()
                    
                    logger.info(f"Fuente eliminada: {fuente_eliminada['nombre']}")
                    
                    return {
                        "exito": True,
                        "mensaje": "Fuente eliminada exitosamente"
                    }
            
            return {
                "exito": False,
                "mensaje": "Fuente no encontrada"
            }
            
        except Exception as e:
            logger.error(f"Error eliminando fuente: {str(e)}")
            return {
                "exito": False,
                "mensaje": f"Error: {str(e)}"
            }

    def actualizar_fuente(self, url: str, **kwargs) -> Dict:
        """Actualiza los datos de una fuente existente"""
        try:
            # Buscar la fuente
            for fuente in self.sources:
                if fuente['url'] == url:
                    # Actualizar campos permitidos
                    campos_permitidos = ['nombre', 'descripcion', 'categoria', 'estado', 'idioma']
                    for campo, valor in kwargs.items():
                        if campo in campos_permitidos:
                            fuente[campo] = valor
                    
                    # Guardar cambios
                    self._guardar_fuentes()
                    
                    logger.info(f"Fuente actualizada: {fuente['nombre']}")
                    
                    return {
                        "exito": True,
                        "mensaje": "Fuente actualizada exitosamente",
                        "fuente": fuente
                    }
            
            return {
                "exito": False,
                "mensaje": "Fuente no encontrada"
            }
            
        except Exception as e:
            logger.error(f"Error actualizando fuente: {str(e)}")
            return {
                "exito": False,
                "mensaje": f"Error: {str(e)}"
            }

    def obtener_estadisticas(self) -> Dict:
        """Obtiene estadísticas de la base de datos"""
        try:
            # Estadísticas básicas
            total_fuentes = len(self.sources)
            fuentes_activas = len([f for f in self.sources if f.get('estado') == 'activo'])
            fuentes_indexadas = len(self.indexados)
            
            # Estadísticas de ChromaDB
            try:
                collection_count = self.collection.count()
            except:
                collection_count = 0
            
            # Categorías
            categorias = {}
            for fuente in self.sources:
                cat = fuente.get('categoria', 'General')
                categorias[cat] = categorias.get(cat, 0) + 1
            
            # Idiomas
            idiomas = {}
            for fuente in self.sources:
                idioma = fuente.get('idioma', 'es')
                idiomas[idioma] = idiomas.get(idioma, 0) + 1
            
            return {
                "total_fuentes": total_fuentes,
                "fuentes_activas": fuentes_activas,
                "fuentes_indexadas": fuentes_indexadas,
                "documentos_total": collection_count,
                "categorias": categorias,
                "idiomas": idiomas,
                "ultima_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {
                "error": str(e)
            }

    def buscar_en_todos_sitios(self, consulta: str, n_resultados: int = 5) -> List[Dict]:
        """Busca en todos los sitios indexados"""
        try:
            results = self.collection.query(
                query_texts=[consulta],
                n_results=n_resultados
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'titulo': results['metadatas'][0][i]['titulo'],
                    'contenido': results['documents'][0][i][:500] + "..." if len(results['documents'][0][i]) > 500 else results['documents'][0][i],
                    'fuente': results['metadatas'][0][i]['fuente'],
                    'url': results['metadatas'][0][i]['url'],
                    'puntuacion': 1.0 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []

# Función de utilidad para testing
if __name__ == "__main__":
    vector_db = MultiSiteVectorDB()
    
    # Ejemplo de uso
    print("🔍 Fuentes actuales:")
    for fuente in vector_db.obtener_fuentes():
        print(f"- {fuente['nombre']}: {fuente['url']}")
    
    # Ejemplo de búsqueda
    consulta = "python android studio desarrollo"
    print(f"\n🔍 Buscando: '{consulta}'")
    resultados = vector_db.buscar_en_todos_sitios(consulta, n_resultados=3)
    
    for i, resultado in enumerate(resultados, 1):
        print(f"\n{i}. {resultado['titulo']}")
        print(f"   Fuente: {resultado['fuente']}")
        print(f"   Relevancia: {resultado['puntuacion']:.1%}")
        print(f"   Fragmento: {resultado['contenido'][:200]}...")