import os
import io
import base64
import sqlite3
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image
import torch

# Imports para os modelos
from transformers import AutoProcessor, AutoModelForCausalLM
# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import supervision as sv

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oil_detection_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gerenciador do banco de dados SQLite"""
    
    def __init__(self, db_path: str = "oil_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados com as tabelas necessárias"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de empresas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS empresas (
                empresa_id TEXT PRIMARY KEY,
                empresa_nome TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de coletores
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coletores (
                coletor_id TEXT PRIMARY KEY,
                empresa_id TEXT NOT NULL,
                coletor_descricao TEXT,
                coletor_localizacao TEXT,
                status TEXT DEFAULT 'ativo',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (empresa_id) REFERENCES empresas (empresa_id)
            )
        ''')
        
        # Tabela principal de detecções
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deteccoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                empresa_id TEXT NOT NULL,
                coletor_id TEXT NOT NULL,
                timestamp_coleta TIMESTAMP NOT NULL,
                timestamp_processamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modelo_usado TEXT NOT NULL,
                imagem_original_path TEXT NOT NULL,
                imagem_processada_path TEXT,
                confidence_yolo REAL,
                objects_detected_yolo INTEGER DEFAULT 0,
                FOREIGN KEY (empresa_id) REFERENCES empresas (empresa_id),
                FOREIGN KEY (coletor_id) REFERENCES coletores (coletor_id)
            )
        ''')
        
        # Tabela para resultados do Florence-2
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS florence_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deteccao_id INTEGER NOT NULL,
                caption TEXT,
                detailed_caption TEXT,
                more_detailed_caption TEXT,
                dense_region_caption TEXT,
                object_detection TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (deteccao_id) REFERENCES deteccoes (id)
            )
        ''')
        
        # Tabela para resultados do GroundingDINO
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grounding_dino_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deteccao_id INTEGER NOT NULL,
                oil_spill_detected BOOLEAN NOT NULL,
                max_confidence REAL,
                total_detections INTEGER DEFAULT 0,
                bounding_boxes TEXT,
                logits TEXT,
                phrases TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (deteccao_id) REFERENCES deteccoes (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_empresa(self, empresa_id: str, empresa_nome: str):
        """Insere ou atualiza empresa"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO empresas (empresa_id, empresa_nome)
            VALUES (?, ?)
        ''', (empresa_id, empresa_nome))
        conn.commit()
        conn.close()
    
    def insert_coletor(self, coletor_id: str, empresa_id: str, descricao: str, localizacao: str):
        """Insere ou atualiza coletor"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO coletores 
            (coletor_id, empresa_id, coletor_descricao, coletor_localizacao)
            VALUES (?, ?, ?, ?)
        ''', (coletor_id, empresa_id, descricao, localizacao))
        conn.commit()
        conn.close()
    
    def insert_deteccao(self, dados_deteccao: Dict) -> int:
        """Insere uma nova detecção e retorna o ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO deteccoes 
            (empresa_id, coletor_id, timestamp_coleta, modelo_usado, 
             imagem_original_path, confidence_yolo, objects_detected_yolo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            dados_deteccao['empresa_id'],
            dados_deteccao['coletor_id'],
            dados_deteccao['timestamp_coleta'],
            dados_deteccao['modelo_usado'],
            dados_deteccao['imagem_original_path'],
            dados_deteccao.get('confidence_yolo'),
            dados_deteccao.get('objects_detected_yolo', 0)
        ))
        
        deteccao_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return deteccao_id
    
    def insert_florence_result(self, deteccao_id: int, florence_data: Dict):
        """Insere resultado do Florence-2"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO florence_results 
            (deteccao_id, caption, detailed_caption, more_detailed_caption,
             dense_region_caption, object_detection)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            deteccao_id,
            florence_data.get('<CAPTION>', ''),
            florence_data.get('<DETAILED_CAPTION>', ''),
            florence_data.get('<MORE_DETAILED_CAPTION>', ''),
            json.dumps(florence_data.get('<DENSE_REGION_CAPTION>', {})),
            json.dumps(florence_data.get('<OD>', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def insert_grounding_result(self, deteccao_id: int, grounding_data: Dict):
        """Insere resultado do GroundingDINO"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        detection_data = grounding_data.get('detection_data', {})
        logits = detection_data.get('logits', [])
        
        cursor.execute('''
            INSERT INTO grounding_dino_results 
            (deteccao_id, oil_spill_detected, max_confidence, total_detections,
             bounding_boxes, logits, phrases)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            deteccao_id,
            len(logits) > 0,
            float(max(logits)) if logits else 0.0,
            len(logits),
            json.dumps(detection_data.get('boxes', []).tolist() if hasattr(detection_data.get('boxes', []), 'tolist') else detection_data.get('boxes', [])),
            json.dumps(logits.tolist() if hasattr(logits, 'tolist') else logits),
            json.dumps(detection_data.get('phrases', []))
        ))
        
        conn.commit()
        conn.close()

class FileManager:
    """Gerenciador de arquivos e diretórios"""
    
    def __init__(self, base_path: str = "Archived"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def create_directory_structure(self, empresa_id: str, coletor_id: str, modelo: str) -> Tuple[Path, Path]:
        """Cria estrutura de diretórios e retorna paths para raw e processed"""
        base_dir = self.base_path / empresa_id / coletor_id / modelo
        raw_dir = base_dir / "raw_image"
        processed_dir = base_dir / "processed"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        return raw_dir, processed_dir
    
    def generate_filename(self, empresa_id: str, coletor_id: str, timestamp: str, 
                         modelo: str, objects_count: int, prefix: str = "RAW") -> str:
        """Gera nome do arquivo seguindo o padrão especificado"""
        return f"{prefix}-{empresa_id}-{coletor_id}-{timestamp}-{modelo}-{objects_count}.jpg"
    
    def save_image(self, image: Image.Image, filepath: Path) -> bool:
        """Salva imagem no caminho especificado"""
        try:
            image.save(filepath, "JPEG", quality=95)
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar imagem {filepath}: {e}")
            return False

class ModelManager:
    """Gerenciador dos modelos Florence-2 e GroundingDINO"""
    
    def __init__(self):
        self.florence_processor = None
        self.florence_model = None
        self.grounding_model = None
        self._load_models()
    
    def _load_models(self):
        """Carrega os modelos na inicialização"""
        try:
            # Carregar Florence-2
            logger.info("Loading Florence-2 model...")
            self.florence_processor = AutoProcessor.from_pretrained(
                "microsoft/florence-2-base-ft", trust_remote_code=True
            )
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/florence-2-base-ft", trust_remote_code=True
            )
            logger.info("Florence-2 model loaded successfully")
            
            # Carregar GroundingDINO (comentado por enquanto - você precisa ajustar os imports)
            # logger.info("Loading GroundingDINO model...")
            # self.grounding_model = load_model(CONFIG_PATH, WEIGHTS_PATH, device='cpu')
            # logger.info("GroundingDINO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
    
    def generate_florence_captions(self, image: Image.Image, max_new_tokens: int = 1024) -> Dict:
        """Gera captions usando Florence-2"""
        if not self.florence_model or not self.florence_processor:
            return {}
        
        prompts = [
            "<CAPTION>",
            "<DETAILED_CAPTION>", 
            "<MORE_DETAILED_CAPTION>",
            "<DENSE_REGION_CAPTION>",
            "<OD>",
        ]
        
        results = {}
        for prompt in prompts:
            try:
                inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt")
                generated_ids = self.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens
                )
                generated_text = self.florence_processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed_answer = self.florence_processor.post_process_generation(
                    generated_text,
                    task=prompt,
                    image_size=(image.width, image.height)
                )
                results[prompt] = parsed_answer.get(prompt, "")
            except Exception as e:
                logger.error(f"Erro no Florence-2 com prompt {prompt}: {e}")
                results[prompt] = ""
        
        return results
    
    def process_grounding_detection(self, image: Image.Image, 
                                  text_prompt: str = "oil spill",
                                  box_threshold: float = 0.3,
                                  text_threshold: float = 0.0) -> Dict:
        """Processa detecção com GroundingDINO"""
        # Por enquanto retorna um mock - você precisa implementar com o GroundingDINO
        return {
            'annotated_image': np.array(image),
            'detection_data': {
                'boxes': np.array([]),
                'logits': np.array([]),
                'phrases': [],
                'image_dimensions': (image.height, image.width)
            }
        }

class OilDetectionServer:
    """Servidor principal para detecção de óleo"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.db_manager = DatabaseManager()
        self.file_manager = FileManager()
        self.model_manager = ModelManager()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configura as rotas do Flask"""
        
        @self.app.route('/oil_detection', methods=['POST'])
        def oil_detection_endpoint():
            return self.process_oil_detection()
        
        @self.app.route('/processar', methods=['POST'])
        def processar():
            return self.legacy_processar()
        
        @self.app.route('/skf_analysis', methods=['POST'])
        def handle_skf_analysis():
            return self.skf_analysis_endpoint()
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            return self.get_detection_stats()
    
    def process_oil_detection(self):
        """Endpoint principal para processamento de detecção de óleo"""
        try:
            data = request.get_json()
            
            # Validar dados obrigatórios
            required_fields = ['imagem_base64', 'empresa_id', 'coletor_id', 'empresa_nome']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        "status": "error", 
                        "message": f"Campo obrigatório ausente: {field}"
                    }), 400
            
            # Extrair dados
            empresa_id = data['empresa_id']
            empresa_nome = data['empresa_nome']
            coletor_id = data['coletor_id']
            coletor_descricao = data.get('coletor_descricao', '')
            coletor_localizacao = data.get('coletor_localizacao', '')
            modelo_usado = data.get('modelo_usado', 'oil_spill')
            confidence_yolo = data.get('confidence_threshold')
            timestamp_coleta = data.get('timestamp', datetime.now().isoformat())
            
            # Converter imagem
            imagem_bytes = base64.b64decode(data['imagem_base64'])
            image = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")
            
            # Inserir/atualizar empresa e coletor
            self.db_manager.insert_empresa(empresa_id, empresa_nome)
            self.db_manager.insert_coletor(coletor_id, empresa_id, coletor_descricao, coletor_localizacao)
            
            # Criar estrutura de diretórios
            raw_dir, processed_dir = self.file_manager.create_directory_structure(
                empresa_id, coletor_id, modelo_usado
            )
            
            # Processar com Florence-2
            logger.info(f"Processing image with Florence-2 for {empresa_id}/{coletor_id}")
            florence_results = self.model_manager.generate_florence_captions(image)
            
            # Processar com GroundingDINO
            logger.info(f"Processing image with GroundingDINO for {empresa_id}/{coletor_id}")
            grounding_results = self.model_manager.process_grounding_detection(image)
            
            # Determinar quantidade de objetos detectados
            objects_detected = len(grounding_results['detection_data']['logits'])
            
            # Gerar nomes de arquivo
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            raw_filename = self.file_manager.generate_filename(
                empresa_id, coletor_id, timestamp_str, modelo_usado, objects_detected, "RAW"
            )
            processed_filename = self.file_manager.generate_filename(
                empresa_id, coletor_id, timestamp_str, modelo_usado, objects_detected, "PROC"
            )
            
            # Salvar imagens
            raw_path = raw_dir / raw_filename
            processed_path = processed_dir / processed_filename
            
            self.file_manager.save_image(image, raw_path)
            
            # Salvar imagem processada se houver detecções
            if 'annotated_image' in grounding_results:
                processed_image = Image.fromarray(grounding_results['annotated_image'].astype('uint8'))
                self.file_manager.save_image(processed_image, processed_path)
            
            # Inserir no banco de dados
            dados_deteccao = {
                'empresa_id': empresa_id,
                'coletor_id': coletor_id,
                'timestamp_coleta': timestamp_coleta,
                'modelo_usado': modelo_usado,
                'imagem_original_path': str(raw_path),
                'confidence_yolo': confidence_yolo,
                'objects_detected_yolo': objects_detected
            }
            
            deteccao_id = self.db_manager.insert_deteccao(dados_deteccao)
            self.db_manager.insert_florence_result(deteccao_id, florence_results)
            self.db_manager.insert_grounding_result(deteccao_id, grounding_results)
            
            # Preparar resposta
            response = {
                "status": "success",
                "deteccao_id": deteccao_id,
                "objects_detected": objects_detected,
                "oil_spill_detected": objects_detected > 0,
                "max_confidence": float(max(grounding_results['detection_data']['logits'])) if grounding_results['detection_data']['logits'] else 0.0,
                "florence_caption": florence_results.get('<CAPTION>', ''),
                "files_saved": {
                    "raw_image": str(raw_path),
                    "processed_image": str(processed_path) if objects_detected > 0 else None
                },
                "timestamp_processamento": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed detection {deteccao_id} for {empresa_id}/{coletor_id}")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            return jsonify({
                "status": "error",
                "message": f"Erro interno do servidor: {str(e)}"
            }), 500
    
    def legacy_processar(self):
        """Endpoint legacy para compatibilidade"""
        dados = request.get_json()
        imagem_b64 = dados.get('imagem_base64')
        nome_imagem = dados.get('nome_imagem', 'imagem_recebida.jpg')
        texto = dados.get('texto', '')
        
        # Salva a imagem temporariamente
        with open(nome_imagem, "wb") as f:
            f.write(base64.b64decode(imagem_b64))
        
        tamanho_kb = os.path.getsize(nome_imagem) / 1024
        resposta = f"{nome_imagem}: {tamanho_kb:.1f}kb | Texto recebido: {texto}"
        
        return jsonify({"resposta": resposta})
    
    def skf_analysis_endpoint(self):
        """Endpoint para análise SKF (Florence-2)"""
        try:
            data = request.json
            
            imagem_base64 = data.get("imagem_base64")
            if not imagem_base64:
                return jsonify({"status": "error", "message": "Imagem base64 não fornecida"}), 400
            
            # Converter base64 para imagem
            imagem_bytes = base64.b64decode(imagem_base64)
            image_stream = io.BytesIO(imagem_bytes)
            image = Image.open(image_stream).convert("RGB")
            
            # Processar com Florence-2
            result = self.model_manager.generate_florence_captions(image)
            
            return jsonify({
                "status": "success",
                "return": result
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Erro no servidor: {str(e)}"
            }), 500
    
    def get_detection_stats(self):
        """Retorna estatísticas das detecções"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Estatísticas gerais
            cursor.execute("SELECT COUNT(*) FROM deteccoes")
            total_deteccoes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM grounding_dino_results WHERE oil_spill_detected = 1")
            deteccoes_positivas = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT empresa_id) FROM deteccoes")
            total_empresas = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT coletor_id) FROM deteccoes")
            total_coletores = cursor.fetchone()[0]
            
            conn.close()
            
            return jsonify({
                "status": "success",
                "stats": {
                    "total_deteccoes": total_deteccoes,
                    "deteccoes_positivas": deteccoes_positivas,
                    "taxa_deteccao": (deteccoes_positivas / total_deteccoes * 100) if total_deteccoes > 0 else 0,
                    "total_empresas": total_empresas,
                    "total_coletores": total_coletores
                }
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    def run(self, host="0.0.0.0", port=8080, debug=False):
        """Executa o servidor"""
        logger.info(f"Starting Oil Detection Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Inicialização do servidor
if __name__ == '__main__':
    server = OilDetectionServer()
    server.run(debug=True)