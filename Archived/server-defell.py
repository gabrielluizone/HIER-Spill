#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERVIDOR COMPLETO DE ANÁLISE E GERENCIAMENTO DE DADOS
====================================================
Sistema robusto para recepção, processamento e armazenamento de dados de detecção
com análise Florence-2 integrada e banco de dados por empresa.

Autor: Gabriel L Skaftell
Data: 2025
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, g
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from PIL import Image

import traceback
import threading
import logging
import sqlite3
import hashlib
import base64
import shutil
import torch
import uuid
import json
import time
import os
import io

# Imports do Florence-2 (assumindo que já estão configurados)
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    FLORENCE_AVAILABLE = True
except ImportError:
    FLORENCE_AVAILABLE = False
    print("⚠️  Florence-2 não disponível. Instale: pip install transformers torch")

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

class Config:
    # Diretórios
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "images" 
    DATABASES_DIR = DATA_DIR / "databases"
    LOGS_DIR = DATA_DIR / "logs"
    
    # Servidor
    HOST = "0.0.0.0"
    PORT = 8080
    DEBUG = False
    
    # Florence-2
    FLORENCE_MODEL_NAME = "microsoft/Florence-2-large"
    FLORENCE_PROMPTS = [
        "<CAPTION>",           # Descrição geral da imagem
        "<DETAILED_CAPTION>",  # Descrição detalhada
        "<OD>"                 # Object Detection
    ]
    
    # Logs
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Limpeza automática
    AUTO_CLEANUP_DAYS = 30
    MAX_DB_SIZE_MB = 500

# Criar diretórios necessários
for directory in [Config.DATA_DIR, Config.IMAGES_DIR, Config.DATABASES_DIR, Config.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SISTEMA DE LOGGING AVANÇADO
# =============================================================================

class LoggerManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.setup_logging()
            self.initialized = True
    
    def setup_logging(self):
        """Configura sistema de logging robusto"""
        # Log principal
        main_log = Config.LOGS_DIR / f"servidor_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configurar logging
        logging.basicConfig(
            level=Config.LOG_LEVEL,
            format=Config.LOG_FORMAT,
            handlers=[
                logging.FileHandler(main_log, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Loggers específicos
        self.main_logger = logging.getLogger('SERVIDOR')
        self.db_logger = logging.getLogger('DATABASE')
        self.florence_logger = logging.getLogger('FLORENCE')
        self.api_logger = logging.getLogger('API')
        
        self.main_logger.info("🚀 Sistema de logging inicializado")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Retorna logger específico"""
        return logging.getLogger(name)

# Instância global
logger_manager = LoggerManager()
main_logger = logger_manager.get_logger('SERVIDOR')

# =============================================================================
# MODELOS DE DADOS
# =============================================================================

@dataclass
class EmpresaInfo:
    empresa_id: int
    empresa_nome: str
    coletor_id: int
    modelo_id: int
    coletor_descricao: str
    localizacao: str

@dataclass
class ProcessingResult:
    success: bool
    message: str
    processing_id: Optional[str] = None
    florence_results: Optional[Dict] = None
    errors: Optional[List[str]] = None

# =============================================================================
# GERENCIADOR DE BANCO DE DADOS POR EMPRESA
# =============================================================================

class DatabaseManager:
    def __init__(self):
        self.logger = logger_manager.get_logger('DATABASE')
        self.connections = {}
        self._lock = threading.Lock()
    
    def get_db_path(self, empresa_id: int) -> Path:
        """Retorna caminho do banco da empresa"""
        return Config.DATABASES_DIR / f"empresa_{empresa_id}.db"
    
    @contextmanager
    def get_connection(self, empresa_id: int):
        """Context manager para conexões thread-safe"""
        db_path = self.get_db_path(empresa_id)
        
        with self._lock:
            try:
                conn = sqlite3.connect(str(db_path), timeout=30.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                yield conn
            finally:
                conn.close()
    
    def initialize_database(self, empresa_id: int) -> bool:
        """Inicializa banco de dados da empresa"""
        try:
            with self.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                # Tabela principal de processamentos
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processamentos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        processing_id TEXT UNIQUE NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        empresa_id INTEGER NOT NULL,
                        empresa_nome TEXT NOT NULL,
                        coletor_id INTEGER NOT NULL,
                        modelo_id INTEGER NOT NULL,
                        coletor_descricao TEXT,
                        localizacao TEXT,
                        
                        -- Imagens
                        name_imagem_original TEXT NOT NULL,
                        name_imagem_processada TEXT NOT NULL,
                        path_imagem_original TEXT NOT NULL,
                        path_imagem_processada TEXT NOT NULL,
                        
                        -- Dados de detecção
                        detection_data TEXT, -- JSON
                        metadata TEXT,       -- JSON
                        confidence_threshold REAL,
                        detections_count INTEGER,
                        
                        -- Análise Florence-2
                        florence_caption TEXT,
                        florence_detailed_caption TEXT,
                        florence_objects TEXT, -- JSON
                        florence_processed_at DATETIME,
                        florence_success BOOLEAN DEFAULT 0,
                        
                        -- Status
                        status TEXT DEFAULT 'received',
                        processing_time_ms INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabela de logs por empresa
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS logs_empresa (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        processing_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        FOREIGN KEY (processing_id) REFERENCES processamentos(processing_id)
                    )
                """)
                
                # Tabela de estatísticas
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS estatisticas (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data DATE NOT NULL,
                        total_processamentos INTEGER DEFAULT 0,
                        total_deteccoes INTEGER DEFAULT 0,
                        tempo_medio_processamento REAL DEFAULT 0,
                        taxa_sucesso REAL DEFAULT 0,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(data)
                    )
                """)
                
                # Índices para performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_id ON processamentos(processing_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON processamentos(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_empresa ON processamentos(empresa_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON processamentos(status)")
                
                conn.commit()
                self.logger.info(f"✅ Banco de dados inicializado para empresa {empresa_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar banco empresa {empresa_id}: {e}")
            return False
    
    def save_processing(self, empresa_id: int, processing_data: Dict) -> bool:
        """Salva dados de processamento no banco"""
        try:
            with self.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO processamentos (
                        processing_id, empresa_id, empresa_nome, coletor_id, modelo_id,
                        coletor_descricao, localizacao, name_imagem_original, name_imagem_processada,
                        path_imagem_original, path_imagem_processada, detection_data, metadata,
                        confidence_threshold, detections_count, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    processing_data['processing_id'],
                    processing_data['empresa_id'],
                    processing_data['empresa_nome'],
                    processing_data['coletor_id'],
                    processing_data['modelo_id'],
                    processing_data['coletor_descricao'],
                    processing_data['localizacao'],
                    processing_data['name_imagem_original'],
                    processing_data['name_imagem_processada'],
                    processing_data['path_imagem_original'],
                    processing_data['path_imagem_processada'],
                    json.dumps(processing_data.get('detection_data')),
                    json.dumps(processing_data.get('metadata')),
                    processing_data.get('confidence_threshold'),
                    processing_data.get('detections_count'),
                    'received'
                ))
                
                conn.commit()
                self.logger.info(f"✅ Processamento salvo: {processing_data['processing_id']}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar processamento: {e}")
            return False
    
    def update_florence_results(self, empresa_id: int, processing_id: str, florence_data: Dict) -> bool:
        """Atualiza resultados do Florence-2"""
        try:
            with self.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE processamentos SET
                        florence_caption = ?,
                        florence_detailed_caption = ?,
                        florence_objects = ?,
                        florence_processed_at = CURRENT_TIMESTAMP,
                        florence_success = 1,
                        status = 'processed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE processing_id = ?
                """, (
                    florence_data.get('caption'),
                    florence_data.get('detailed_caption'),
                    json.dumps(florence_data.get('objects')),
                    processing_id
                ))
                
                conn.commit()
                self.logger.info(f"✅ Florence-2 atualizado: {processing_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao atualizar Florence-2: {e}")
            return False
    
    def get_processamentos(self, empresa_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Consulta processamentos da empresa"""
        try:
            with self.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM processamentos 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao consultar processamentos: {e}")
            return []
    
    def get_statistics(self, empresa_id: int, days: int = 30) -> Dict:
        """Retorna estatísticas da empresa"""
        try:
            with self.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                # Estatísticas básicas
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_processamentos,
                        SUM(detections_count) as total_deteccoes,
                        AVG(processing_time_ms) as tempo_medio,
                        SUM(CASE WHEN florence_success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as taxa_sucesso
                    FROM processamentos 
                    WHERE timestamp >= datetime('now', '-{} days')
                """.format(days))
                
                stats = dict(cursor.fetchone())
                
                # Processamentos por dia
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as data,
                        COUNT(*) as count
                    FROM processamentos 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY data DESC
                """.format(days))
                
                daily_stats = [dict(row) for row in cursor.fetchall()]
                stats['daily_stats'] = daily_stats
                
                return stats
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao obter estatísticas: {e}")
            return {}

# =============================================================================
# PROCESSADOR FLORENCE-2 AVANÇADO
# =============================================================================

class FlorenceProcessor:
    def __init__(self):
        self.logger = logger_manager.get_logger('FLORENCE')
        self.model = None
        self.processor = None
        self.model_loaded = False
        self._lock = threading.Lock()
        
        if FLORENCE_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Carrega modelo Florence-2"""
        try:
            self.logger.info("🔄 Carregando modelo Florence-2...")
            
            self.processor = AutoProcessor.from_pretrained(
                Config.FLORENCE_MODEL_NAME, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.FLORENCE_MODEL_NAME, 
                trust_remote_code=True
            )
            
            # Mover para CPU (mais estável)
            self.model = self.model.to('cpu')
            self.model_loaded = True
            
            self.logger.info("✅ Modelo Florence-2 carregado com sucesso")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar Florence-2: {e}")
            self.model_loaded = False
    
    def process_image(self, image: Image.Image, processing_id: str) -> Dict:
        """Processa imagem com Florence-2"""
        if not self.model_loaded:
            return {"error": "Modelo Florence-2 não disponível"}
        
        results = {}
        
        with self._lock:
            try:
                self.logger.info(f"🔄 Processando imagem com Florence-2: {processing_id}")
                
                for prompt in Config.FLORENCE_PROMPTS:
                    try:
                        # Processar entrada
                        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                        
                        # Gerar resultado
                        generated_ids = self.model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            early_stopping=False,
                            do_sample=False,
                            num_beams=3,
                        )
                        
                        # Decodificar resultado
                        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                        parsed_answer = self.processor.post_process_generation(
                            generated_text, 
                            task=prompt, 
                            image_size=(image.width, image.height)
                        )
                        
                        # Mapear resultados
                        if prompt == "<CAPTION>":
                            results['caption'] = parsed_answer.get(prompt, "")
                        elif prompt == "<DETAILED_CAPTION>":
                            results['detailed_caption'] = parsed_answer.get(prompt, "")
                        elif prompt == "<OD>":
                            results['objects'] = parsed_answer.get(prompt, {})
                        
                        self.logger.debug(f"✅ Prompt {prompt} processado")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Erro no prompt {prompt}: {e}")
                        results[f'error_{prompt}'] = str(e)
                
                self.logger.info(f"✅ Florence-2 concluído: {processing_id}")
                return results
                
            except Exception as e:
                self.logger.error(f"❌ Erro geral no Florence-2: {e}")
                return {"error": str(e)}

# =============================================================================
# GERENCIADOR DE ARQUIVOS
# =============================================================================

class FileManager:
    def __init__(self):
        self.logger = logger_manager.get_logger('FILES')
    
    def get_empresa_dir(self, empresa_id: int) -> Path:
        """Retorna diretório da empresa"""
        empresa_dir = Config.IMAGES_DIR / f"empresa_{empresa_id}"
        empresa_dir.mkdir(parents=True, exist_ok=True)
        return empresa_dir
    
    def get_daily_dir(self, empresa_id: int, date: datetime = None) -> Path:
        """Retorna diretório do dia"""
        if date is None:
            date = datetime.now()
        
        daily_dir = self.get_empresa_dir(empresa_id) / date.strftime("%Y-%m-%d")
        daily_dir.mkdir(parents=True, exist_ok=True)
        return daily_dir
    
    def save_image(self, empresa_id: int, image_name: str, image_b64: str, processing_id: str) -> Optional[str]:
        """Salva imagem decodificada"""
        try:
            # Decodificar base64
            image_bytes = base64.b64decode(image_b64)
            
            # Diretório do dia
            daily_dir = self.get_daily_dir(empresa_id)
            
            # Nome único com processing_id
            safe_name = f"{processing_id}_{image_name}"
            image_path = daily_dir / safe_name
            
            # Salvar arquivo
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            self.logger.info(f"✅ Imagem salva: {image_path}")
            return str(image_path)
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar imagem {image_name}: {e}")
            return None
    
    def load_image_as_pil(self, image_path: str) -> Optional[Image.Image]:
        """Carrega imagem como PIL"""
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar imagem {image_path}: {e}")
            return None
    
    def cleanup_old_files(self, days: int = 30):
        """Remove arquivos antigos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            removed_count = 0
            
            for empresa_dir in Config.IMAGES_DIR.iterdir():
                if not empresa_dir.is_dir() or not empresa_dir.name.startswith('empresa_'):
                    continue
                
                for date_dir in empresa_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    try:
                        dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                        if dir_date < cutoff_date:
                            import shutil
                            shutil.rmtree(date_dir)
                            removed_count += 1
                            self.logger.info(f"🗑️  Removido: {date_dir}")
                    except ValueError:
                        continue
            
            self.logger.info(f"✅ Limpeza concluída: {removed_count} diretórios removidos")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na limpeza: {e}")

# =============================================================================
# SERVIDOR PRINCIPAL
# =============================================================================

class ServidorCompleto:
    def __init__(self):
        self.logger = main_logger
        self.db_manager = DatabaseManager()
        self.florence_processor = FlorenceProcessor()
        self.file_manager = FileManager()
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Background tasks
        self.start_background_tasks()
        
        self.logger.info("🚀 Servidor Completo inicializado")
    
    def setup_routes(self):
        """Configura todas as rotas da API"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check do servidor"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "florence_available": self.florence_processor.model_loaded,
                "version": "2.0.0"
            })
        
        @self.app.route('/process', methods=['POST'])
        def process_data():
            """Endpoint principal para receber dados"""
            start_time = time.time()
            
            try:
                # Validar dados
                data = request.json
                if not data:
                    return jsonify({"status": 0, "message": "Dados não fornecidos"}), 400
                
                # Extrair informações da empresa
                empresa_info = data.get('empresa_info', {})
                empresa_id = empresa_info.get('empresa_id')
                
                if not empresa_id:
                    return jsonify({"status": 0, "message": "empresa_id obrigatório"}), 400
                
                # Gerar ID único para processamento
                processing_id = self.generate_processing_id(data)
                
                self.logger.info(f"🔄 Iniciando processamento: {processing_id} | Empresa: {empresa_id}")
                
                # Inicializar banco da empresa se necessário
                self.db_manager.initialize_database(empresa_id)
                
                # Salvar imagens
                original_path = None
                processed_path = None
                
                if data.get('imagem_original_base64'):
                    original_path = self.file_manager.save_image(
                        empresa_id, 
                        data.get('name_imagem_original', 'original.jpg'),
                        data['imagem_original_base64'],
                        processing_id
                    )
                
                if data.get('imagem_processada_base64'):
                    processed_path = self.file_manager.save_image(
                        empresa_id,
                        data.get('name_imagem_processada', 'processed.jpg'),
                        data['imagem_processada_base64'],
                        processing_id
                    )
                
                # Preparar dados para banco
                processing_data = {
                    'processing_id': processing_id,
                    'empresa_id': empresa_id,
                    'empresa_nome': empresa_info.get('empresa_nome', ''),
                    'coletor_id': empresa_info.get('coletor_id'),
                    'modelo_id': empresa_info.get('modelo_id'),
                    'coletor_descricao': empresa_info.get('coletor_descricao', ''),
                    'localizacao': empresa_info.get('localizacao', ''),
                    'name_imagem_original': data.get('name_imagem_original', ''),
                    'name_imagem_processada': data.get('name_imagem_processada', ''),
                    'path_imagem_original': original_path or '',
                    'path_imagem_processada': processed_path or '',
                    'detection_data': data.get('detection_data'),
                    'metadata': data.get('metadata'),
                    'confidence_threshold': data.get('confidence_threshold'),
                    'detections_count': data.get('detections_count', 0)
                }
                
                # Salvar no banco
                if not self.db_manager.save_processing(empresa_id, processing_data):
                    return jsonify({"status": 0, "message": "Erro ao salvar dados"}), 500
                
                # Processar com Florence-2 em background
                threading.Thread(
                    target=self.process_florence_background,
                    args=(empresa_id, processing_id, original_path or processed_path),
                    daemon=True
                ).start()
                
                # Calcular tempo de processamento
                processing_time = int((time.time() - start_time) * 1000)
                
                self.logger.info(f"✅ Processamento concluído: {processing_id} ({processing_time}ms)")
                
                return jsonify({
                    "status": 1,
                    "return": 1,
                    "message": "Dados processados com sucesso",
                    "processing_id": processing_id,
                    "processing_time_ms": processing_time
                })
                
            except Exception as e:
                self.logger.error(f"❌ Erro no processamento: {e}")
                return jsonify({
                    "status": 0, 
                    "message": f"Erro interno: {str(e)}"
                }), 500
        
        @self.app.route('/consulta/<int:empresa_id>', methods=['GET'])
        def consultar_dados(empresa_id):
            """Consulta dados da empresa"""
            try:
                limit = int(request.args.get('limit', 100))
                offset = int(request.args.get('offset', 0))
                
                processamentos = self.db_manager.get_processamentos(empresa_id, limit, offset)
                
                return jsonify({
                    "status": 1,
                    "empresa_id": empresa_id,
                    "total": len(processamentos),
                    "processamentos": processamentos
                })
                
            except Exception as e:
                self.logger.error(f"❌ Erro na consulta: {e}")
                return jsonify({"status": 0, "message": str(e)}), 500
        
        @self.app.route('/estatisticas/<int:empresa_id>', methods=['GET'])
        def obter_estatisticas(empresa_id):
            """Estatísticas da empresa"""
            try:
                days = int(request.args.get('days', 30))
                stats = self.db_manager.get_statistics(empresa_id, days)
                
                return jsonify({
                    "status": 1,
                    "empresa_id": empresa_id,
                    "period_days": days,
                    "statistics": stats
                })
                
            except Exception as e:
                self.logger.error(f"❌ Erro nas estatísticas: {e}")
                return jsonify({"status": 0, "message": str(e)}), 500
        
        @self.app.route('/empresas', methods=['GET'])
        def listar_empresas():
            """Lista todas as empresas cadastradas"""
            try:
                empresas = []
                for db_file in Config.DATABASES_DIR.glob("empresa_*.db"):
                    empresa_id = int(db_file.stem.split('_')[1])
                    empresas.append(empresa_id)
                
                return jsonify({
                    "status": 1,
                    "empresas": sorted(empresas)
                })
                
            except Exception as e:
                return jsonify({"status": 0, "message": str(e)}), 500
    
    def generate_processing_id(self, data: Dict) -> str:
        """Gera ID único para processamento"""
        timestamp = datetime.now().isoformat()
        empresa_id = data.get('empresa_info', {}).get('empresa_id', 0)
        hash_input = f"{timestamp}_{empresa_id}_{data.get('name_imagem_original', '')}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def process_florence_background(self, empresa_id: int, processing_id: str, image_path: str):
        """Processa Florence-2 em background"""
        try:
            if not image_path or not Path(image_path).exists():
                self.logger.warning(f"⚠️  Imagem não encontrada para Florence-2: {image_path}")
                return
            
            self.logger.info(f"🔄 Iniciando Florence-2 background: {processing_id}")
            
            # Carregar imagem
            image = self.file_manager.load_image_as_pil(image_path)
            if not image:
                self.logger.error(f"❌ Erro ao carregar imagem PIL: {image_path}")
                return
            
            # Processar com Florence-2
            florence_results = self.florence_processor.process_image(image, processing_id)
            
            # Atualizar banco de dados
            if florence_results and not florence_results.get('error'):
                self.db_manager.update_florence_results(empresa_id, processing_id, florence_results)
                self.logger.info(f"✅ Florence-2 finalizado: {processing_id}")
            else:
                self.logger.error(f"❌ Florence-2 falhou: {processing_id} - {florence_results.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            self.logger.error(f"❌ Erro no Florence-2 background: {e}")
            self.logger.error(traceback.format_exc())
    
    def start_background_tasks(self):
        """Inicia tarefas em background"""
        def cleanup_task():
            """Task de limpeza automática"""
            while True:
                try:
                    time.sleep(3600 * 24)  # 24 horas
                    self.logger.info("🧹 Iniciando limpeza automática...")
                    self.file_manager.cleanup_old_files(Config.AUTO_CLEANUP_DAYS)
                except Exception as e:
                    self.logger.error(f"❌ Erro na limpeza automática: {e}")
        
        # Iniciar thread de limpeza
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        
        self.logger.info("✅ Tarefas em background iniciadas")
    
    def run(self):
        """Executa o servidor"""
        self.logger.info(f"🚀 Iniciando servidor em {Config.HOST}:{Config.PORT}")
        self.logger.info(f"📁 Dados em: {Config.DATA_DIR}")
        self.logger.info(f"🤖 Florence-2: {'✅ Ativo' if self.florence_processor.model_loaded else '❌ Inativo'}")
        
        self.app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True
        )

# =============================================================================
# UTILITÁRIOS AVANÇADOS
# =============================================================================

class AdvancedUtils:
    """Utilitários avançados para o sistema"""
    
    @staticmethod
    def validate_payload(data: Dict) -> Tuple[bool, List[str]]:
        """Valida payload recebido"""
        errors = []
        
        # Campos obrigatórios
        required_fields = [
            'empresa_info',
            'name_imagem_original',
            'timestamp'
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Campo obrigatório ausente: {field}")
        
        # Validar empresa_info
        if 'empresa_info' in data:
            empresa_info = data['empresa_info']
            required_empresa_fields = ['empresa_id', 'empresa_nome']
            
            for field in required_empresa_fields:
                if field not in empresa_info:
                    errors.append(f"Campo empresa_info.{field} obrigatório")
        
        # Validar imagens base64
        if 'imagem_original_base64' in data:
            try:
                base64.b64decode(data['imagem_original_base64'])
            except Exception:
                errors.append("imagem_original_base64 inválida")
        
        if 'imagem_processada_base64' in data:
            try:
                base64.b64decode(data['imagem_processada_base64'])
            except Exception:
                errors.append("imagem_processada_base64 inválida")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Formata tamanho de arquivo"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Retorna estatísticas do sistema"""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "uptime": time.time()
        }

# =============================================================================
# SISTEMA DE MONITORAMENTO
# =============================================================================

class MonitoringSystem:
    """Sistema de monitoramento e alertas"""
    
    def __init__(self):
        self.logger = logger_manager.get_logger('MONITORING')
        self.alerts = defaultdict(list)
        self.metrics = defaultdict(float)
    
    def record_metric(self, name: str, value: float):
        """Registra métrica"""
        self.metrics[name] = value
        
        # Alertas automáticos
        if name == "processing_time_ms" and value > 30000:  # 30 segundos
            self.add_alert(f"Processamento lento detectado: {value}ms")
        
        if name == "error_rate" and value > 0.1:  # 10% de erro
            self.add_alert(f"Taxa de erro alta: {value*100:.1f}%")
    
    def add_alert(self, message: str):
        """Adiciona alerta"""
        timestamp = datetime.now()
        self.alerts[timestamp.date()].append({
            "timestamp": timestamp.isoformat(),
            "message": message
        })
        self.logger.warning(f"🚨 ALERTA: {message}")
    
    def get_alerts(self, days: int = 7) -> List[Dict]:
        """Retorna alertas recentes"""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        recent_alerts = []
        
        for date, alerts in self.alerts.items():
            if date >= cutoff_date:
                recent_alerts.extend(alerts)
        
        return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)

# =============================================================================
# API DE RELATÓRIOS AVANÇADOS
# =============================================================================

class ReportsAPI:
    """API para geração de relatórios avançados"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logger_manager.get_logger('REPORTS')
    
    def generate_daily_report(self, empresa_id: int, date: str) -> Dict:
        """Gera relatório diário"""
        try:
            with self.db_manager.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                # Estatísticas do dia
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_processamentos,
                        SUM(detections_count) as total_deteccoes,
                        AVG(processing_time_ms) as tempo_medio,
                        MIN(timestamp) as primeiro_processamento,
                        MAX(timestamp) as ultimo_processamento,
                        COUNT(CASE WHEN florence_success = 1 THEN 1 END) as florence_sucessos
                    FROM processamentos 
                    WHERE DATE(timestamp) = ?
                """, (date,))
                
                stats = dict(cursor.fetchone() or {})
                
                # Top detecções por coletor
                cursor.execute("""
                    SELECT 
                        coletor_id,
                        coletor_descricao,
                        COUNT(*) as processamentos,
                        SUM(detections_count) as total_deteccoes
                    FROM processamentos 
                    WHERE DATE(timestamp) = ?
                    GROUP BY coletor_id, coletor_descricao
                    ORDER BY total_deteccoes DESC
                    LIMIT 10
                """, (date,))
                
                top_coletores = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "date": date,
                    "statistics": stats,
                    "top_coletores": top_coletores,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Erro no relatório diário: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self, empresa_id: int, days: int = 30) -> Dict:
        """Relatório de performance"""
        try:
            with self.db_manager.get_connection(empresa_id) as conn:
                cursor = conn.cursor()
                
                # Performance por dia
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as data,
                        COUNT(*) as processamentos,
                        AVG(processing_time_ms) as tempo_medio,
                        MIN(processing_time_ms) as tempo_minimo,
                        MAX(processing_time_ms) as tempo_maximo
                    FROM processamentos 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY data DESC
                """.format(days))
                
                performance_data = [dict(row) for row in cursor.fetchall()]
                
                # Estatísticas Florence-2
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_processados,
                        SUM(CASE WHEN florence_success = 1 THEN 1 ELSE 0 END) as sucessos,
                        AVG(CASE WHEN florence_success = 1 THEN 1.0 ELSE 0.0 END) as taxa_sucesso
                    FROM processamentos 
                    WHERE timestamp >= datetime('now', '-{} days')
                """.format(days))
                
                florence_stats = dict(cursor.fetchone() or {})
                
                return {
                    "period_days": days,
                    "performance_data": performance_data,
                    "florence_statistics": florence_stats,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Erro no relatório de performance: {e}")
            return {"error": str(e)}

# =============================================================================
# MAIN - PONTO DE ENTRADA
# =============================================================================

def main():
    """Função principal"""
    try:
        # Banner de inicialização
        print("=" * 80)
        print("🚀 SERVIDOR COMPLETO DE ANÁLISE E GERENCIAMENTO")
        print("   Sistema Robusto para Processamento de Dados com Florence-2")
        print("=" * 80)
        print()
        
        # Verificar dependências
        if not FLORENCE_AVAILABLE:
            print("⚠️  AVISO: Florence-2 não disponível. Funcionalidades limitadas.")
            print("   Para ativar: pip install transformers torch")
            print()
        
        # Criar instância do servidor
        servidor = ServidorCompleto()
        
        # Adicionar rotas de relatórios
        reports_api = ReportsAPI(servidor.db_manager)
        monitoring = MonitoringSystem()
        
        @servidor.app.route('/relatorio/diario/<int:empresa_id>/<date>', methods=['GET'])
        def relatorio_diario(empresa_id, date):
            """Relatório diário da empresa"""
            report = reports_api.generate_daily_report(empresa_id, date)
            return jsonify({"status": 1, "report": report})
        
        @servidor.app.route('/relatorio/performance/<int:empresa_id>', methods=['GET'])
        def relatorio_performance(empresa_id):
            """Relatório de performance"""
            days = int(request.args.get('days', 30))
            report = reports_api.generate_performance_report(empresa_id, days)
            return jsonify({"status": 1, "report": report})
        
        @servidor.app.route('/sistema/status', methods=['GET'])
        def sistema_status():
            """Status completo do sistema"""
            try:
                stats = AdvancedUtils.get_system_stats()
                alerts = monitoring.get_alerts(7)
                
                return jsonify({
                    "status": 1,
                    "system_stats": stats,
                    "recent_alerts": alerts,
                    "florence_status": servidor.florence_processor.model_loaded,
                    "uptime": datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({"status": 0, "error": str(e)})
        
        @servidor.app.route('/sistema/cleanup', methods=['POST'])
        def manual_cleanup():
            """Limpeza manual do sistema"""
            try:
                days = int(request.json.get('days', Config.AUTO_CLEANUP_DAYS))
                servidor.file_manager.cleanup_old_files(days)
                return jsonify({"status": 1, "message": f"Limpeza executada para {days} dias"})
            except Exception as e:
                return jsonify({"status": 0, "error": str(e)})
        
        # Informações de inicialização
        print(f"📍 Host: {Config.HOST}:{Config.PORT}")
        print(f"📁 Diretório de dados: {Config.DATA_DIR}")
        print(f"🎯 Florence-2: {'Ativo' if servidor.florence_processor.model_loaded else 'Inativo'}")
        print(f"🧹 Limpeza automática: {Config.AUTO_CLEANUP_DAYS} dias")
        print()
        print("📋 ENDPOINTS DISPONÍVEIS:")
        print("   POST /process                    - Processar dados principais")
        print("   GET  /consulta/<empresa_id>      - Consultar processamentos")
        print("   GET  /estatisticas/<empresa_id>  - Estatísticas da empresa")
        print("   GET  /empresas                   - Listar empresas")
        print("   GET  /relatorio/diario/<empresa_id>/<date> - Relatório diário")
        print("   GET  /relatorio/performance/<empresa_id>   - Relatório performance")
        print("   GET  /sistema/status             - Status do sistema")
        print("   POST /sistema/cleanup            - Limpeza manual")
        print("   GET  /health                     - Health check")
        print()
        print("🌐 Servidor pronto para receber dados via NGROK!")
        print("=" * 80)
        
        # Executar servidor
        servidor.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        traceback.print_exc()
    finally:
        print("👋 Servidor finalizado")

if __name__ == '__main__':
    main()