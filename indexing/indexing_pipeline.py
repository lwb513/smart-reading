
from langchain_community.embeddings import OpenAIEmbeddings

from config.setting import QaConfig
from indexing.ingestor import PdfIngestor
from indexing.storage import PdfBytesStore
from indexing.vectorStoreManager import VectorStoreManager


class IndexingPipeline:
    def __init__(self,config:QaConfig)->  None:
        self.config= config
        self.store=PdfBytesStore()
        self.ingestor=PdfIngestor(config)


    def build_from_bytes(self,pdf_bytes: bytes) -> str:
        # 1. 计算 hash，并且存储临时文件（将hash值作为临时文件的名字）
        file_hash=self.store.compute_hash(pdf_bytes)
        self.store.store(pdf_bytes,file_hash)
        # 2. 加载 & 切分文档
        chunks=self.ingestor.ingest(pdf_bytes)
        # 3.构建向量数据库
        embeddings=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key="",
            base_url=""
        )
        vector_store_manager=VectorStoreManager(self.config,embeddings)
        vector_store_manager.load_or_build(file_hash,chunks)
        return file_hash

    def build_from_file(self, pdf_path: str) -> str:
        """从文件路径构建索引"""
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return self.build_from_bytes(pdf_bytes)