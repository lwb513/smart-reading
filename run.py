
from typing import  Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline
from querying.rag_pipeline import RagPipeline

class PDFQA:
    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.config = QaConfig()
        self.file_hash: Optional[str] = None
        self.chat_history: Optional[InMemoryChatMessageHistory] = None
        self.rag: Optional[RagPipeline] = None

    def ask(self,question:str):
        if self.file_hash is None:
            self._build_index()
        result=self.rag.query(self.file_hash,question,
                       self.chat_history.messages,
                       )
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(result["answer"])
        return result


    def _build_index(self)->None:
        if self.file_hash is not None:
            return
        index=IndexingPipeline(self.config)
        self.file_hash=index.build_from_bytes(self.pdf_bytes)
        self.chat_history=InMemoryChatMessageHistory()
        self.rag=RagPipeline(self.config)
