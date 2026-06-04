import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig
from langchain_community.document_loaders import PyMuPDFLoader

class PdfIngestor:
    def __init__(self,config:QaConfig)-> None:
        self.__config= config

    def ingest(self,pdf_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp:
            temp.write(pdf_bytes)
            temp.flush()
            temp_path=temp.name
        try:
            #1.加载
            documents=self.load(temp_path)
            #2.切分
            return self.split(documents)
        finally:
            os.unlink(temp_path)

    def load(self,pdf_path: str) -> List[Document]:
        return PyMuPDFLoader(pdf_path).load()

    def split(self,doce) :
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.__config.chunk_size,
            chunk_overlap=self.__config.chunk_overlap,
            is_separator_regex=True,
            separators=["(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "(?<=，)", " ", "\n"],
        )
        return splitter.split_documents(doce)
