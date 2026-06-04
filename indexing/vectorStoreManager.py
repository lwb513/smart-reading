import os.path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.setting import QaConfig
from langchain_openai.embeddings import OpenAIEmbeddings


class VectorStoreManager:
    def __init__(self,config:QaConfig,embeddings:OpenAIEmbeddings):
        self.__config= config
        self.__embeddings= embeddings

    def load_or_build(self,file_hash:str,chunks:List[Document]= None)-> Chroma:
        persist_dir=os.path.join(self.__config.chroma_root_dir,file_hash)
        collection_name=f"pdf_qa_{file_hash}"
        store = Chroma(
            collection_name=collection_name,
            embedding_function=self.__embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        if store._collection.count() > 0:
            return store
        # 无数据且未提供 chunks，则报错
        if not chunks:
            raise ValueError("向量库不存在且未提供 chunks，无法建库")
        store.add_documents(
            documents=chunks,
            ids=[f"id-{idx}" for idx in range(1, len(chunks) + 1)],
        )
        return  store