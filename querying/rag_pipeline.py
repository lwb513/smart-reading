from typing import Dict, Any

from langchain_chroma import Chroma

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from config.setting import QaConfig
from indexing.vectorStoreManager import VectorStoreManager
from querying.answer import AnswerGenerator
from querying.rerank import RerankPipeline
from querying.rewriter import QueryRewriter
from querying.vector_retriever import recall_with_scores


class RagPipeline:
    def __init__(self, config: QaConfig) -> None:
        self.__config = config
        self.__llm= None
        self._embeddings=None

    def _init_llm(self)->ChatOpenAI:
        if self.__llm is None:
            self.__llm = ChatOpenAI(
                model="qwen3.7-max",
                api_key="",
                base_url="",
            )
            return self.__llm
        return self.__llm

    def _init_embeddings(self)->OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key="",
            base_url="",
        )
        return self._embeddings

    def _load_vector_store(self,file_hash:str)->Chroma:
        embedding=self._init_embeddings()
        manager=VectorStoreManager(self.__config,embedding)
        return manager.load_or_build(file_hash)

    def query(self,
              file_hash:str,
              question:str,
              chat_history:list[BaseMessage],
              *,
              enable_rerank:bool=True,
              enable_rewrite:bool=True,
              recall_k:int=30,
              hybrid_top_m:int=12,
              top_k:int=4)->Dict[str, Any]:
        llm=self._init_llm()
        vector_store=self._load_vector_store(file_hash)
        # 1. 查询改写
        search_query=question
        if enable_rewrite:
            rewriter=QueryRewriter(llm)
            message=chat_history
            search_query=rewriter.rewrite(question,message)

        # 2. 向量召回
        recalled_docs,vec_store= recall_with_scores(search_query,vector_store,k=recall_k)
        #3.重排序
        evidence_list = []
        if enable_rerank:
            rerank_pipeline=RerankPipeline(self.__config,llm)
            evidence_list,context=rerank_pipeline.rerank(search_query,recalled_docs,vec_store,hybrid_top_m,top_k)
        else:
            # 简单检索
            docs = vector_store.similarity_search(search_query, k=top_k)
            context = "\n\n".join([doc.page_content for doc in docs])
        answer_generator=AnswerGenerator(llm)
        answer=answer_generator.generate(question,context,chat_history)

        return {
            "answer": answer,
            "context": context,
            "evidence": evidence_list,
            "search_query": search_query,
            "rewritten": enable_rewrite and (search_query != question),
        }


