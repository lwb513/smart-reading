import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from config.setting import QaConfig
from querying.query_prompt import RERANK_PROMPT
@dataclass
class RetrievedEvidence:
    """检索到的证据片段（经过重排序后输出的结构化信息）"""
    content: str  # 片段文本内容
    source: str  # 来源（如 PDF 文件名）
    page: Optional[int]  # 页码（如果有）
    score: float  # 最终排序分数（LLM 分数或混合分数）
    vector_score: float  # 原始向量相似度分数（0~1）
    hybrid_score: float  # 混合重排序分数（向量 + 文本重叠）
    llm_score: Optional[float] = None  # LLM 精排给出的分数（0~10）

class RerankPipeline:
    def __init__(self, config: QaConfig, llm: ChatOpenAI) -> None:
        self.__config = config
        prompt = ChatPromptTemplate.from_messages([
            ("system", RERANK_PROMPT),
            ("human", "问题:{question}\n\n候选片段:\n{chunk}\n\n分数:")
        ])
        self.__llm = llm
        self.__chain = prompt | llm | StrOutputParser()

    def rerank(self,
               query: str,
               recalled_docs: List[Document],
               vec_scores: List[float],
               hybrid_top_m: int = 12,
               final_top_n: int = 4
               )-> Tuple[List[RetrievedEvidence], str] :
        #预过滤
        pre_docs=self.__filter(recalled_docs)
        # {文档1:分数1, 文档2:分数2.。。文档30:分数30.} score_map
        score_map = {doc.page_content: s for doc, s in zip(recalled_docs, vec_scores)}
        aligned_vec = [score_map.get(doc.page_content, 0.5) for doc in pre_docs]
        #混合重排
        hybrid_docs, hybrid_scores = self.__hybrid_rerank(query, pre_docs, aligned_vec, top_k=hybrid_top_m)
        final_docs, llm_scores =self.__llm_rerank(query, hybrid_docs, final_top_n)
        evidence_list=[]
        for i,doc in enumerate(final_docs):
            content=doc.page_content
            meta=doc.metadata or {}
            hybrid_score=0.0,
            for hd,hs in zip(hybrid_docs, hybrid_scores):
                if content == hd.page_content.strip():
                    hybrid_score=hs
                    break
            ev=RetrievedEvidence(
                content=content,
                source=meta.get("source"),
                page=meta.get("page"),
                score=llm_scores[i],
                vector_score=score_map.get(content, 0.0),
                hybrid_score=hybrid_score,
                llm_score=llm_scores[i]
            )
            evidence_list.append(ev)
            # 格式化上下文
        context_lines = []
        for idx, ev in enumerate(evidence_list, 1):
                # page_str = f"第{ev.page + 1}页" if ev.page else "未知页"
            page_str = f"第{ev.page + 1}页" if ev.page is not None else "未知页"
            context_lines.append(f"[证据{idx} | 来源：{ev.source} | {page_str}]\n{ev.content}")
        context = "\n\n".join(context_lines)

        return evidence_list, context




    def __filter(self,docs:List[Document],min_len:int=40,max_len:int=2000) ->List[Document]:
        """
                过滤掉过短（<min_len）或过长（>max_len）的片段，并去重
                :param docs: 原始文档列表
                :param min_len: 最小字符数
                :param max_len: 最大字符数
                :return: 过滤后的文档列表
        """
        filtered=[]
        for doc in docs:
            text=doc.page_content
            if min_len<=len(text)<=max_len:
                filtered.append(doc)

        is_contains=set()
        kept=[]
        for doc in filtered:
            content=doc.page_content
            if content not in is_contains:
                kept.append(doc)
                is_contains.add(content)
        return  kept

    def __hybrid_rerank(self, question: str, documents: List[Document],
                        vector_scores: List[float], top_k: int = 12) -> Tuple[List[Document], List[float]]:
        """
        混合重排序：结合向量分数和文本重叠分数，并加入长度/关键词规则
        :param question: 用户问题（用于计算重叠和关键词）
        :param documents: 待重排的文档列表
        :param vector_scores: 对应的向量分数列表
        :param top_k: 保留前 top_k 个
        :return: (排序后的文档列表, 对应的混合分数列表)
        """
        if not documents:
            return [], []
        # 计算每个文档的文本重叠分数
        overlap_scores = [self.__simple_overlap_score(question, doc.page_content) for doc in documents]

        final = []  # 存储 (原始索引, 混合分数)
        for i, (vector_score, overlap_score) in enumerate(zip(vector_scores, overlap_scores)):
            # 混合分数 = 0.3 * 向量分 + 0.7 * 文本重叠分
            hybrid = 0.3 * vector_score + 0.7 * overlap_score
            text = documents[i].page_content
            # 长度规则：果断或过长降权
            if len(text) < 80:
                hybrid *= 0.8
            elif len(text) > 1600:
                hybrid *= 0.9
            # 关键词规则：如果问题和文本有共同单词，适当提权
            if set(question.lower().split()) & set(text.lower().split()):
                hybrid *= 1.1
            final.append((i, hybrid))
        # 按照混合分数降序排序
        final.sort(key=lambda t: t[1], reverse=True)
        idx_sorted = [i for i, _ in final[:top_k]]
        docs_sorted = [documents[i] for i in idx_sorted]
        scores_sorted = [final[i][1] for i in range(top_k) if i < len(final)]
        return docs_sorted, scores_sorted


    def __simple_overlap_score(self, query: str, text: str) -> float:
        """
        计算查询与文本的 Jaccard 重叠度（支持中文双字词 + 英文单词）
        :param query: 查询字符串
        :param text: 文本片段
        :return: 重叠度，范围 [0,1]
        """

        def tokens(s: str):
            s = s.lower()
            # 提取连续的中文字符（至少两个）
            chinese = re.findall(r'[\u4e00-\u9fff]{2,}', s)
            # 提取英文单词
            english = re.findall(r'[a-z]+', s)
            # 对中文生成双字 bigram
            bigrams = [ch[i:i + 2] for ch in chinese for i in range(len(ch) - 1)]
            return set(bigrams + english)

        q_tokens = tokens(query)
        t_tokens = tokens(text)
        if not q_tokens:
            return 0.0
        inter = q_tokens & t_tokens
        union = q_tokens | t_tokens
        return len(inter) / len(union) if union else 0.0

    def __llm_rerank(self,question:str,documents:List[Document],top_k:int=4)->Tuple[List[Document],List[float]]:
        if not documents:
            return [],[]
        scored=[]
        for doc in documents:
            score_text=self.__chain.invoke({"question":question,"chunk":doc.page_content})
            try:
                match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                score = float(match.group(1)) if match else 0.0
            except Exception:
                score = 0.0
            scored.append((score,doc))
        scored.sort(key=lambda t:t[0],reverse=True)
        docs=[doc for score,doc in scored[:top_k]]
        scores=[score for score,doc in scored[:top_k]]
        return docs,scores

