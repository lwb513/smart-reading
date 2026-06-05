from typing import Tuple, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def recall_with_scores(
        query:str,
        vector_store: Chroma,
        k: int = 4,

)-> Tuple[List[ Document],List[float]]:
    result= vector_store.similarity_search_with_relevance_scores(query,k)
    docs=[]
    scores=[]
    for doc,score in result:
        doc.page_content=doc.page_content.strip()
        docs.append(doc)
        scores.append(score)
    return docs,scores