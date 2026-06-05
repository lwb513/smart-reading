from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from querying.query_prompt import REWRITE_PROMPT


class QueryRewriter:
    def __init__(self,llm: ChatOpenAI) -> None:
        self.__prompt=ChatPromptTemplate.from_messages(
            [
                ("system",REWRITE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human","{question}"),
            ]
        )
        self.__llm=llm


    def rewrite(self,question:str ,chat_history:list[BaseMessage])-> str:
        q=(question or "").strip() # strip删除空格制表符等等
        if not q:
            return q
        try:
            chain=self.__prompt | self.__llm |StrOutputParser()
            rewrite=chain.invoke({"question":q,"chat_history":chat_history})
            return rewrite
        except Exception as e:
            return q