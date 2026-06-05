from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from querying.query_prompt import READING_HELPER_PROMPT


class AnswerGenerator:
    """答案生成器：根据问题、对话历史、检索上下文生成答案"""

    def __init__(self, llm: ChatOpenAI, ) -> None:
        """
                初始化生成器，预编译 prompt 链
                :param llm: 已配置好通义千问 LLM 实例
                """
        self.__llm = llm
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", READING_HELPER_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}\n\n{context}"),
            ]
        )
        self.__chain=prompt | self.__llm | StrOutputParser()

    def generate(self,question:str,context:str,chat_history:list[BaseMessage])->str:
        """
                生成答案
                :param question: 用户原始问题（不是改写后的查询）
                :param chat_history: 对话历史对象（需有 .messages 属性，如 ChatMessageHistory）
                :param context: 格式化后的上下文字符串（包含引用编号和片段内容）
                :return: 答案字符串
                """
        return self.__chain.invoke({"question":question,"chat_history":chat_history,"context":context})