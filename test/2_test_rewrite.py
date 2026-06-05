from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from querying.rewriter import QueryRewriter
from dotenv import load_dotenv
import os
load_dotenv()
llm=ChatOpenAI(
    model="qwen3.7-max",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    )
query_rewriter=QueryRewriter(llm)
history=InMemoryChatMessageHistory()
history.add_user_message("我之前问过那篇关于注意力机制的论文")
history.add_ai_message("是的，那篇论文是《Attention Is All You Need》")

question = "那篇论文的核心贡献是什么？"
result=query_rewriter.rewrite(question,history.messages)
print(question)
print(result)
