from langserve import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser

llm = RemoteRunnable("http://0.0.0.0:9012/basic_chat/") | StrOutputParser()
for token in llm.stream("Hello World! How is it going?"):
    print(token, end='')