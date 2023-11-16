from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings import GPT4AllEmbeddings  # , OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


def load_file(path: str) -> list[Document]:
    loader = TextLoader(path)
    data: list[Document] = loader.load()
    return data


def split(data: list[Document]) -> list[Document]:
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    splits: list[Document] = text_spliter.split_documents(data)
    return splits


def embed(data: list[Document]):
    vectorstore: Chroma = Chroma.from_documents(
        documents=data,
        embedding=GPT4AllEmbeddings(),
    )
    return vectorstore


def retrieve(question: str, vectorstore: Chroma) -> str:
    answer: str = vectorstore.similarity_search(question)
    return answer


def rag(question: str, vectorstore: Chroma):
    llm = Ollama(
        # base_url="http://localhost:11434",
        model="llama2:13b",
        verbose=False,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    qa_chain: BaseRetrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    answer = qa_chain({"query": question})
    return answer


def main() -> None:
    path: str = "./cuento.txt"

    data: list[Document] = load_file(path)
    data = split(data)

    embeddings = embed(data)

    question: str = "¿Qué deseaba hacer Salto que lo hacía diferente de los otros sapos?. Dame la respuesta en Español y de forma corta."

    answer: str = retrieve(question, embeddings)

    answer = rag(question, embeddings)["query"]
    print(answer)


if __name__ == "__main__":
    main()
