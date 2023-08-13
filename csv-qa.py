from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_KEY"] = ""

def generate_vector():
    loader = DirectoryLoader('./data/', glob='./*.csv', loader_cls=CSVLoader)
    data = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    db = FAISS.from_documents(data, embeddings)

    return db

def load_model():
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    return model
 
def main(query):
    db = generate_vector()
    model = load_model()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=db.as_retriever(), memory=memory)
    result = chain({"question": query})

    return result["answer"]
    

if __name__ == '__main__':  
    answer = main('What is the department of Rachel Booker?')
    print("answer: ", answer)