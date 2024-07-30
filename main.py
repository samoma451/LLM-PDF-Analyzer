import os

# Import necessary modules from langchain
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Initialize global variables
conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None
# Function to initialize the language model and its embeddings
def init_llm():
    global llm, llm_embeddings
    # Initialize the language model with the OpenAI API key
    openai_api_key = os.environ['OPENAI_API_KEY']
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = OpenAI(model_name="text-davinci-003")
    # Initialize the embeddings for the language model
    llm_embeddings = OpenAIEmbeddings()