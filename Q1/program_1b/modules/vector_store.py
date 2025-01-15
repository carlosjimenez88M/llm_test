#=====================#
# ---- libraries ---- #
#=====================#
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
import os

#=========================#
# ---- main function ---- #
#=========================#
def create_vector_store(data_path: str, openai_api_key: str):
    """
    Create a FAISS vector store from CSV data.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing data for the vector store.
    openai_api_key : str
        OpenAI API key for generating embeddings.

    Returns
    -------
    FAISS
        The FAISS vector store instance.
    """
    # Verificar que el archivo existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found at {data_path}")

    try:
        # Load data from CSV
        loader = CSVLoader(
            file_path=data_path,
            source_column="query",  # Usamos la columna 'query' como fuente
            csv_args={
                'delimiter': ',',
                'quotechar': '"'
            }
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} documents from CSV")

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create and return vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store

    except Exception as e:
        print(f"Error details: {str(e)}")
        raise RuntimeError(f"Error processing {data_path}: {str(e)}")