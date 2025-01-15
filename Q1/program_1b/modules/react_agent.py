#=====================#
# ---- libraries ---- #
#=====================#
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA

#=========================#
# ---- main function ---- #
#=========================#
def create_react_agent(vector_store: FAISS, openai_api_key: str):
    """
    Create a React agent for answering insurance-related questions.

    Parameters
    ----------
    vector_store : FAISS
        The FAISS vector store instance containing insurance information.
    openai_api_key : str
        OpenAI API key for the language model.

    Returns
    -------
    Agent
        The initialized React agent.
    """
    try:
        # Initialize LLM
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        # Define tools for the agent
        tools = [
            Tool(
                name="Insurance QA System",
                func=qa_chain.run,
                description="Use this tool to answer questions about insurance coverage, claims, and policies."
            )
        ]
        
        # Initialize the agent with the correct type
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Cambiado a un tipo válido
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent
        
    except Exception as e:
        print(f"Error creating React agent: {str(e)}")
        raise