import os
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from typing import Dict

from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

# Initialize GROQ LLM
groq_api_key = os.getenv("GROQ_API")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

async def customize_conversation( chat_history_manager: Dict[str, Dict[str, list]] , unique_id: str, user_id: str , text: str ,assistant_name: str , company_name: str):
    try:
        print(unique_id)
        print({user_id})
        folder_path = f"./store/{unique_id}"

        if os.path.exists(folder_path):
            vectors = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
        else:
            return {"status": "failed", "answer": "Please provide your Id"}
        
        general_system_template = f"""
        You are {assistant_name} AI assistant of {company_name} Company and the context shared is the information about your company. 
        Your role is to have a professional conversation with users about the context only. 
        Say 'I don't have information' if something is asked out of the context. Provide responses in 10-15 words only. 
        Greet well, behave professionally, and think like a human.
        ----
        {{context}}
        ----
        """
        
        general_user_template = "Question:{question}"
        messages = [
             SystemMessagePromptTemplate.from_template(general_system_template),
             HumanMessagePromptTemplate.from_template(general_user_template),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        chat_history_user = chat_history_manager.get(user_id)
        
        qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever= vectors.as_retriever(),
                combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        response = qa({"question": text, "chat_history": chat_history_user["chat_history"]})
        print({response['answer']})

        chat_history_user["chat_answers_history"].append(response['answer'])
        chat_history_user["user_prompt_history"].append(text)
        chat_history_user["chat_history"].append((text, response['answer']))

        return {"status": "success", "message": response["answer"]}

    except Exception as e:
        print(f"Error in conversation customization: {str(e)}")
        return {"status": "error", "message": str(e)}
    