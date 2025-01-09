from typing import Dict
import uuid

def authorize_function(
    id_param: str, 
    url_param: str, 
    db, 
    chat_history_manager: Dict[str, Dict[str, list]]
):
    try:
        print({"id_param": id_param})
        print({"url_param": url_param})

        userId = str(uuid.uuid4())
        chat_history_manager[userId] = {
            "chat_history": [],
            "user_prompt_history": [],
            "chat_answers_history": []
        }

        docs = db.collection("user_data").where("id", "==", id_param).where("url", "==", url_param).stream()
        for doc in docs:
            data = doc.to_dict()
            return {
                "authorized": True,
                "id": data.get("id"),
                "user_id": userId
            }

        return {"authorized": False, "status_code": 401}

    except Exception as e:
        print(f"Error during authorization: {str(e)}")
        return {"error": str(e), "status_code": 500}
