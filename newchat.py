import os
import time
import uuid
import json
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, File, UploadFile , Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pyttsx3

from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.schema import Document
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment

from elevenlabs import Voice, VoiceSettings, play , save
from elevenlabs.client import ElevenLabs

from firebase_admin import credentials, initialize_app, firestore, get_app
from urllib.parse import parse_qs, urlparse

from vosk import Model, KaldiRecognizer, SetLogLevel

import sys
import wave
import json
import time

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SetLogLevel(0)

chat_history_manager = {}

model_path = "./vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    print(f"Please download the model from https://alphacephei.com/vosk/models and unpack as {model_path}")
    sys.exit()
    
print(f"Reading your vosk model '{model_path}'...")
modelVocs = Model(model_path)
print(f"'{model_path}' model was successfully read")



def initialize_firebase():
    try:
         get_app()
    except ValueError:
        cred = credentials.Certificate("keys.json")
        initialize_app(cred)

initialize_firebase()
db = firestore.client()


load_dotenv()

app = FastAPI()
engine = pyttsx3.init()


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionState:
    vectors = None
    txt_data = None
    final_documents = None
    embeddings = None
    qa = None
    chat_history = []
    user_prompt_history = []
    chat_answers_history = []

session_state = SessionState()


# Initialize GROQ LLM
groq_api_key = os.getenv("GROQ_API")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


class AuthorizeRequest(BaseModel):
    id: str  
    url: str

@app.post("/authorize")
async def authorize(request: Request, body: AuthorizeRequest):
    try:
        id_param = body.id
        url_param = body.url
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
            return JSONResponse(content={"authorized": True, "id": data.get("id") , "user_id": userId})

        # If authorization fails
        return JSONResponse(content={"authorized": False}, status_code=401)

    except Exception as e:
        print(f"Error during authorization: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.post("/initialize-vectors")
async def initialize_vectors(file: UploadFile = File(...)):
    try:
        print("Initializing vector store DB...")
        st = time.time()
        unique_id = str(uuid.uuid4())
        folder_path = f"./faiss_db/{unique_id}"
        os.makedirs(folder_path, exist_ok=True)

        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["txt", "pdf"]:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only .txt and .pdf are supported.")

        file_path = f"{folder_path}/uploaded_file.{file_extension}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if file_extension == "txt":
            with open(file_path, "r") as f:
                session_state.txt_data = f.read()

        elif file_extension == "pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            session_state.txt_data = "\n\n".join(page.extract_text() for page in reader.pages)

        else:
            raise HTTPException(status_code=400, detail="File processing not implemented for this format.")

        chunks = session_state.txt_data.split("\n\n")
        session_state.final_documents = [Document(page_content=chunk) for chunk in chunks]

        session_state.vectors = FAISS.from_documents(session_state.final_documents, session_state.embeddings)
        session_state.vectors.save_local(folder_path=folder_path, index_name="index")

        end = time.time() - st
        print({'time': end})
        print("Vector store DB is set up and ready!")
        return JSONResponse(content={"status": "success", "message": "Vector store DB initialized.", "time_taken": end, "id": unique_id})

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



@app.get("/intialVoice")
async def getIntialVoice(assistant_name: str, company_name: str):
    try:
        Intial_info = {"message": f"Hello! I am {assistant_name} from {company_name} Company. How may I help you today?"}

        output_mp3_path = f"{uuid.uuid4()}.mp3"
        output_wav_path = f"{uuid.uuid4()}.wav"

        try:
            useElavenlabsVoice(Intial_info, output_mp3_path, output_wav_path)
            print("Generated audio using Eleven Labs successfully.")
        except Exception as e_labs:
            print(f"Eleven Labs audio generation failed: {e_labs}")
            print("Falling back to pyttsx3.")
            await usePyttsx3Voice(Intial_info, output_wav_path)

        file_id = os.path.splitext(output_wav_path)[0] 
        # Processing the generated audio file
        result = await vockWords(output_wav_path, modelVocs)
        print(result)
        
        return {
            "message": "Audio processed successfully",
            "Id": f"{file_id}",
            "word_timing": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/getVoice")
async def get_voice(request: Request, file: UploadFile = File(...) , vector_id: str = Form(...) , user_id: str = Form(...), assistant_name : str = Form(...) , company_name : str = Form(...)):
    try:
        start_time_i = time.time()  

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        if not vector_id:
            return {"status": "failed", "message": "Vector ID is required"}
            

        print({'vector' :  vector_id})
        text = await generateTextWithAudio(file)
        print({"text": text})
        
        response = await customize_conversation(vector_id, user_id , text , assistant_name , company_name)
        print(response)
        
        if response["status"] != "success":
            return {"status": response["status"], "message": response["message"]}

        
        output_mp3_path = f"{uuid.uuid4()}.mp3"
        output_wav_path = f"{uuid.uuid4()}.wav"
        print('hii')
        start_time = time.time()  
        try:
            useElavenlabsVoice(response, output_mp3_path ,output_wav_path)
            print("Generated audio using eleven successfully.")
        except Exception as e_labs:
            print(f"eleven lab  audio generation failed: {e_labs}")
            print("Falling back to pyttsx3.")
            await usePyttsx3Voice(response, output_wav_path)
        
        print(f"Execution time text to speech: {time.time() - start_time} seconds")  

        #output_file_path = f"{uuid.uuid4()}.wav"
        #convert_to_mono(file_path, output_file_path)
        #os.remove(file_path)

        start_time = time.time()  
        
        file_id = os.path.splitext(output_wav_path)[0]
        #json_file_path = file_id + ".json"
        
        #result = rhubarbProcess(output_wav_path , output_mp3_path, json_file_path)
        result  = await vockWords(output_wav_path , modelVocs)
        print(result)

        print(f"Execution time rhubarb: {time.time() - start_time_i} seconds")  

        start_time = time.time()


        #with open(json_file_path, "r") as json_file:
            #json_output = json_file.read()
        #json_final_output =  convert_to_json(json_output , "" , 0)

        print(f"Execution time convert to json: {time.time() - start_time} seconds")  
        return {
            "message": "Audio processed successfully",
            "Id": f"{file_id}", 
            "word_timing": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    
@app.get("/audio/{filename}")
async def stream_audio(filename: str, background_tasks: BackgroundTasks):  
    response = StreamingResponse(iterfile(filename), media_type="audio/mp3")
    background_tasks.add_task(remove_files, filename) 
    return response

def iterfile(filename):
    try:
        file_path = f"{filename}.wav"
        with open(file_path, mode="rb") as audio_file:
            while chunk := audio_file.read(1024 * 1024): 
                yield chunk
        os.remove(f"{filename}.wav")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

def remove_files(filename):
    try:
        os.remove(f"{filename}.json")
        #os.remove(f"{filename}.wav")
    except FileNotFoundError:
        print(f"File not found for {filename} cleanup.")
    except PermissionError as e:
        print(f"Permission error while trying to delete files: {e}")
    except Exception as e:
        print(f"Error while deleting files: {e}")


def transcript_to_phenomes(sentance, phenom_to_morph):
     phenome_sequence = []
     words = sentance.split("")

     for word in words:
         phenomes = []

         for char in word:
              if char in phenom_to_morph:
                   phenomes.append(char)
        
         if phenomes:
             phenome_sequence.append("".join(phenomes))
    
     return phenome_sequence
                   


def useElavenlabsVoice(request ,output_mp3_path, output_wav_path):
    elaven_labs_key = os.getenv("Eleven_labs_Key")
    elaven_labs_voice_id = os.getenv("Elaven_labs_voice_id")
    client = ElevenLabs( api_key= elaven_labs_key )
    audio = client.generate(
                 text=request['message'],
                 voice= elaven_labs_voice_id,
                 model="eleven_multilingual_v2"
        )
    save(audio , output_mp3_path)
    convert_mp3_to_mono_wav(output_mp3_path, output_wav_path)


def convert_mp3_to_mono_wav(input_mp3_path, output_wav_path):
    command = [
        "ffmpeg", "-y",
        "-i", input_mp3_path,  # Input MP3 file
        "-ac", "1",  # Convert to mono (1 channel)
        "-ar", "22050",  # Sample rate 22,050 Hz (adjust if needed)
        "-acodec", "pcm_s16le",  # 16-bit PCM codec for WAV
        output_wav_path  # Output WAV file path
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Successfully converted {input_mp3_path} to {output_wav_path}")

        os.remove(input_mp3_path)
        print(f"Deleted the MP3 file: {input_mp3_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode('utf-8')}")


async def usePyttsx3Voice(request , output_file_path):
        message = request.get('message', '')
        voices = engine.getProperty('voices')
       
        engine.setProperty('voice' , voices[0].id)
        newVoiceRate = 145
        engine.setProperty('rate',newVoiceRate)
        print({'voices' : voices})
        print({'message' : message})
        engine.save_to_file(message, output_file_path)
        engine.runAndWait()



def convert_to_json(input_data, sound_file, duration):
    lines = input_data.strip().split("\n")

    events = []
    for i in range(len(lines) - 1):
        time, value = lines[i].split("\t")
        next_time, _ = lines[i + 1].split("\t")
        start_time = float(time)
        end_time = float(next_time)
        
        events.append({
            "start": start_time,
            "end": end_time,
            "value": value
        })

    last_time, last_value = lines[-1].split("\t")
    events.append({
        "start": float(last_time),
        "end": duration,
        "value": last_value
    })
    
    metadata = {
        "metadata": {
            "soundFile": sound_file,
            "duration": duration
        },
        "mouthCues": events
    }
    
    return json.dumps(metadata, indent=2)

async def generateTextWithAudio(file):
        audio_data = BytesIO(await file.read())
        try:
           audio_segment = AudioSegment.from_file(audio_data, format="wav")
        except Exception as e:
           raise HTTPException(status_code=400, detail=f"Invalid audio file format: {str(e)}")

        wav_audio_data = BytesIO()
        audio_segment.export(wav_audio_data, format="wav")
        wav_audio_data.seek(0) 
        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_audio_data) as source:
             audio_content = recognizer.record(source)

        try:
           text = recognizer.recognize_google(audio_content)
           return text
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech could not be understood.")
        except sr.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Recognition service error: {str(e)}")


def rhubarbProcess(output_wav_path, output_mp3_path, json_file_path):
    rhubarb_path = r"rhubarb.exe"
    if os.path.exists(output_mp3_path):
        os.remove(output_mp3_path)
    command = [
        rhubarb_path,
        output_wav_path,        
        "-o", json_file_path,  
        "-r", "phonetic"       
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result  
    except subprocess.CalledProcessError as e:
        print(f"Rhubarb process failed: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Rhubarb executable not found at {rhubarb_path}. Check the path.")
        return None

async def vockWords(output_wav_path , model):
    wf = wave.open(output_wav_path, "rb")
    print(f"'{output_wav_path}' file was successfully read")

    SetLogLevel(0)
    wf = wave.open(output_wav_path, "rb")
    print(f"'{output_wav_path}' file was successfully read")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
       print("Audio file must be WAV format mono PCM.")
       sys.exit()

    start_time = time.time()
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []

    while True:
       data = wf.readframes(4000)
       if len(data) == 0:
            break
       if rec.AcceptWaveform(data):
           part_result = json.loads(rec.Result())
           results.append(part_result)

  
    part_result = json.loads(rec.FinalResult())
    results.append(part_result)
   
    end_time = time.time()
    return results


async def customize_conversation(unique_id: str, user_id: str , text: str ,assistant_name: str , company_name: str):
    try:
        print(unique_id)
        print({user_id})
        folder_path = f"./faiss_db/{unique_id}"

        if os.path.exists(folder_path):
            vectors = FAISS.load_local(folder_path, session_state.embeddings, allow_dangerous_deserialization=True)
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
    

@app.get("/removeUserId")
async def remove(user_id: str):
    try:
        if user_id in chat_history_manager:
            del chat_history_manager[user_id]
            return JSONResponse(content={"message": f"User {user_id} removed successfully."})
        else:
            return JSONResponse(content={"error": f"User {user_id} not found."}, status_code=404)
    except Exception as e:
        print(f"Error while removing user_id {user_id}: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("newchat:app", host="0.0.0.0", port=8000, reload=True)