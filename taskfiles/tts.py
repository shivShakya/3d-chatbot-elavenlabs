import os
from elevenlabs.client import ElevenLabs

elaven_labs_key = os.getenv("Eleven_labs_Key")
elaven_labs_voice_id = os.getenv("Elaven_labs_voice_id")
client = ElevenLabs( api_key= elaven_labs_key )

def useElavenlabsVoice(request):
    try:
        response = client.text_to_speech.convert_with_timestamps(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            output_format="mp3_44100_128",
            text=request['message'],
            model_id="eleven_multilingual_v2"
        )
        return response
    except KeyError as e:
        print(f"Error: Missing expected key {e} in the request")
    except Exception as e:
        print(f"An error occurred: {e}")
