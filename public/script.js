let nodes = {};
let lipsync;
let audio = null;
let mixer ;

let controller;
let signal;
let scene;

let model;  
let morphTargetNames = []
let closedBtnclick = false;
let intialAudioPlaying = false;
let link = ""
let processingRequest = false;

let userId;
let finalTranscript = "";

const states = document.getElementById('states');
const waveform = document.getElementById('mic-overlay1');


let currentIndex = 0;

function getQueryParams() {
    const urlParams = new URLSearchParams(window.location.search);
    return {
        id: urlParams.get('id'),
        assistant_name: urlParams.get("assistant_name"),
        company_name: urlParams.get("company_name")
    };
}

async function callAuthorizeAPI() {
    const parentOrigin = document.referrer || window.top.location.origin; 
    console.log({parentOrigin})
    let params = getQueryParams();

    if (!params.id || !parentOrigin) {
        console.error('Missing required parameters: id or url');
        return;
    }

    try {
        const response = await fetch(`${link}/authorize`, {
            method: 'POST', 
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                id: params.id,
                url: parentOrigin
            }), 
        });

        if (response.ok) {
            const data = await response.json();
            console.log('API Response:', data);
            return data;
        } else {
            console.error('Error in API request:', response.status);
        }
    } catch (error) {
        console.error('Error in fetch:', error);
    }
}

window.onload = function () {
    display3DModel(nodes); 
};



function display3DModel(nodes) {
    let overlay = document.getElementById('canvas-container');

    let width = window.innerWidth;
    let height = window.innerHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width , height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0xff00ff, 0); 
    overlay.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60,width/height, 0.01, 1000);
    camera.updateProjectionMatrix();

    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 5); 
    directionalLight2.position.set(0,1,1000);
    scene.add(directionalLight2);


    const ambientLight = new THREE.AmbientLight(0xffffff, 2); 
    scene.add(ambientLight);



   const loader = new THREE.GLTFLoader();
   const loaderFbx = new THREE.FBXLoader();

   const ktx2Loader = new THREE.KTX2Loader()
   .setTranscoderPath('./assets/basis/')
   .detectSupport(renderer);
   
   loader.setKTX2Loader(ktx2Loader);
   loader.setMeshoptDecoder(MeshoptDecoder);
    loader.load('assets/model/dave-v12.glb', function(gltf) {
        model = gltf.scene;
        model.scale.set(0.8,0.8,0.8); 
        scene.add(model);


        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());

        model.position.set(0, 0, 0);
        camera.position.set(0,1.2,1.7);
    
        camera.lookAt(center);

        // Log blend shapes
        model.traverse(function (node) {
            if (node.name.includes('Mesh_4')) {
                        console.log({node});
                        nodes[node.name] = node;
                        console.log('AvatarHead')     
            } 
        });

        model.traverse((child)=>{ 
              if(child.isMesh && child.morphTargetInfluences){
                   morphTargetNames = Object.keys(child.morphTargetDictionary);
                   console.log('Morph target detected', morphTargetNames);
              }
        })
        
        mixer = new THREE.AnimationMixer(model);
        console.log({'anim' : gltf.animations})
        const specificClip = gltf.animations[0]; 
        mixer.clipAction(specificClip).play();


    

        render();
    });

    function render(){
            requestAnimationFrame(render);
               if(mixer){
                     mixer.update(0.01);
             }
            renderer.render(scene, camera);
    }


    overlay = document.getElementById('canvasOverlay');
    window.addEventListener('resize', () => {
        const width = window.innerWidth;
        const height = window.innerHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      });


    function animate(nodes , mixer) {
        requestAnimationFrame(() => animate(nodes , mixer));

        if(mixer){
               mixer.update(0.01);
        }
        renderer.render(scene, camera);
}
    animate(nodes , mixer);
}


function speechToText(callback) {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
  
    if (!SpeechRecognition) {
      throw new Error("SpeechRecognition is not supported in this browser.");
    }
  
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = true;
  
    recognition.onstart = () => {
      console.log("Speech recognition started.");

        waveform.style.display = 'flex';
        waveform.src = "assets/user_waveform.gif";
        waveform.style.width = "17vh";
        states.style.display = 'none';
    };
  
    recognition.onend = () => {
      console.log("Speech recognition stopped.");
    };
  
    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
        finalTranscript = transcript;
      }
      callback(transcript);
    };
  
    recognition.onerror = (event) => {
      console.error("Speech Recognition Error:", event.error);
      callback(null, event.error);
    };
  
    return {
      start: () => {
        console.log("Starting speech recognition...");
        recognition.start();
      },
      stop: () => {
        console.log("Stopping speech recognition...");
        recognition.stop();
      },
    };
  }
  
  const speech = speechToText((transcript, error) => {
    if (error) {
      console.error("Error:", error);
    } else {
      console.log("Transcript:", transcript);
    }
  });
  

let queue = Promise.resolve();

async function speak(transcript) {  
    queue = queue.then(async () => {
        const params = getQueryParams();
        console.log('ID:', params.id);
        console.log('transcript Final', transcript);
        const formData = new FormData();
        formData.append('transcript', transcript);
        formData.append('vector_id', params.id);
        formData.append('user_id', userId);
        formData.append('assistant_name' , params.assistant_name);
        formData.append('company_name', params.company_name);

        try {
            processingRequest = true;

            const response = await fetch(`${link}/getVoice`, {
                method: 'POST',
                body: formData,
                signal: signal,
            });

            if (!response.ok) {
                processingRequest = false;
                myvad.start();
                speech.start();
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            console.log(data);

            if (data.audio_base64) {        
                lipsyncAnimation(data.audio_base64 , data.alignment);
            } else {
                console.log("No audio data available.");
            }
   
        } catch (error) {
            states.textContent = 'Listening...';
            processingRequest = false;
            myvad.start();
            speech.start();
            if (error.name === 'AbortError') {
                console.log('Speak request was aborted');
            } else {
                console.error('Error:', error);
            }
        }
    });
}


function generatePhoneme(char) {
    const phonemeToMorph = {
        "a": "a",  "b": "p", "c": "k", "d": "d", 
        "e": "e",  "f": "f", "g": "E", "h": "i", 
        "i": "i",  "j": "p", "k": "k", "l": "k", 
        "m": "S",  "n": "k", "o": "o", "p": "p", 
        "q": "k",  "r": "r", "s": "s", "t": "t", 
        "u": "u",  "v": "E", "w": "uO", "x": "kf", 
        "y": "i",  "z": "rf"
    };

    return phonemeToMorph[char] || "p";  
}



async function lipsyncAnimation(audioUrl, alignment) {
    audio = new Audio(`data:audio/mpeg;base64,${audioUrl}`);

    const { characters, character_start_times_seconds, character_end_times_seconds } = alignment;
    const phonemeSequence = characters
        .map((char, i) => {
            const phoneme = generatePhoneme(char);
            return phoneme
                ? {
                      character: char,
                      phoneme,
                      startTime: character_start_times_seconds[i],
                      endTime: character_end_times_seconds[i],
                  }
                : null;
        })
        .filter(Boolean);


    const headMesh = model?.getObjectByName("head"); 
    if (!headMesh) {
        console.error("headMesh not found in the model.");
        return;
    }
    audio.play();

    audio.addEventListener('play', () => {

        waveform.style.display = 'flex';
        waveform.src = "assets/chatbot_wavform.gif";
        waveform.style.width = "17vh";

        states.style.display = 'none';
        console.log('Audio playback has started (play event triggered).');
    });
    

    audio.addEventListener("timeupdate", () => {
        const currentTime = audio.currentTime;
        const currentPhoneme = phonemeSequence.find(
            ({ startTime, endTime }) => currentTime >= startTime && currentTime < endTime
        );

        if (currentPhoneme) {
            const { phoneme } = currentPhoneme;
            for (let char of phoneme) {
                const morphTargetIndex = headMesh.morphTargetDictionary?.[char];
                if (morphTargetIndex !== undefined) {
                    headMesh.morphTargetInfluences[morphTargetIndex] = 1;
                    setTimeout(() => {
                        headMesh.morphTargetInfluences[morphTargetIndex] = 0;
                    }, 150);
                }
             }
        }
    });


    audio.addEventListener('ended', () => {
        console.log('audio ended');
        finalTranscript = "";
        processingRequest = false;
        intialAudioPlaying = false; 

        waveform.style.display = 'none';
        states.style.display = 'flex';
        states.textContent = 'Listening...';
        myvad.start();
        speech.start();
    }); 
}

           let myvad = null;
           async function toggleCanvasOverlay() {
            try {
                const overlay = document.getElementById('box');
              
                states.style.display = 'flex';
                states.textContent = 'Listening...'

                
                const isOverlayVisible = overlay.style.display === 'flex';
                if (!isOverlayVisible) {
                    let noSpeechTimeout = null; 
                    let isWaveformActive = false;

                    myvad = await vad.MicVAD.new({
                        positiveSpeechThreshold: 0.85,
                        minSpeechFrames: 8,
                        preSpeechPadFrames: 10,
                        onSpeechStart: () => {
                           
                        },
                        onFrameProcessed: (probs) => {
                        
                        },
                        onSpeechEnd: async (arr) => {
                            console.log("speech end");
                            console.log({intialAudioPlaying});
                            if(!intialAudioPlaying){
                                states.style.display = 'flex';
                                waveform.style.display = 'none';
                                states.textContent = 'Processing...'    
                                console.log('gone inside')
                                console.log({closedBtnclick , processingRequest , finalTranscript});
                                if (!closedBtnclick && !processingRequest && finalTranscript != "") {
                                    console.log('speak called');
                                    speech.stop();
                                    await speak(finalTranscript);
                                }
                            }else{
                                console.log("Intial Audio is already playing")
                            }
                        },
                    });
        
                    window.myvad = myvad;
        
                    // Start the VAD when the overlay is opened
                    window.toggleVAD = async() => {
                        console.log("ran toggle vad");
                        if (myvad.listening === false) {
                            myvad.start();
                            speech.stop();
                            console.log("VAD is running");
                            const openBtn = document.getElementById("openBtn");
                            openBtn.style.display = 'none';

                            intialAudioPlaying = true; 
                            
                            waveform.style.display = 'none';
                            states.style.display = 'flex';
                            states.textContent = 'Introducing...';
                            await speak("Introduce yourself in just 2 lines.")  
                        } else {
                            myvad.pause();
                            speech.stop();
                            console.log("VAD is paused");
                        }
                    };
        
                    // Start VAD when opening the overlay
                    window.toggleVAD();
                } else {
                    if (window.myvad) {
                        window.myvad.pause();
                        speech.stop();
                        console.log("VAD is paused");
                    }
                }
                overlay.style.display = isOverlayVisible ? 'none' : 'flex';

            } catch (e) {
                console.error("Failed:", e);
            }
        }

// Event listener for the Open Canvas button

openBtn.addEventListener('click', async () => {
    try {  

        const data = await callAuthorizeAPI();
        console.log({data})
        
        userId = data.user_id;
        if (data) {
            closedBtnclick = false;
            const imgOpen = document.getElementById('img-open');
            const loaderDiv = document.getElementById('line-loader');
            imgOpen.style.display = 'none';
            loaderDiv.style.display = 'flex';
            toggleCanvasOverlay(); 
        } else {
            console.warn('Authorization failed or no data received.');
        }

        controller = new AbortController();
        signal = controller.signal;
            

    } catch (error) {
        console.error('Error during the process:', error);
    }
});


const closeBtn = document.getElementById("closeBtn");
closeBtn.addEventListener('click', async () => {
    closedBtnclick = true; 
    toggleCanvasOverlay();
    
    const openBtn = document.getElementById("openBtn");
    const imgOpen = document.getElementById('img-open');
    const loaderDiv = document.getElementById('line-loader');
    imgOpen.style.display = 'flex';
    loaderDiv.style.display = 'none';
    openBtn.style.display = 'flex';

    if (audio) {
        audio.pause();
        audio.src = ""; 
        audio.remove();   
        audio = null;
    }

    myvad.pause();
    speech.stop();
    controller.abort();

    try {
        const response = await fetch(`${link}/removeUserId?user_id=${userId}`, {method: 'GET'});

        if (response.ok) {
            const result = await response.json();
            console.log("API Response:", result);
        } else {
            console.error("Failed to remove user ID:", response.statusText);
        }
    } catch (error) {
        console.error("Error calling API:", error);
    }
});


// Get the mic overlay element
const micOverlay = document.getElementById("mic-overlay2");
let isMicOn = true;
let mediaStream = null; 
function muteMic() {
    if (mediaStream) {
        console.log("Microphone muted");
        mediaStream.getAudioTracks().forEach(track => {
            track.enabled = false;  
        });
    }
}

function unmuteMic() {
    if (mediaStream) {
        console.log("Microphone unmuted");
        mediaStream.getAudioTracks().forEach(track => {
            track.enabled = true; 
        });
    }
}

// Event listener to toggle mic on click
micOverlay.addEventListener('click', () => {
    if (isMicOn) {
        muteMic();
        micOverlay.src = "assets/micoff.png";

        waveform.style.display = 'none';
        states.style.display = 'flex';
        states.textContent = 'Stopped....'
        processingRequest = false;
        myvad.pause();
        speech.stop();

        if(audio){
            audio.pause();
            audio.src = ""; 
            audio.remove(); 
            audio = null;
        }
        controller.abort();
        console.log("vad is stopped")

    } else {
        unmuteMic();
        micOverlay.src = "assets/mic.png";
   
        waveform.style.display = 'none';
        states.style.display = 'flex';
        states.textContent = 'Listening...'
        intialAudioPlaying = false;
        myvad.start();
        speech.start();
 
        controller = new AbortController();
        signal = controller.signal;
        console.log("vad is started")
    }
    isMicOn = !isMicOn;
});