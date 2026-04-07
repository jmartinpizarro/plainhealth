import { useState, useRef } from 'react'
import uc3m from './assets/uc3m.png'
import './App.css'

function App() {
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [chunks, setChunks] = useState<Blob[]>([]);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const allChunksRef = useRef<Blob[]>([]); // for saving the chunks

  const startRecording = async () => {
    try {
      // open the audio stream with some configs that do not work correctly, love it
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          noiseSuppression: true,
          echoCancellation: true,
        }
      });

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      allChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          allChunksRef.current.push(event.data);
          setChunks(prev => [...prev, event.data]);
          
          // process chunks here
          processChunk(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        // TODO: do i really want this?
        const fullBlob = new Blob(allChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(fullBlob);
        setRecordedAudioUrl(url);
        
        // stop using the microphone
        stream.getTracks().forEach(track => track.stop());
      };

      // set up the chunk size
      mediaRecorder.start(2000);
      setIsRecording(true);

    } catch (err) {
      console.error('Error when accessing to the microphone', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processChunk = async (chunk: Blob) => {
    // const formData = new FormData();
    // formData.append('audio', chunk, 'chunk.webm');
    // await fetch('/api/transcribe', { method: 'POST', body: formData });
    console.log('Procesando chunk de', chunk.size, 'bytes');
  };

  return (
    <>
      <section id="center">
        <div className="hero">
          <img src={uc3m} className="base" width="500" height="200" alt="Logo de Universidad Carlos III de Madrid" />
        </div>
        <div>
          <h1>Transcribe en tiempo real</h1>
          <p>
            Pulsa el botón de <code>GRABAR</code> y comienza a hablar.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
          <button onClick={startRecording} disabled={isRecording}>
            Grabar
          </button>
          <button onClick={stopRecording} disabled={!isRecording}>
            Parar
          </button>
        </div>

        {isRecording && (
          <p style={{ color: 'red' }}>
            Grabando...
          </p>
        )}

        {recordedAudioUrl && (
          <audio controls src={recordedAudioUrl} style={{ marginTop: '1rem' }} />
        )}
      </section>
      <div className="ticks"></div>
      <div className="ticks"></div>
      <section id="spacer"></section>
    </>
  );
}

export default App