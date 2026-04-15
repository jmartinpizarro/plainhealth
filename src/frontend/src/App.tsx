import { useState, useRef } from 'react'
import uc3m from './assets/uc3m.png'
import './App.css'

const API_URL = "http://localhost:5000/api"

function App() {
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [chunks, setChunks] = useState<Blob[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [liveTranscript, setLiveTranscript] = useState('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const allChunksRef = useRef<Blob[]>([]); // for saving the chunks
  const sessionIdRef = useRef<string>('');

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
      setChunks([]);
      setLiveTranscript('');

      // a random sessionId per audio sent - this might not make any sense at the moment but 
      // because is a PoC and there are not any users, fuck it, we ball
      const currentSessionId = self.crypto.randomUUID();
      sessionIdRef.current = currentSessionId;
      setSessionId(currentSessionId);
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          allChunksRef.current.push(event.data);
          setChunks(prev => [...prev, event.data]);
        }
      };

      mediaRecorder.onstop = () => {
        const fullBlob = new Blob(allChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(fullBlob);
        setRecordedAudioUrl(url);
        void processFullAudio(fullBlob, currentSessionId);
        
        // stop using the microphone
        stream.getTracks().forEach(track => track.stop());
      };

      // Keep chunks only for local buffering; upload happens once on stop.
      mediaRecorder.start(1000);
      setIsRecording(true);

    } catch (err) {
      console.error('Error when accessing to the microphone', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      // Flush pending data before stopping.
      mediaRecorderRef.current.requestData();
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processFullAudio = async (audioBlob: Blob, currentSessionId: string) => {
    const formData = new FormData();

    formData.append('session_id', currentSessionId);
    formData.append('audio', audioBlob, 'recording.webm');

    try {
      const response = await fetch(API_URL+'/audio', { method: 'POST', body: formData });
      if (!response.ok) {
        const details = await response.text();
        throw new Error(`Response status: ${response.status} - ${details}`);
      }
      const result = await response.json();
      setLiveTranscript((result.text || '').trim());
    } catch (error) {
      console.log(error)
    }
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

        <div
          style={{
            marginTop: '1rem',
            width: 'min(90vw, 760px)',
            minHeight: '120px',
            border: '1px solid #ddd',
            borderRadius: '12px',
            padding: '0.75rem 1rem',
            background: '#fff',
            textAlign: 'left',
          }}
        >
          <strong>Transcripción</strong>
          <p style={{ marginTop: '0.5rem', lineHeight: 1.4, whiteSpace: 'pre-wrap' }}>
            {liveTranscript || 'Aquí aparecerá el texto reconocido...'}
          </p>
          <small>Sesión: {sessionId || '-'}</small>
          <br />
          <small>Chunks grabados: {chunks.length}</small>
        </div>

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