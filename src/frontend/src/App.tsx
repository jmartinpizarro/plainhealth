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
  const isStoppingRef = useRef(false);
  const sendQueueRef = useRef<Promise<void>>(Promise.resolve());
  const lastHandledChunkRef = useRef(-1);

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
      lastHandledChunkRef.current = -1;

      // a random sessionId per audio sent - this might not make any sense at the moment but 
      // because is a PoC and there are not any users, fuck it, we ball
      const currentSessionId = self.crypto.randomUUID();
      sessionIdRef.current = currentSessionId;
      setSessionId(currentSessionId);
      
      let chunk_counter = 0;
      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          allChunksRef.current.push(event.data);
          setChunks(prev => [...prev, event.data]);

          // Send a self-contained chunk for each request.
          // Timesliced blobs are not always independently decodable in isolation.
          const cumulativeBlob = new Blob(allChunksRef.current, { type: 'audio/webm' });
          const isLastChunk = isStoppingRef.current;
          const currentChunkIndex = chunk_counter;
          sendQueueRef.current = sendQueueRef.current.then(() =>
            processChunk(cumulativeBlob, currentChunkIndex, isLastChunk, currentSessionId)
          );
          chunk_counter++;
        }
      };

      mediaRecorder.onstop = () => {
        // TODO: do i really want this?
        const fullBlob = new Blob(allChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(fullBlob);
        setRecordedAudioUrl(url);
        
        // stop using the microphone
        stream.getTracks().forEach(track => track.stop());
        isStoppingRef.current = false;
      };

      // set up the chunk size
      mediaRecorder.start(1000);
      setIsRecording(true);

    } catch (err) {
      console.error('Error when accessing to the microphone', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      isStoppingRef.current = true;
      // Flush pending data before stopping.
      mediaRecorderRef.current.requestData();
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processChunk = async (
    chunk: Blob,
    chunkCounter: number,
    isLast: boolean,
    currentSessionId: string,
  ) => {
    const formData = new FormData();

    formData.append('session_id', currentSessionId); // session_id
    formData.append('chunk_index', chunkCounter.toString()); // chunk_index
    formData.append('is_last', isLast ? 'true' : 'false'); // is_last
    formData.append('audio', chunk, 'chunk.webm'); // audio

    try {
      const response = await fetch(API_URL+'/audio', { method: 'POST', body: formData });
      if (!response.ok) {
        const details = await response.text();
        throw new Error(`Response status: ${response.status} - ${details}`);
      }
      const result = await response.json();
      console.log(result)

      if (result.chunk_index > lastHandledChunkRef.current) {
        lastHandledChunkRef.current = result.chunk_index;
        const chunkText = (result.text || '').trim();
        if (chunkText) {
          setLiveTranscript(chunkText);
        }
      }
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
          <strong>Transcripción en vivo</strong>
          <p style={{ marginTop: '0.5rem', lineHeight: 1.4, whiteSpace: 'pre-wrap' }}>
            {liveTranscript || 'Aquí aparecerá el texto reconocido...'}
          </p>
          <small>Sesión: {sessionId || '-'}</small>
          <br />
          <small>Chunks enviados: {chunks.length}</small>
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