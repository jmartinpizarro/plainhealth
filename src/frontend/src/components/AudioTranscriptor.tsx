/**
 * This script contains the module AudioTranscriptor, used for recording the conversation between
 * the medic and the patient for then being sent to the backend
 */

import { useState, useRef } from "react"

import MedicalResumeFactory from "./MedicalResumeFactory";

const API_URL = "http://localhost:5000/api"

function AudioTranscriptor() {
    const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [fullTranscript, setFullTranscript] = useState('');
    const [medicalReport, setMedicalReport] = useState('');
    const [medicalReportStatus, setMedicalReportStatus] = useState<'idle' | 'processing' | 'ready' | 'error'>('idle');

    /**
     * Internally, it was kind of complicated of sending each chunk to the API, decode it and 
     * process it (because of headers). So what it is done here is to process all the chunks 
     * recorded in 1000ms intervals. The last chunk (isLast), is the one which is sent to the 
     * backend in order to be processed. 
     * TODO: this can be optimised by recording the entire audio as an unique chunk, 
     * not as fragments
     */

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

            setFullTranscript('');
            setMedicalReport('');
            setMedicalReportStatus('idle');

            // a random sessionId per audio sent - this might not make any sense at the moment but 
            // because is a PoC and there are not any users, fuck it, we ball
            const currentSessionId = self.crypto.randomUUID();

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    allChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                // TODO: do i really want this?
                const fullBlob = new Blob(allChunksRef.current, { type: 'audio/webm' });
                const url = URL.createObjectURL(fullBlob);
                setRecordedAudioUrl(url);

                // Send only once when recording stops.
                await processChunk(fullBlob, 0, true, currentSessionId);

                // stop using the microphone
                stream.getTracks().forEach(track => track.stop());
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
        if (isLast) {
            setMedicalReportStatus('processing');

            const formData = new FormData();

            formData.append('session_id', currentSessionId); // session_id
            formData.append('chunk_index', chunkCounter.toString()); // chunk_index
            formData.append('is_last', isLast ? 'true' : 'false'); // is_last
            formData.append('audio', chunk, 'chunk.webm'); // audio

            // fetch transcription
            try {
                const response = await fetch(API_URL + '/transcribe_audio', { method: 'POST', body: formData });
                if (!response.ok) {
                    const details = await response.text();
                    throw new Error(`Response status: ${response.status} - ${details}`);
                }
                const result = await response.json();

                if (result.is_last) { // asume always true
                    setFullTranscript((result.full_transcript || '').trim());
                    const transcriptResult = result.full_transcript;

                    // fetch resume by llm
                    try {
                        const resumeFormData = new FormData();
                        resumeFormData.append('transcription', transcriptResult);
                        const response = await fetch(API_URL + '/generate_resume', { method: 'POST', body: resumeFormData })

                        if (!response.ok) {
                            const details = await response.text();
                            throw new Error(`Response status: ${response.status} - ${details}`);
                        }

                        const result = await response.json();

                        if (result.medical_report) {
                            setMedicalReport(String(result.medical_report).trim());
                            setMedicalReportStatus('ready');
                        } else {
                            setMedicalReport('No se pudo generar informe medico. Revisa la API key (GOOGLE_API_KEY/ENV_API_KEY/GEMINI_API_KEY) y el log del backend.');
                            setMedicalReportStatus('error');
                        }

                    } catch (error) {
                        if (isLast) {
                            setMedicalReport('Error al solicitar la transcripcion/informe. Revisa la consola del navegador y del backend.');
                            setMedicalReportStatus('error');
                        }
                    }



                }
            } catch (error) {
                if (isLast) {
                    setMedicalReport('Error al solicitar la transcripcion/informe. Revisa la consola del navegador y del backend.');
                    setMedicalReportStatus('error');
                }
                console.log(error)
            }
        }
    };


    return (
        <>
            <section>
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

                <div
                    style={{
                        marginTop: '1rem',
                        width: 'min(90vw, 760px)',
                        border: '1px solid #ddd',
                        borderRadius: '12px',
                        padding: '0.75rem 1rem',
                        background: '#fff',
                        textAlign: 'left',
                    }}
                >
                    <strong>Transcripcion final</strong>
                    <p style={{ marginTop: '0.5rem', lineHeight: 1.4, whiteSpace: 'pre-wrap' }}>
                        {fullTranscript || 'Aun no disponible. Aparecera al parar la grabacion.'}
                    </p>
                </div>
            </section>

            <MedicalResumeFactory report={medicalReport} reportStatus={medicalReportStatus} />
        </>




    )
}

export default AudioTranscriptor
