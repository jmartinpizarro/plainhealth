import AudioTranscriptor from './components/AudioTranscriptor'
import './App.css'

function App() {

  return (
    <>
      <section id="center">
        <div>
          <h1 >Transcribe en tiempo real</h1>
          <p>
            Pulsa el botón de <code>GRABAR</code> y comienza a hablar.
          </p>
        </div>

        <AudioTranscriptor /> {/* module used for audio transcription */}

      </section>
      <div className="ticks"></div>
      <div className="ticks"></div>
      <section id="spacer"></section>
    </>
  );
}

export default App