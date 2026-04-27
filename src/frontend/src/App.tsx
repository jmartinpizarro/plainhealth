import uc3m from './assets/uc3m.png'

import AudioTranscriptor from './components/AudioTranscriptor'
import './App.css'

function App() {

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

        <AudioTranscriptor /> {/* module used for audio transcription */}

      </section>
      <div className="ticks"></div>
      <div className="ticks"></div>
      <section id="spacer"></section>
    </>
  );
}

export default App