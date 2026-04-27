import { useEffect, useState } from "react";

type MedicalResumeFactoryProps = {
    report: string;
    reportStatus: 'idle' | 'processing' | 'ready' | 'error';
};

function MedicalResumeFactory({report, reportStatus} : MedicalResumeFactoryProps) {

    const [medicalReport, setMedicalReport] = useState('');
    const [medicalReportStatus, setMedicalReportStatus] = useState<'idle' | 'processing' | 'ready' | 'error'>('idle');

    useEffect(() => {
        setMedicalReport(report),
        setMedicalReportStatus(reportStatus)
    }, [report, reportStatus])

    return (
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
            <strong>Informe medico estructurado (Gemini)</strong>
            <p style={{ marginTop: '0.5rem' }}>
                Estado: {medicalReportStatus}
            </p>
            <pre
                style={{
                    marginTop: '0.5rem',
                    lineHeight: 1.35,
                    whiteSpace: 'pre-wrap',
                    overflowX: 'auto',
                    background: '#f8f8f8',
                    borderRadius: '8px',
                    padding: '0.75rem',
                }}
            >
                {medicalReport || 'Aun no disponible. Se generara automaticamente al cerrar la grabacion.'}
            </pre>
        </div>
    )
}

export default MedicalResumeFactory