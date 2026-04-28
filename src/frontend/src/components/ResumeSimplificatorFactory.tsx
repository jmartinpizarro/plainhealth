import { useEffect, useState } from "react";

type ResumeSimplificatorFactoryProps = {
    report: string;
    reportStatus: 'idle' | 'processing' | 'ready' | 'error';
};

function ResumeSimplificatorFactory({ report, reportStatus }: ResumeSimplificatorFactoryProps) {

    const [simplifiedMedicalReport, setSimplifiedMedicalReport] = useState('');
    const [simplifiedMedicalReportStatus, setSimplifiedMedicalReportStatus] = useState<'idle' | 'processing' | 'ready' | 'error'>('idle');

    useEffect(() => {
        setSimplifiedMedicalReport(report),
            setSimplifiedMedicalReportStatus(reportStatus)
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
            <strong>Informe Simplificado</strong>
            <p style={{ marginTop: '0.5rem' }}>
                Estado: {simplifiedMedicalReportStatus}
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
                {simplifiedMedicalReport || 'Aun no disponible. Se generara automaticamente al cerrar la grabacion.'}
            </pre>
        </div>
    )
}

export default ResumeSimplificatorFactory