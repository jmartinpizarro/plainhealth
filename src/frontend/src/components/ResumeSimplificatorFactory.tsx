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
        <div className="mt-4 w-[min(90vw,760px)] border-2 border-solid rounded-lg border-gray-100 px-3 py-4 bg-white text-left"
        >
            <strong>Informe Simplificado</strong>
            <p className="mt-2">
                Estado: {simplifiedMedicalReportStatus}
            </p>
            <pre className="mt-2 text-base/7 whitespace-pre-wrap bg-[#f8f8f8] overflow-x-auto p-3 rounded-lg">
                {simplifiedMedicalReport || 'Aun no disponible. Se generara automaticamente al cerrar la grabacion.'}
            </pre>
        </div>
    )
}

export default ResumeSimplificatorFactory