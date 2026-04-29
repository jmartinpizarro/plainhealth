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
        <div className="mt-4 w-[min(90vw,760px)] border-2 border-solid rounded-lg border-gray-100 px-3 py-4 bg-white text-left">
            <strong>Informe médico</strong>
            <p className="mt-2">
                Estado: {medicalReportStatus}
            </p>
            <pre className="mt-2 text-base/7 whitespace-pre-wrap bg-[#f8f8f8] overflow-x-auto p-3 rounded-lg"
            >
                {medicalReport || 'Aun no disponible. Se generara automaticamente al cerrar la grabacion.'}
            </pre>
        </div>
    )
}

export default MedicalResumeFactory