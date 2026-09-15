import React from 'react';
import { Upload, Cpu, ShieldCheck, CheckCircle2, AlertTriangle, FileText, UserCheck, Layers, Award } from 'lucide-react';

export default function EvidenceTimeline({ decision, riskScore, modelVersions }) {
  const steps = [
    { label: "Upload", icon: Upload, status: "completed" },
    { label: "Preprocessing", icon: Layers, status: "completed" },
    { label: "OCR & MRZ", icon: FileText, status: "completed" },
    { label: "Validation", icon: ShieldCheck, status: "completed" },
    { label: "AI Detection", icon: Cpu, status: "completed" },
    { label: "Forensic Fusion", icon: CheckCircle2, status: "completed" },
    { label: "Face Match", icon: UserCheck, status: "completed" },
    { label: "Risk Model", icon: AlertTriangle, status: "completed" },
    { label: "Decision Engine", icon: Award, status: "completed" },
  ];

  const finalStatus = decision?.status || "genuine";

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-5 border-b border-slate-800/80 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            Verification Pipeline Timeline
          </h3>
          <p className="text-xs text-slate-400">9-Stage End-to-End Forensics & Model Decision Chain</p>
        </div>
        {modelVersions && (
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
            <span>Risk: {modelVersions.risk_model || "v2"}</span>
            <span className="text-slate-600">|</span>
            <span>AI: {modelVersions.ai_detector || "v1"}</span>
          </div>
        )}
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="flex items-center justify-between min-w-[700px]">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isLast = idx === steps.length - 1;
            return (
              <React.Fragment key={idx}>
                <div className="flex flex-col items-center gap-2 relative group">
                  <div className="w-9 h-9 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-400 flex items-center justify-center font-bold shadow-lg shadow-indigo-500/10">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-medium text-slate-300 text-center whitespace-nowrap">
                    {step.label}
                  </span>
                </div>
                {!isLast && (
                  <div className="flex-1 h-0.5 bg-gradient-to-r from-indigo-500/50 to-slate-700 mx-2" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
}
