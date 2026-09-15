import React, { useState } from 'react';
import { ArrowLeft, Code, FileText, Calendar, CheckCircle2, ShieldAlert, Cpu, Layers, Activity, FileCheck, Sliders } from 'lucide-react';
import RiskScoreBadge from '../components/RiskScoreBadge';
import ExplanationBanner from '../components/ExplanationBanner';
import FieldTable from '../components/FieldTable';
import ForensicCard from '../components/ForensicCard';
import OcrExtractedView from '../components/OcrExtractedView';
import AiImageAnalysisCard from '../components/AiImageAnalysisCard';
import EvidenceTimeline from '../components/EvidenceTimeline';

export default function ResultPage({ resultData, onBackToUpload }) {
  const [showVectorModal, setShowVectorModal] = useState(false);

  if (!resultData) return null;

  const formattedDate = new Date(resultData.created_at || Date.now()).toLocaleString();
  const decision = resultData.decision || { status: resultData.prediction || "MANUAL_REVIEW", confidence: 0.65, reason_codes: [] };
  const docVal = resultData.document_validity || { score: resultData.validation_pass_rate || 1.0, ocr_confidence: resultData.ocr_confidence || 0.9, structural_validity: true };
  const aiVal = resultData.ai_analysis || resultData.synthetic_analysis || { probability: resultData.ai_probability || 0.0 };
  const tampVal = resultData.tampering_analysis || { probability: resultData.tampering_score || 0.0 };
  const qualityVal = resultData.image_quality || { score: 0.80, blur_score: 120.0 };

  return (
    <div className="max-w-7xl mx-auto space-y-6 py-6 px-4">
      {/* Top Bar Navigation & Info */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-5 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-4">
          <button
            onClick={onBackToUpload}
            className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-all flex items-center space-x-2 text-xs font-semibold"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Screen Another Document</span>
          </button>

          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Verification Result #{resultData.id}</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono font-normal">
                {resultData.document_type}
              </span>
            </h2>
            <p className="text-xs text-slate-400 font-mono flex items-center space-x-1">
              <Calendar className="w-3.5 h-3.5 text-cyan-400" />
              <span>Evaluated: {formattedDate}</span>
            </p>
          </div>
        </div>

        <button
          onClick={() => setShowVectorModal(true)}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-cyan-400 border border-cyan-500/20 text-xs font-mono transition-all"
        >
          <Code className="w-4 h-4" />
          <span>Inspect Feature Payload</span>
        </button>
      </div>

      {/* Verification Evidence Timeline */}
      <EvidenceTimeline
        decision={decision}
        riskScore={resultData.risk_score}
        modelVersions={resultData.model_versions}
      />

      {/* Gemini / Offline AI Officer Summary Banner */}
      <ExplanationBanner
        explanation={resultData.explanation || resultData.gemini?.summary}
        riskScore={resultData.risk_score}
        gemini={resultData.gemini}
      />

      {/* Four Distinct Metric Score Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: DOCUMENT VALIDITY */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur-md">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <FileCheck className="w-4 h-4 text-emerald-400" /> Document Validity
            </span>
            <span className="text-sm font-extrabold text-emerald-400">
              {Math.round((docVal.score ?? 1.0) * 100)}%
            </span>
          </div>
          <p className="text-xs font-semibold text-slate-400">
            Status: <span className={docVal.structural_validity ? "text-emerald-400" : "text-amber-400"}>
              {docVal.structural_validity ? "STRUCTURALLY VALID" : "CHECK FAILED"}
            </span>
          </p>
        </div>

        {/* Card 2: AI GENERATION */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur-md">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-indigo-400" /> AI Generation
            </span>
            <span className="text-sm font-extrabold text-indigo-400">
              {((aiVal.probability ?? 0.0) * 100).toFixed(1)}%
            </span>
          </div>
          <p className="text-xs font-semibold text-slate-400">
            Status: <span className={
              aiVal.status === "AI_GENERATED" || (aiVal.probability ?? 0.0) >= 0.65
                ? "text-red-400"
                : aiVal.status === "INCONCLUSIVE" || ((aiVal.probability ?? 0.0) >= 0.35 && (aiVal.probability ?? 0.0) < 0.65)
                ? "text-amber-400"
                : "text-emerald-400"
            }>
              {
                aiVal.status === "AI_GENERATED" || (aiVal.probability ?? 0.0) >= 0.65
                  ? "AI GENERATED"
                  : aiVal.status === "INCONCLUSIVE" || ((aiVal.probability ?? 0.0) >= 0.35 && (aiVal.probability ?? 0.0) < 0.65)
                  ? "INCONCLUSIVE FORENSICS"
                  : "LOW AI EVIDENCE"
              }
            </span>
          </p>
        </div>

        {/* Card 3: DIGITAL TAMPERING */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur-md">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 text-rose-400" /> Digital Tampering
            </span>
            <span className="text-sm font-bold text-rose-400">
              {((tampVal.probability ?? 0.0) * 100).toFixed(1)}%
            </span>
          </div>
          <p className="text-xs font-semibold text-slate-400">
            Status: <span className={(tampVal.probability ?? 0.0) >= 0.70 ? "text-red-400" : "text-emerald-400"}>
              {(tampVal.probability ?? 0.0) >= 0.70 ? "ALTERED DOCUMENT" : "LOW TAMPERING"}
            </span>
          </p>
        </div>

        {/* Card 4: IMAGE QUALITY */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur-md">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-cyan-400" /> Image Quality
            </span>
            <span className="text-sm font-bold text-cyan-400">
              {Math.round((qualityVal.score ?? 0.8) * 100)}%
            </span>
          </div>
          <p className="text-xs font-semibold text-slate-400">
            Status: <span className={(qualityVal.score ?? 0.8) >= 0.45 ? "text-emerald-400" : "text-amber-400"}>
              {(qualityVal.score ?? 0.8) >= 0.45 ? "ACCEPTABLE QUALITY" : "DEGRADED QUALITY"}
            </span>
          </p>
        </div>
      </div>

      {/* Main Grid: Gauge + Forensic Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Risk/Confidence Gauge */}
        <div className="lg:col-span-1 space-y-6">
          <RiskScoreBadge
            riskScore={resultData.risk_score}
            prediction={resultData.prediction}
            decision={decision}
          />

          {/* Decision Reason Codes (WHY?) */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3 font-mono text-xs">
            <div className="text-slate-300 font-sans font-bold border-b border-slate-800 pb-2 flex items-center justify-between">
              <span>Why This Decision?</span>
              <span className="text-[10px] text-slate-400 font-mono">Reason Codes</span>
            </div>

            {decision.reason_codes && decision.reason_codes.length > 0 ? (
              <div className="space-y-1.5 pt-1">
                {decision.reason_codes.map((code, idx) => (
                  <div key={idx} className="flex items-center space-x-2 text-slate-300 text-[11px]">
                    <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="font-semibold">{code}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-400 text-xs">Document integrity and security checks verified.</p>
            )}
          </div>
        </div>

        {/* Right Column: Extracted Fields & Forensic Details */}
        <div className="lg:col-span-2 space-y-6">
          <AiImageAnalysisCard
            aiAnalysis={resultData.ai_analysis}
            syntheticAnalysis={resultData.synthetic_analysis}
          />

          <ForensicCard
            tamperingScore={tampVal.probability ?? resultData.tampering_score ?? 0.0}
            tamperingSignals={tampVal.signals || resultData.tampering_signals || {
              ela_score: tampVal.ela_score || 0.0,
              noise_inconsistency: tampVal.noise_inconsistency || 0.0,
              copy_move_score: tampVal.copy_move_score || 0.0,
              splicing_score: tampVal.splicing_score || 0.0,
              metadata_score: tampVal.metadata_score || 0.0
            }}
            idChecksumValid={docVal.checksum_valid ?? resultData.id_checksum_valid}
            expiryValid={docVal.expiry_valid ?? resultData.expiry_valid}
            validationPassRate={docVal.score ?? resultData.validation_pass_rate}
            faceMatchScore={resultData.face_verification?.score ?? resultData.face_match_score}
            blacklistHit={resultData.blacklist_hit}
            failedRules={docVal.failed_rules ?? resultData.failed_rules}
            layoutScore={docVal.layout_score ?? resultData.layout_score}
            layoutAnomalies={resultData.layout_anomalies}
          />

          <OcrExtractedView
            rawText={resultData.raw_text}
            llmValidation={resultData.llm_validation}
            warnings={resultData.preprocessing_warnings}
          />

          <FieldTable fields={resultData.extracted_fields} />
        </div>
      </div>

      {/* Feature Vector JSON Inspector Modal */}
      {showVectorModal && (
        <div className="fixed inset-0 z-50 glass-panel bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-card max-w-2xl w-full p-6 rounded-3xl border border-slate-800 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2 text-cyan-400 font-mono text-sm font-semibold">
                <Cpu className="w-5 h-5" />
                <span>Target Architecture Forensic Payload</span>
              </div>
              <button
                onClick={() => setShowVectorModal(false)}
                className="text-slate-400 hover:text-white text-xs font-mono px-2 py-1 bg-slate-800 rounded-lg"
              >
                Close (ESC)
              </button>
            </div>

            <pre className="p-4 rounded-2xl bg-slate-950 border border-slate-800 text-xs font-mono text-cyan-300 overflow-x-auto max-h-96">
              {JSON.stringify(resultData.feature_vector || resultData, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
