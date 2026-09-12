import React, { useState } from 'react';
import { ArrowLeft, Code, FileText, Calendar, CheckCircle2, ShieldAlert, Cpu } from 'lucide-react';
import RiskScoreBadge from '../components/RiskScoreBadge';
import ExplanationBanner from '../components/ExplanationBanner';
import FieldTable from '../components/FieldTable';
import ForensicCard from '../components/ForensicCard';

export default function ResultPage({ resultData, onBackToUpload }) {
  const [showVectorModal, setShowVectorModal] = useState(false);

  if (!resultData) return null;

  const formattedDate = new Date(resultData.created_at || Date.now()).toLocaleString();

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

        {/* Technical Vector Button */}
        <button
          onClick={() => setShowVectorModal(true)}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-cyan-400 border border-cyan-500/20 text-xs font-mono transition-all"
        >
          <Code className="w-4 h-4" />
          <span>Inspect 12-Feature Vector</span>
        </button>
      </div>

      {/* Gemini AI Officer Summary Banner */}
      <ExplanationBanner
        explanation={resultData.explanation}
        riskScore={resultData.risk_score}
      />

      {/* Main Grid: Gauge + Forensic Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Risk Gauge */}
        <div className="lg:col-span-1 space-y-6">
          <RiskScoreBadge
            riskScore={resultData.risk_score}
            prediction={resultData.prediction}
          />

          {/* Quick Metrics Summary */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3 font-mono text-xs">
            <div className="text-slate-400 font-sans font-semibold border-b border-slate-800 pb-2">
              Model Signals Summary
            </div>

            <div className="flex justify-between">
              <span className="text-slate-400">OCR Mean Confidence:</span>
              <span className="text-slate-200 font-bold">{Math.round(resultData.ocr_confidence * 100)}%</span>
            </div>

            <div className="flex justify-between">
              <span className="text-slate-400">Validation Pass Rate:</span>
              <span className="text-slate-200 font-bold">{Math.round(resultData.validation_pass_rate * 100)}%</span>
            </div>

            <div className="flex justify-between">
              <span className="text-slate-400">Tampering Index:</span>
              <span className="text-slate-200 font-bold">{(resultData.tampering_score * 100).toFixed(1)}%</span>
            </div>

            <div className="flex justify-between">
              <span className="text-slate-400">Fraud Probability:</span>
              <span className="text-rose-400 font-bold">{(resultData.probability * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Right Column: Extracted Fields & Forensic Details */}
        <div className="lg:col-span-2 space-y-6">
          <ForensicCard
            tamperingScore={resultData.tampering_score}
            tamperingSignals={resultData.tampering_signals}
            idChecksumValid={resultData.id_checksum_valid}
            expiryValid={resultData.expiry_valid}
            validationPassRate={resultData.validation_pass_rate}
            faceMatchScore={resultData.face_match_score}
            blacklistHit={resultData.blacklist_hit}
            failedRules={resultData.failed_rules}
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
                <span>RandomForest 12-Feature Vector Payload</span>
              </div>
              <button
                onClick={() => setShowVectorModal(false)}
                className="text-slate-400 hover:text-white text-xs font-mono px-2 py-1 bg-slate-800 rounded-lg"
              >
                Close (ESC)
              </button>
            </div>

            <pre className="p-4 rounded-2xl bg-slate-950 border border-slate-800 text-xs font-mono text-cyan-300 overflow-x-auto max-h-96">
              {JSON.stringify(resultData.feature_vector, null, 2)}
            </pre>

            <p className="text-[11px] text-slate-400 font-mono text-right">
              Model Contract: RandomForestClassifier (200 trees, scikit-learn 1.6.1)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
