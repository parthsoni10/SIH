import React, { useState } from 'react';
import { Cpu, Activity, ShieldAlert, Layers, ChevronDown, ChevronUp, AlertTriangle, XCircle } from 'lucide-react';

export default function AiImageAnalysisCard({ aiAnalysis, syntheticAnalysis }) {
  const [showDetails, setShowDetails] = useState(false);
  const data = aiAnalysis || syntheticAnalysis || {};

  const prob = data.probability ?? data.ai_probability ?? null;
  const globalProb = data.global_probability ?? prob;
  const patchTopk = data.patch_topk_probability ?? null;
  const freqAnomaly = data.frequency_anomaly ?? data.frequency_score ?? 0.0;
  const noiseAnomaly = data.noise_anomaly ?? data.noise_score ?? 0.0;
  const signalCount = data.strong_signal_count ?? 0;
  const strongSignals = data.strong_signals ?? [];
  const corroborated = data.corroborated ?? (signalCount >= 2);
  const modelVersion = data.model_version ?? "convnext_base_ai_detector_v1";
  const modelName = data.model_name ?? "ConvNeXt-Base";
  const modelLoaded = data.model_loaded ?? data.is_weights_loaded ?? false;
  const isTrained = data.trained ?? false;
  const isCalibrated = data.calibrated ?? false;
  const evidenceQuality = data.evidence_quality ?? "unavailable";
  const reasons = data.reasons ?? [];

  const getStatusBadge = () => {
    // Model unavailable
    if (!modelLoaded || prob === null || prob === undefined) {
      return (
        <span className="px-3 py-1 bg-slate-500/20 text-slate-400 border border-slate-500/40 rounded-full text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
          <XCircle className="w-3.5 h-3.5" /> DETECTOR UNAVAILABLE
        </span>
      );
    }
    // High AI + corroborated
    if (corroborated && prob >= 0.65) {
      return (
        <span className="px-3 py-1 bg-red-500/20 text-red-400 border border-red-500/40 rounded-full text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
          <ShieldAlert className="w-3.5 h-3.5" /> AI-GENERATED IMAGE
        </span>
      );
    }
    // High AI but single signal OR inconclusive range
    if (prob >= 0.35) {
      return (
        <span className="px-3 py-1 bg-amber-500/20 text-amber-400 border border-amber-500/40 rounded-full text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5" /> INCONCLUSIVE
        </span>
      );
    }
    // Low AI evidence — NOT "LIKELY REAL"
    return (
      <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 rounded-full text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
        <Activity className="w-3.5 h-3.5" /> LOW AI EVIDENCE
      </span>
    );
  };

  const formatProb = (val) => {
    if (val === null || val === undefined) return "N/A";
    return `${(val * 100).toFixed(1)}%`;
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-md space-y-5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-500/10 border border-indigo-500/30 rounded-lg text-indigo-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              AI Generation Detection
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Model: {modelName} · {modelVersion}
            </p>
          </div>
        </div>
        {getStatusBadge()}
      </div>

      {/* Main Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* AI Probability */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" /> AI Probability
            </span>
            <span className="text-sm font-bold text-indigo-400">{formatProb(globalProb)}</span>
          </div>
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                globalProb !== null && globalProb > 0.65 ? 'bg-red-500' :
                globalProb !== null && globalProb > 0.35 ? 'bg-amber-500' : 'bg-indigo-500'
              }`}
              style={{ width: `${globalProb !== null ? Math.min(100, Math.max(0, globalProb * 100)) : 0}%` }}
            />
          </div>
        </div>

        {/* Multi-Scale Patch Top-K */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" /> Patch Consistency (Top-K)
            </span>
            <span className="text-sm font-bold text-cyan-400">{formatProb(patchTopk)}</span>
          </div>
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-500 transition-all duration-500"
              style={{ width: `${patchTopk !== null ? Math.min(100, Math.max(0, patchTopk * 100)) : 0}%` }}
            />
          </div>
        </div>

        {/* 2D FFT Frequency Anomaly */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-purple-400" /> Frequency Anomaly
            </span>
            <span className="text-sm font-bold text-purple-400">{(freqAnomaly * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-500 transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, freqAnomaly * 100))}%` }}
            />
          </div>
        </div>

        {/* Corroborating Signals */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-slate-300 block mb-1">
              Corroborating Signals
            </span>
            <p className="text-xs text-slate-400">
              Corroborated: <span className={corroborated ? "text-red-400 font-bold" : "text-emerald-400 font-bold"}>{corroborated ? "YES" : "NO"}</span>
            </p>
          </div>
          <div className="text-right">
            <span className={`text-xl font-extrabold ${signalCount >= 2 ? 'text-red-400' : 'text-emerald-400'}`}>
              {signalCount} / 4
            </span>
          </div>
        </div>
      </div>

      {/* Model Status Bar */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3 text-center">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 block mb-1">Model Loaded</span>
          <span className={`text-sm font-bold ${modelLoaded ? 'text-emerald-400' : 'text-red-400'}`}>
            {modelLoaded ? 'YES' : 'NO'}
          </span>
        </div>
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3 text-center">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 block mb-1">Trained</span>
          <span className={`text-sm font-bold ${isTrained ? 'text-emerald-400' : 'text-amber-400'}`}>
            {isTrained ? 'YES' : 'NO'}
          </span>
        </div>
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3 text-center">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 block mb-1">Calibration</span>
          <span className={`text-sm font-bold ${isCalibrated ? 'text-emerald-400' : 'text-amber-400'}`}>
            {isCalibrated ? 'CALIBRATED' : 'NOT CALIBRATED'}
          </span>
        </div>
      </div>

      {/* Expandable Technical Evidence */}
      <div className="pt-2">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full flex items-center justify-between p-3 bg-slate-950/60 hover:bg-slate-800/60 rounded-lg border border-slate-800 text-xs text-slate-300 transition-all"
        >
          <span className="font-semibold flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            Inspect AI Evidence & Reason Codes
          </span>
          {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showDetails && (
          <div className="mt-3 p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-3 animate-float-in">
            <div className="text-xs font-mono space-y-1 text-slate-300">
              <p>• Global AI Probability: <span className="font-bold text-indigo-400">{formatProb(globalProb)}</span></p>
              <p>• Patch Mean Probability: <span className="font-bold text-cyan-400">{formatProb(data.patch_mean_probability)}</span></p>
              <p>• Patch Median Probability: <span className="font-bold text-cyan-400">{formatProb(data.patch_median_probability)}</span></p>
              <p>• Spatial Noise Anomaly: <span className="font-bold text-purple-400">{(noiseAnomaly * 100).toFixed(1)}%</span></p>
              <p>• Evidence Quality: <span className="font-bold text-slate-200">{evidenceQuality.toUpperCase()}</span></p>
            </div>

            {strongSignals.length > 0 && (
              <div className="pt-2 border-t border-slate-800">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                  Strong Signals Detected
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {strongSignals.map((s, idx) => (
                    <span key={idx} className="px-2 py-0.5 bg-red-500/10 text-red-400 rounded border border-red-500/30 text-[11px] font-mono">
                      ⚠ {s}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {reasons && reasons.length > 0 && (
              <div className="pt-2 border-t border-slate-800">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                  Evidence Reason Codes
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {reasons.map((r, idx) => (
                    <span key={idx} className="px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700 text-[11px] font-mono">
                      • {r}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
