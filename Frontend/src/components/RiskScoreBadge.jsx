import React from 'react';
import { AlertTriangle, CheckCircle2, ShieldAlert, Cpu, Eye, FileX } from 'lucide-react';

export default function RiskScoreBadge({ riskScore, prediction, decision }) {
  const status = (decision?.status || prediction || "MANUAL_REVIEW").toUpperCase();
  const rawConfidence = decision?.decision_confidence ?? decision?.confidence ?? null;
  const confidence = rawConfidence !== null ? Math.round(rawConfidence * 100) : null;
  const displayConfidence = confidence !== null ? confidence : null;

  const displayRiskScore = riskScore !== undefined && riskScore !== null ? riskScore : (
    status === 'GENUINE' ? 10 :
    status === 'INCOMPLETE_SUBMISSION' ? 35 :
    status === 'MANUAL_REVIEW' ? 60 :
    status === 'SUSPICIOUS' ? 75 :
    status === 'ALTERED' ? 90 :
    status === 'AI_GENERATED' ? 95 : 100
  );

  let colorTheme = {
    border: 'border-emerald-500/30',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    glow: 'glow-emerald',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    Icon: CheckCircle2,
    label: 'GENUINE DOCUMENT'
  };

  if (status === 'AI_GENERATED') {
    colorTheme = {
      border: 'border-purple-500/30',
      bg: 'bg-purple-500/10',
      text: 'text-purple-400',
      glow: 'glow-purple',
      badge: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
      Icon: Cpu,
      label: 'AI-GENERATED SYNTHETIC'
    };
  } else if (status === 'ALTERED') {
    colorTheme = {
      border: 'border-rose-500/30',
      bg: 'bg-rose-500/10',
      text: 'text-rose-400',
      glow: 'glow-rose',
      badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
      Icon: ShieldAlert,
      label: 'DIGITALLY ALTERED'
    };
  } else if (status === 'AI_GENERATED_AND_ALTERED') {
    colorTheme = {
      border: 'border-rose-600/40',
      bg: 'bg-rose-600/15',
      text: 'text-rose-400',
      glow: 'glow-rose',
      badge: 'bg-rose-600/30 text-rose-200 border-rose-600/50',
      Icon: ShieldAlert,
      label: 'AI-GENERATED + ALTERED'
    };
  } else if (status === 'SUSPICIOUS') {
    colorTheme = {
      border: 'border-amber-500/30',
      bg: 'bg-amber-500/10',
      text: 'text-amber-400',
      glow: 'glow-amber',
      badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
      Icon: FileX,
      label: 'SUSPICIOUS STRUCTURE'
    };
  } else if (status === 'INCOMPLETE_SUBMISSION') {
    colorTheme = {
      border: 'border-blue-500/30',
      bg: 'bg-blue-500/10',
      text: 'text-blue-400',
      glow: 'glow-blue',
      badge: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
      Icon: AlertTriangle,
      label: 'INCOMPLETE SUBMISSION'
    };
  } else if (status === 'MANUAL_REVIEW') {
    colorTheme = {
      border: 'border-amber-500/30',
      bg: 'bg-amber-500/10',
      text: 'text-amber-400',
      glow: 'glow-amber',
      badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
      Icon: Eye,
      label: 'MANUAL OFFICER REVIEW'
    };
  }

  const { Icon } = colorTheme;
  const strokeDashoffset = displayRiskScore !== null 
    ? 283 - (283 * displayRiskScore) / 100 
    : 141;

  return (
    <div className={`glass-panel p-6 rounded-2xl border ${colorTheme.border} ${colorTheme.glow} flex flex-col items-center justify-center text-center relative overflow-hidden`}>
      <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full ${colorTheme.bg} blur-2xl pointer-events-none`}></div>

      {/* SVG Circular Progress Gauge representing Risk Score */}
      <div className="relative w-36 h-36 flex items-center justify-center my-2">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45" className="stroke-slate-800" strokeWidth="8" fill="transparent" />
          <circle
            cx="50"
            cy="50"
            r="45"
            className={`transition-all duration-1000 ease-out ${colorTheme.text}`}
            strokeWidth="8"
            strokeDasharray="283"
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center px-1">
          <span className="text-3xl font-black text-white tracking-tight">{displayRiskScore}</span>
          <span className="text-[10px] text-slate-400 font-mono tracking-wider uppercase">RISK SCORE (/100)</span>
          {displayConfidence !== null && (
            <span className="text-[10px] text-cyan-400 font-mono font-bold mt-0.5">
              {displayConfidence}% Confidence
            </span>
          )}
        </div>
      </div>

      {/* Status Badge & Confidence Pill */}
      <div className="flex flex-col items-center gap-1.5 mt-2">
        <div className={`px-4 py-1.5 rounded-full border text-xs font-bold flex items-center space-x-2 tracking-wide uppercase ${colorTheme.badge}`}>
          <Icon className="w-4 h-4" />
          <span>{colorTheme.label}</span>
        </div>
        {displayConfidence !== null && (
          <span className="text-[11px] font-mono text-slate-400">
            Decision Confidence: <strong className="text-cyan-300">{displayConfidence}%</strong>
          </span>
        )}
      </div>
    </div>
  );
}
