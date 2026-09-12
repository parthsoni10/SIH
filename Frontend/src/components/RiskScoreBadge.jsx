import React from 'react';
import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function RiskScoreBadge({ riskScore, prediction }) {
  const isHighRisk = riskScore >= 50 || prediction === 'fraudulent';
  const isModerateRisk = riskScore >= 30 && riskScore < 50;
  
  let colorTheme = {
    border: 'border-emerald-500/30',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    glow: 'glow-emerald',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    Icon: CheckCircle2,
    label: 'GENUINE DOCUMENT'
  };

  if (isHighRisk) {
    colorTheme = {
      border: 'border-rose-500/30',
      bg: 'bg-rose-500/10',
      text: 'text-rose-400',
      glow: 'glow-rose',
      badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
      Icon: ShieldAlert,
      label: 'FRAUDULENT / HIGH RISK'
    };
  } else if (isModerateRisk) {
    colorTheme = {
      border: 'border-amber-500/30',
      bg: 'bg-amber-500/10',
      text: 'text-amber-400',
      glow: 'glow-amber',
      badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
      Icon: AlertTriangle,
      label: 'MODERATE RISK / MANUAL REVIEW'
    };
  }

  const { Icon } = colorTheme;
  const strokeDashoffset = 283 - (283 * riskScore) / 100;

  return (
    <div className={`glass-panel p-6 rounded-2xl border ${colorTheme.border} ${colorTheme.glow} flex flex-col items-center justify-center text-center relative overflow-hidden`}>
      {/* Background glow effect */}
      <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full ${colorTheme.bg} blur-2xl pointer-events-none`}></div>

      {/* SVG Circular Progress Gauge */}
      <div className="relative w-36 h-36 flex items-center justify-center my-2">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            className="stroke-slate-800"
            strokeWidth="8"
            fill="transparent"
          />
          {/* Progress circle */}
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

        {/* Center Score Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-black text-white tracking-tight">{riskScore}</span>
          <span className="text-[10px] text-slate-400 font-mono tracking-wider">RISK SCORE</span>
        </div>
      </div>

      {/* Status Badge */}
      <div className={`mt-3 px-4 py-1.5 rounded-full border text-xs font-bold flex items-center space-x-2 tracking-wide uppercase ${colorTheme.badge}`}>
        <Icon className="w-4 h-4" />
        <span>{colorTheme.label}</span>
      </div>
    </div>
  );
}
