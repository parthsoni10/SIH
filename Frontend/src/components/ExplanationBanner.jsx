import React from 'react';
import { Sparkles, Bot, ShieldCheck } from 'lucide-react';

export default function ExplanationBanner({ explanation, riskScore, gemini }) {
  const isHighRisk = riskScore >= 50;
  const isLiveGemini = gemini?.available === true;
  const displayText = explanation || gemini?.summary || 'Evaluating document features and generating risk assessment explanation...';

  return (
    <div className={`p-5 rounded-2xl border transition-all ${
      isHighRisk
        ? 'bg-gradient-to-r from-rose-950/70 via-slate-900 to-slate-950 border-rose-500/30 glow-rose'
        : 'bg-gradient-to-r from-emerald-950/70 via-slate-900 to-slate-950 border-emerald-500/30 glow-emerald'
    }`}>
      <div className="flex items-start space-x-3.5">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border ${
          isHighRisk
            ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
            : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
        }`}>
          {isLiveGemini ? (
            <Sparkles className="w-5 h-5 animate-pulse text-purple-300" />
          ) : (
            <ShieldCheck className="w-5 h-5 text-cyan-300" />
          )}
        </div>

        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono font-bold tracking-wider text-cyan-400 uppercase flex items-center gap-1">
              <Bot className="w-3.5 h-3.5" />
              <span>{isLiveGemini ? 'Gemini AI Officer Summary' : 'AI Forensic Summary (Offline Rule Engine)'}</span>
            </span>
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${
              isLiveGemini
                ? 'bg-purple-500/10 text-purple-300 border-purple-500/20'
                : 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
            }`}>
              {isLiveGemini ? 'Gemini Live' : 'Offline Mode'}
            </span>
          </div>

          <p className="text-sm font-medium text-slate-100 leading-relaxed">
            "{displayText}"
          </p>
        </div>
      </div>
    </div>
  );
}
