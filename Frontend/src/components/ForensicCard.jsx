import React from 'react';
import { Activity, ShieldCheck, AlertCircle, Eye, UserCheck, FileCheck } from 'lucide-react';

export default function ForensicCard({
  tamperingScore,
  tamperingSignals = {},
  idChecksumValid,
  expiryValid,
  validationPassRate,
  faceMatchScore,
  blacklistHit,
  failedRules = []
}) {
  const elaScore = Math.round((tamperingSignals.ela_score || 0) * 100);
  const metaScore = Math.round((tamperingSignals.metadata_score || 0) * 100);
  const noiseScore = Math.round((tamperingSignals.noise_inconsistency || 0) * 100);
  const passRatePct = Math.round(validationPassRate * 100);

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2 text-sm font-semibold text-slate-200">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span>Forensic & Security Indicators</span>
        </div>
        <span className="text-xs font-mono text-slate-400">Deep Inspection</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
        {/* Signal 1: ID Checksum */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <FileCheck className={`w-4 h-4 ${idChecksumValid ? 'text-emerald-400' : 'text-rose-400'}`} />
            <span className="text-slate-300 font-sans font-medium">ID / MRZ Checksum</span>
          </div>
          <span className={`px-2 py-0.5 rounded font-bold ${idChecksumValid ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
            {idChecksumValid ? 'PASSED' : 'FAILED'}
          </span>
        </div>

        {/* Signal 2: Expiry Date */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <ShieldCheck className={`w-4 h-4 ${expiryValid ? 'text-emerald-400' : 'text-rose-400'}`} />
            <span className="text-slate-300 font-sans font-medium">Document Validity</span>
          </div>
          <span className={`px-2 py-0.5 rounded font-bold ${expiryValid ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
            {expiryValid ? 'VALID' : 'EXPIRED'}
          </span>
        </div>

        {/* Signal 3: Blacklist Check */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <AlertCircle className={`w-4 h-4 ${!blacklistHit ? 'text-emerald-400' : 'text-rose-400'}`} />
            <span className="text-slate-300 font-sans font-medium">Watch-list / Blacklist</span>
          </div>
          <span className={`px-2 py-0.5 rounded font-bold ${!blacklistHit ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300 animate-pulse'}`}>
            {blacklistHit ? 'MATCH DETECTED' : 'CLEAR'}
          </span>
        </div>

        {/* Signal 4: Face Match */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <UserCheck className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-300 font-sans font-medium">Live Face Similarity</span>
          </div>
          <span className="font-bold text-cyan-300">
            {faceMatchScore != null ? `${Math.round(faceMatchScore * 100)}% Match` : 'N/A (No Live Capture)'}
          </span>
        </div>
      </div>

      {/* Forensic Tampering Scores Progress */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3 font-mono text-xs">
        <div className="flex justify-between items-center text-slate-300 font-sans font-semibold">
          <span>Image Tampering Analysis (ELA & EXIF)</span>
          <span className={tamperingScore >= 0.35 ? 'text-rose-400' : 'text-emerald-400'}>
            Overall Score: {(tamperingScore * 100).toFixed(1)}%
          </span>
        </div>

        <div className="space-y-2">
          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Error Level Analysis (ELA diff map)</span>
              <span>{elaScore}%</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div className="bg-cyan-400 h-full" style={{ width: `${elaScore}%` }} />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>EXIF Software & Metadata Tag Inspection</span>
              <span>{metaScore}%</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div className="bg-amber-400 h-full" style={{ width: `${metaScore}%` }} />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Block Noise Inconsistency Variance</span>
              <span>{noiseScore}%</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div className="bg-emerald-400 h-full" style={{ width: `${noiseScore}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Failed Rules List */}
      {failedRules.length > 0 && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs">
          <p className="font-semibold text-rose-300 mb-1.5 flex items-center space-x-1">
            <AlertCircle className="w-3.5 h-3.5" />
            <span>Flagged Rule Anomalies ({failedRules.length})</span>
          </p>
          <ul className="list-disc list-inside space-y-0.5 text-rose-200/80 font-mono">
            {failedRules.map((rule, idx) => (
              <li key={idx}>{rule.replace('_', ' ')}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
