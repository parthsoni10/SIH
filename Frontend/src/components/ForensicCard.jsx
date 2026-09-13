import React from 'react';
import { Activity, ShieldCheck, AlertCircle, UserCheck, FileCheck, Layout, CheckCircle2 } from 'lucide-react';

const RULE_EXPLANATIONS = {
  missing_name: 'Holder Name Missing or Unreadable in OCR',
  missing_document_number: 'Document / ID Number Missing',
  invalid_document_number_format: 'ID Number Format Does Not Match Standard Pattern',
  invalid_id_checksum: 'Document Checksum Verification Failed',
  missing_or_invalid_dob: 'Date of Birth Field Missing or Unparseable',
  future_dob: 'Date of Birth is Set in the Future',
  implausible_dob_age: 'Date of Birth Indicates Unlikely Age (>120 Years)',
  missing_expiry_date: 'Document Expiry Date Missing',
  expired_document: 'Document Has Expired',
  invalid_government_layout_template: 'Document Layout Violates Official Template',
  blacklist_hit: 'Document ID Found on Watch-list / Blacklist',
};

export default function ForensicCard({
  tamperingScore,
  tamperingSignals = {},
  idChecksumValid,
  expiryValid,
  validationPassRate,
  faceMatchScore,
  blacklistHit,
  failedRules = [],
  layoutScore = 1.0,
  layoutAnomalies = []
}) {
  const elaScore = Math.round((tamperingSignals.ela_score || 0) * 100);
  const metaScore = Math.round((tamperingSignals.metadata_score || 0) * 100);
  const noiseScore = Math.round((tamperingSignals.noise_inconsistency || 0) * 100);
  const layoutPct = Math.round((layoutScore ?? 1.0) * 100);
  const layoutValid = layoutPct >= 70;

  const renderFaceMatchBadge = () => {
    if (faceMatchScore === null || faceMatchScore === undefined) {
      return (
        <span className="px-2 py-0.5 rounded font-bold bg-slate-800 text-slate-400 border border-slate-700">
          N/A (No Live Photo Provided)
        </span>
      );
    }
    const scorePct = Math.round(faceMatchScore * 100);
    if (scorePct >= 60) {
      return (
        <span className="px-2 py-0.5 rounded font-bold bg-emerald-500/20 text-emerald-300">
          {scorePct}% Match
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded font-bold bg-rose-500/20 text-rose-300">
        {scorePct}% Low Match
      </span>
    );
  };

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

        {/* Signal 4: Live Face Match */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <UserCheck className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-300 font-sans font-medium">Live Face Similarity</span>
          </div>
          {renderFaceMatchBadge()}
        </div>

        {/* Signal 5: Govt Layout & Geometry */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between md:col-span-2">
          <div className="flex items-center space-x-2.5">
            <Layout className={`w-4 h-4 ${layoutValid ? 'text-emerald-400' : 'text-amber-400'}`} />
            <span className="text-slate-300 font-sans font-medium">Government Template Layout & Geometry</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-slate-400">{layoutPct}% Score</span>
            <span className={`px-2 py-0.5 rounded font-bold ${layoutValid ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
              {layoutValid ? 'AUTHENTIC LAYOUT' : 'LAYOUT ANOMALY'}
            </span>
          </div>
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

          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Government Approved Geometry & Emblem Alignment</span>
              <span>{layoutPct}%</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div className={layoutValid ? "bg-emerald-400 h-full" : "bg-amber-400 h-full"} style={{ width: `${layoutPct}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Layout Anomalies List */}
      {layoutAnomalies.length > 0 && (
        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs">
          <p className="font-semibold text-amber-300 mb-1.5 flex items-center space-x-1">
            <AlertCircle className="w-3.5 h-3.5" />
            <span>Layout & Spatial Anomalies Detected ({layoutAnomalies.length})</span>
          </p>
          <ul className="list-disc list-inside space-y-0.5 text-amber-200/80 font-mono">
            {layoutAnomalies.map((anomaly, idx) => (
              <li key={idx}>{anomaly}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Failed Rules List */}
      {failedRules.length > 0 && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs">
          <p className="font-semibold text-rose-300 mb-1.5 flex items-center space-x-1">
            <AlertCircle className="w-3.5 h-3.5" />
            <span>Flagged Rule Anomalies ({failedRules.length})</span>
          </p>
          <ul className="list-disc list-inside space-y-0.5 text-rose-200/80 font-mono">
            {failedRules.map((rule, idx) => (
              <li key={idx}>{RULE_EXPLANATIONS[rule] || rule.replace(/_/g, ' ')}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
