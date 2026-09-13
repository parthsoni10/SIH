import React, { useState } from 'react';
import { FileText, Sparkles, AlertTriangle, CheckCircle, HelpCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';

export default function OcrExtractedView({ rawText, llmValidation, warnings }) {
  const [showRawText, setShowRawText] = useState(false);

  const hasRawText = rawText && rawText.length > 0;
  const llmAvailable = llmValidation?.llm_available;
  const extractedFields = llmValidation?.extracted_fields || {};
  const anomalies = llmValidation?.mismatches_or_anomalies || [];
  const schemaMatched = llmValidation?.schema_matched;

  const getStatusBadge = (status) => {
    switch (status) {
      case 'valid':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px]">
            <CheckCircle className="w-3 h-3" />
            <span>Valid</span>
          </span>
        );
      case 'suspicious':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[10px]">
            <AlertTriangle className="w-3 h-3" />
            <span>Suspicious</span>
          </span>
        );
      case 'unclear':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[10px]">
            <HelpCircle className="w-3 h-3" />
            <span>Unclear</span>
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-slate-500/10 text-slate-400 border border-slate-500/20 text-[10px]">
            <XCircle className="w-3 h-3" />
            <span>Missing</span>
          </span>
        );
    }
  };

  return (
    <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2 text-white font-semibold text-sm">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <span>OCR & LLM Field Extraction Analysis</span>
        </div>

        <div className="flex items-center space-x-2">
          {llmAvailable ? (
            <span className="px-2.5 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 text-[11px] font-mono flex items-center space-x-1">
              <Sparkles className="w-3 h-3 text-purple-400" />
              <span>Gemini AI Validated</span>
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 text-[11px] font-mono">
              Regex Fallback Active
            </span>
          )}

          {schemaMatched !== undefined && (
            <span className={`px-2.5 py-1 rounded-full text-[11px] font-mono border ${
              schemaMatched 
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
            }`}>
              {schemaMatched ? 'Schema Matched' : 'Schema Incomplete'}
            </span>
          )}
        </div>
      </div>

      {/* Preprocessing Warnings (if any) */}
      {warnings && warnings.length > 0 && (
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 space-y-1">
          <div className="text-amber-400 text-xs font-semibold flex items-center space-x-1.5">
            <AlertTriangle className="w-4 h-4" />
            <span>Image Preprocessing Quality Warnings</span>
          </div>
          <ul className="list-disc list-inside text-amber-200/80 text-xs font-mono space-y-0.5 pl-1">
            {warnings.map((warn, idx) => (
              <li key={idx}>{warn}</li>
            ))}
          </ul>
        </div>
      )}

      {/* LLM Mismatches / Anomalies (if any) */}
      {anomalies.length > 0 && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 space-y-1.5">
          <div className="text-rose-400 text-xs font-semibold flex items-center space-x-1.5">
            <AlertTriangle className="w-4 h-4" />
            <span>LLM Field Validation Anomalies & Mismatches</span>
          </div>
          <ul className="list-disc list-inside text-rose-200/90 text-xs font-mono space-y-1 pl-1">
            {anomalies.map((anom, idx) => (
              <li key={idx}>{anom}</li>
            ))}
          </ul>
        </div>
      )}

      {/* LLM Extracted Fields Table */}
      {Object.keys(extractedFields).length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-mono text-slate-400 font-semibold uppercase tracking-wider">
            LLM Validated Field Details
          </div>
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900/80 text-slate-400 uppercase text-[10px]">
                <tr>
                  <th className="px-3 py-2">Field</th>
                  <th className="px-3 py-2">Extracted Value</th>
                  <th className="px-3 py-2">Confidence</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                {Object.entries(extractedFields).map(([key, item]) => {
                  const val = typeof item === 'object' ? item.value : item;
                  const conf = typeof item === 'object' ? item.confidence : 0.85;
                  const status = typeof item === 'object' ? item.status : 'valid';

                  return (
                    <tr key={key} className="hover:bg-slate-900/40 transition-colors">
                      <td className="px-3 py-2 text-slate-300 font-semibold capitalize">
                        {key.replace(/_/g, ' ')}
                      </td>
                      <td className="px-3 py-2 text-cyan-300 font-mono">
                        {val || <span className="text-slate-600 font-normal">N/A</span>}
                      </td>
                      <td className="px-3 py-2 text-slate-400">
                        {conf != null ? `${Math.round(conf * 100)}%` : '-'}
                      </td>
                      <td className="px-3 py-2">
                        {getStatusBadge(status)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Toggle Raw PaddleOCR Text Drawer */}
      {hasRawText && (
        <div className="pt-2 border-t border-slate-800/60">
          <button
            onClick={() => setShowRawText(!showRawText)}
            className="w-full flex items-center justify-between px-4 py-2.5 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300 transition-all"
          >
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-cyan-400" />
              <span>Raw PaddleOCR Extracted Text ({rawText.length} lines)</span>
            </div>
            {showRawText ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
          </button>

          {showRawText && (
            <div className="mt-3 p-4 rounded-xl bg-slate-950 border border-slate-800 max-h-60 overflow-y-auto font-mono text-xs text-slate-300 space-y-1">
              {rawText.map((line, idx) => (
                <div key={idx} className="flex items-start space-x-3 hover:bg-slate-900/40 px-2 py-0.5 rounded">
                  <span className="text-slate-600 text-[10px] select-none w-6 text-right pt-0.5">
                    {idx + 1}
                  </span>
                  <span className="text-slate-200">{line}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
