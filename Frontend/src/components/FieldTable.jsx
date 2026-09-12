import React from 'react';
import { Database, CheckCircle2, AlertCircle } from 'lucide-react';

export default function FieldTable({ fields = {} }) {
  const entries = Object.entries(fields);

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2 text-sm font-semibold text-slate-200">
          <Database className="w-4 h-4 text-cyan-400" />
          <span>Extracted OCR Field Attributes</span>
        </div>
        <span className="text-xs font-mono text-slate-400">{entries.length} Fields Extracted</span>
      </div>

      {entries.length === 0 ? (
        <div className="p-6 text-center text-slate-500 text-xs font-mono">
          No structured text fields detected by OCR parser.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900/90 text-slate-400 uppercase font-mono border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Field Name</th>
                <th className="py-2.5 px-3">Extracted Value</th>
                <th className="py-2.5 px-3 text-right">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {entries.map(([key, valObj]) => {
                const value = typeof valObj === 'object' && valObj !== null ? valObj.value : String(valObj);
                const conf = typeof valObj === 'object' && valObj !== null && valObj.confidence != null
                  ? Math.round(valObj.confidence * 100)
                  : 90;

                return (
                  <tr key={key} className="hover:bg-slate-800/40 transition-all">
                    <td className="py-3 px-3 capitalize font-semibold text-slate-200">
                      {key.replace('_', ' ')}
                    </td>
                    <td className="py-3 px-3 text-cyan-300 font-semibold">
                      {value || <span className="text-slate-500 italic">Not detected</span>}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <div className="flex items-center justify-end space-x-2">
                        <div className="w-20 bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all ${
                              conf >= 80 ? 'bg-emerald-400' : conf >= 50 ? 'bg-amber-400' : 'bg-rose-400'
                            }`}
                            style={{ width: `${conf}%` }}
                          />
                        </div>
                        <span className="text-xs font-semibold w-8">{conf}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
