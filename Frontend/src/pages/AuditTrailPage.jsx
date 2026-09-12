import React, { useState, useEffect } from 'react';
import { History, Search, RefreshCw, Eye, ShieldAlert, CheckCircle2, Filter } from 'lucide-react';
import { fetchAuditTrail } from '../services/api';
import ResultPage from './ResultPage';

export default function AuditTrailPage() {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [filterDocType, setFilterDocType] = useState('');
  const [filterPrediction, setFilterPrediction] = useState('');
  
  const [selectedRecord, setSelectedRecord] = useState(null);

  const loadTrail = async () => {
    setLoading(true);
    try {
      const data = await fetchAuditTrail(page, 10, filterDocType || null, filterPrediction || null);
      setLogs(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Audit trail load failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTrail();
  }, [page, filterDocType, filterPrediction]);

  if (selectedRecord) {
    return (
      <ResultPage
        resultData={selectedRecord}
        onBackToUpload={() => setSelectedRecord(null)}
      />
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6 py-6 px-4">
      {/* Title Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-2xl border border-slate-800">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <History className="w-6 h-6 text-cyan-400" />
            <span>Digital Audit Trail Log</span>
          </h2>
          <p className="text-xs text-slate-400">
            Immutable database records of all historical border identity document verifications.
          </p>
        </div>

        <button
          onClick={loadTrail}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 text-xs font-semibold transition-all self-start md:self-auto"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Log</span>
        </button>
      </div>

      {/* Filters Bar */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
          <Filter className="w-4 h-4 text-cyan-400" />
          <span>Filters:</span>
        </div>

        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          {/* Filter Doc Type */}
          <select
            value={filterDocType}
            onChange={(e) => { setFilterDocType(e.target.value); setPage(1); }}
            className="bg-slate-900 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500"
          >
            <option value="">All Document Types</option>
            <option value="Passport">Passport</option>
            <option value="Visa">Visa</option>
            <option value="Aadhaar">Aadhaar</option>
            <option value="PAN Card">PAN Card</option>
            <option value="Driving License">Driving License</option>
          </select>

          {/* Filter Prediction */}
          <select
            value={filterPrediction}
            onChange={(e) => { setFilterPrediction(e.target.value); setPage(1); }}
            className="bg-slate-900 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500"
          >
            <option value="">All Predictions</option>
            <option value="genuine">Genuine Only</option>
            <option value="fraudulent">Fraudulent Only</option>
          </select>
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900/90 text-slate-400 uppercase font-mono border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Log ID</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Document Type</th>
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4 text-center">Risk Score</th>
                <th className="py-3 px-4">Prediction</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-800/60 font-mono">
              {logs.length === 0 ? (
                <tr>
                  <td colSpan="7" className="py-8 text-center text-slate-500 italic">
                    {loading ? 'Loading audit records...' : 'No audit records found matching selected criteria.'}
                  </td>
                </tr>
              ) : (
                logs.map((row) => {
                  const isFraud = row.prediction === 'fraudulent' || row.risk_score >= 50;

                  return (
                    <tr key={row.id} className="hover:bg-slate-800/40 transition-all">
                      <td className="py-3.5 px-4 font-bold text-cyan-400">#{row.id}</td>
                      <td className="py-3.5 px-4 text-slate-400">
                        {new Date(row.created_at).toLocaleString()}
                      </td>
                      <td className="py-3.5 px-4 font-semibold text-slate-200">
                        {row.document_type}
                      </td>
                      <td className="py-3.5 px-4 text-slate-400 max-w-[150px] truncate">
                        {row.filename || 'N/A'}
                      </td>
                      <td className="py-3.5 px-4 text-center font-bold">
                        <span className={`px-2.5 py-1 rounded-full ${isFraud ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}`}>
                          {row.risk_score} / 100
                        </span>
                      </td>
                      <td className="py-3.5 px-4 uppercase font-bold text-xs">
                        <span className={isFraud ? 'text-rose-400' : 'text-emerald-400'}>
                          {row.prediction}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => setSelectedRecord(row)}
                          className="px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-xs font-semibold transition-all inline-flex items-center space-x-1"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>View Detail</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 bg-slate-900/60 border-t border-slate-800 flex items-center justify-between text-xs font-mono text-slate-400">
          <span>Total Records: {total}</span>
          <div className="flex space-x-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 rounded bg-slate-800 disabled:opacity-50 text-slate-300"
            >
              Previous
            </button>
            <span className="py-1 px-2">Page {page}</span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={logs.length < 10}
              className="px-3 py-1 rounded bg-slate-800 disabled:opacity-50 text-slate-300"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
