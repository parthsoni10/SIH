import React, { useState, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';
import {
  ShieldCheck, Loader2, Cpu, ScanLine, LayoutGrid,
  Fingerprint, UserCheck, Sparkles, Sliders, Terminal,
  Zap, Check, Radio, FileText, Lock, Activity,
  ChevronRight, Eye
} from 'lucide-react';

/* ─── Pipeline stage definitions ───────────────────────────────────── */
const STAGES = [
  { id: 1, label: 'Preprocessing',   short: 'PREP',    icon: Sliders,     badge: 'M-01', logMsg: '› Image canvas normalized · EXIF metadata extracted · Color matrix calibrated' },
  { id: 2, label: 'OCR Extraction',  short: 'OCR',     icon: ScanLine,    badge: 'M-02', logMsg: '› PaddleOCR parsed bounding zones · MRZ field extraction complete' },
  { id: 3, label: 'Layout Check',    short: 'LAYOUT',  icon: LayoutGrid,  badge: 'M-03', logMsg: '› Document geometry validated against ICAO-9303 standard templates' },
  { id: 4, label: 'Checksum Audit',  short: 'RULES',   icon: ShieldCheck, badge: 'M-04', logMsg: '› Verhoeff checksum validated · Interpol watchlist cross-reference complete' },
  { id: 5, label: 'Forensic ELA',    short: 'ELA',     icon: Fingerprint, badge: 'M-05', logMsg: '› Error Level Analysis complete · Pixel manipulation anomaly matrix generated' },
  { id: 6, label: 'AI Forensics',    short: 'AI-DET',  icon: Activity,    badge: 'M-06', logMsg: '› ConvNeXt-Base + 2D FFT spectral analysis complete · AI probability calculated' },
  { id: 7, label: 'Face Match',      short: 'FACE',    icon: UserCheck,   badge: 'M-07', logMsg: '› DeepFace embedding extracted · Cosine similarity computed against live frame' },
  { id: 8, label: 'ML Risk Score',   short: 'ML',      icon: Cpu,         badge: 'M-08', logMsg: '› Random Forest V2 evaluated security features · Fraud probability vector ready' },
  { id: 9, label: 'Decision Engine', short: 'DECIDE',  icon: Sparkles,    badge: 'M-09', logMsg: '› Calibrated Decision Engine finalized verdict & synthesized officer report' },
];

/* ─── Circular Progress Ring (SVG) ─────────────────────────────────── */
function ProgressRing({ percent, size = 120, stroke = 5, isDone }) {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (percent / 100) * circ;
  const color = isDone ? '#34d399' : '#22d3ee';
  const glowColor = isDone ? 'rgba(52,211,153,0.5)' : 'rgba(34,211,238,0.5)';

  return (
    <svg width={size} height={size} style={{ transform: 'rotate(-90deg)', filter: `drop-shadow(0 0 8px ${glowColor})` }}>
      {/* Background track */}
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(51,65,85,0.5)" strokeWidth={stroke} />
      {/* Progress arc */}
      <circle
        cx={size/2} cy={size/2} r={r} fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.7s ease-out' }}
      />
    </svg>
  );
}

/* ─── Main Component ───────────────────────────────────────────────── */
export default function PipelineProgressModal({ isOpen, file, documentType = 'Passport', isDone = false }) {
  const [stageIdx, setStageIdx] = useState(0);
  const [pct, setPct] = useState(3);
  const [logs, setLogs] = useState([]);
  const [previewUrl, setPreviewUrl] = useState(null);
  const logEndRef = useRef(null);

  // Body scroll lock
  useEffect(() => {
    document.body.style.overflow = isOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  // File preview
  useEffect(() => {
    if (file?.type?.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setPreviewUrl(null);
  }, [file]);

  // Stage progression
  useEffect(() => {
    if (!isOpen) { setStageIdx(0); setPct(3); setLogs([]); return; }
    const t0 = Date.now();
    const ts = () => `${((Date.now() - t0) / 1000).toFixed(2)}s`;
    setLogs([`[${ts()}] [SYS] Pipeline initialized — ${documentType.toUpperCase()} document accepted`]);

    const timer = setInterval(() => {
      setStageIdx(prev => {
        if (prev < STAGES.length - 1) {
          const next = prev + 1;
          const s = STAGES[next];
          setLogs(l => [...l, `[${ts()}] [${s.badge}] ${s.logMsg}`]);
          setPct(Math.min(92, Math.round(((next + 1) / STAGES.length) * 92)));
          return next;
        }
        return prev;
      });
    }, 1100);
    return () => clearInterval(timer);
  }, [isOpen, documentType]);

  // Done
  useEffect(() => {
    if (isDone) {
      setStageIdx(STAGES.length - 1);
      setPct(100);
      setLogs(l => [...l, `[DONE] ✓ All 9 modules complete — compiling final audit payload`]);
    }
  }, [isDone]);

  // Auto-scroll logs
  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  if (!isOpen) return null;

  const active = STAGES[stageIdx];
  const ActiveIcon = active.icon;

  return ReactDOM.createPortal(
    <div className="pipeline-overlay pipeline-dots flex flex-col items-center justify-center" style={{ padding: '24px 32px' }}>

      {/* ═══ CENTERED CONTENT CARD ════════════════════════════════════ */}
      <div
        className="flex flex-col w-full h-full min-h-0 max-w-7xl mx-auto relative overflow-hidden"
        style={{ borderRadius: 20, border: '1px solid rgba(99,179,237,0.12)', background: 'rgba(2,6,23,0.92)', boxShadow: '0 0 60px rgba(6,182,212,0.08), 0 4px 30px rgba(0,0,0,0.5)' }}
      >
        {/* Gradient crown bar — sits at top of the card */}
        <div style={{ height: 3, flexShrink: 0, background: 'linear-gradient(90deg, transparent, #0ea5e9, #8b5cf6, #ec4899, #8b5cf6, #0ea5e9, transparent)' }} />

        {/* Inner padding wrapper */}
        <div className="flex flex-col flex-1 min-h-0" style={{ padding: '14px 20px', gap: '10px' }}>

        {/* ── ROW 1: HEADER ────────────────────────────────────────── */}
        <div
          className="flex items-center justify-between shrink-0"
          style={{ padding: '10px 20px', borderRadius: 14, background: 'rgba(15,23,42,0.7)', backdropFilter: 'blur(12px)', border: '1px solid rgba(99,179,237,0.12)' }}
        >
          {/* Left */}
          <div className="flex items-center gap-3.5">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center animate-pulse-glow"
              style={{ background: 'linear-gradient(135deg, rgba(14,165,233,0.2), rgba(139,92,246,0.15))', border: '1px solid rgba(14,165,233,0.35)' }}
            >
              <ShieldCheck className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-sm font-black text-white uppercase tracking-[0.12em]">
                  Sentinel Screening Pipeline
                </h1>
                <span className="text-[9px] font-mono font-bold uppercase px-2 py-0.5 rounded-full" style={{ background: 'rgba(14,165,233,0.12)', border: '1px solid rgba(14,165,233,0.3)', color: '#7dd3fc' }}>
                  {documentType}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono flex items-center gap-1.5 mt-0.5">
                <Radio className="w-2.5 h-2.5 text-cyan-400 animate-pulse" />
                Multi-agent forensic verification · 9 modules · real-time
              </p>
            </div>
          </div>

          {/* Center: live indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: isDone ? 'rgba(52,211,153,0.08)' : 'rgba(251,191,36,0.06)', border: isDone ? '1px solid rgba(52,211,153,0.25)' : '1px solid rgba(251,191,36,0.2)' }}>
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: isDone ? '#34d399' : '#fbbf24' }} />
            <span className="text-[10px] font-mono font-bold tracking-widest" style={{ color: isDone ? '#34d399' : '#fbbf24' }}>
              {isDone ? 'COMPLETE' : 'LIVE'}
            </span>
          </div>

          {/* Right: percentage ring */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="relative">
              <ProgressRing percent={pct} size={48} stroke={3} isDone={isDone} />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-black font-mono" style={{ color: isDone ? '#34d399' : '#22d3ee' }}>
                  {pct}%
                </span>
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase">Module</div>
              <div className="text-lg font-black font-mono leading-none" style={{ color: '#7dd3fc' }}>{stageIdx + 1}<span className="text-slate-600 text-xs">/{STAGES.length}</span></div>
            </div>
          </div>
        </div>

        {/* ── ROW 2: HORIZONTAL STEPPER ────────────────────────────── */}
        <div
          className="shrink-0 flex items-center gap-0 justify-between relative"
          style={{ padding: '8px 12px' }}
        >
          {STAGES.map((s, idx) => {
            const done = idx < stageIdx || isDone;
            const isActive = idx === stageIdx && !isDone;
            const pending = !done && !isActive;
            const Icon = s.icon;
            const isLast = idx === STAGES.length - 1;

            return (
              <React.Fragment key={s.id}>
                {/* Node */}
                <div className="flex flex-col items-center gap-1.5 relative" style={{ zIndex: 2 }}>
                  {/* Circle */}
                  <div className="relative">
                    {/* Outer pulse ring for active */}
                    {isActive && (
                      <div
                        className="absolute animate-ping rounded-full"
                        style={{ inset: -4, border: '1.5px solid rgba(34,211,238,0.35)' }}
                      />
                    )}
                    <div
                      className="flex items-center justify-center transition-all duration-500"
                      style={{
                        width: isActive ? 40 : 34,
                        height: isActive ? 40 : 34,
                        borderRadius: '50%',
                        border: `2px solid ${isActive ? '#22d3ee' : done ? '#34d399' : 'rgba(51,65,85,0.6)'}`,
                        background: isActive
                          ? 'radial-gradient(circle, rgba(6,182,212,0.25), rgba(6,182,212,0.05))'
                          : done
                          ? 'radial-gradient(circle, rgba(52,211,153,0.2), rgba(52,211,153,0.03))'
                          : 'rgba(15,23,42,0.9)',
                        boxShadow: isActive
                          ? '0 0 20px rgba(6,182,212,0.4), inset 0 0 12px rgba(6,182,212,0.15)'
                          : done
                          ? '0 0 10px rgba(52,211,153,0.2)'
                          : 'none',
                      }}
                    >
                      {done
                        ? <Check style={{ width: 14, height: 14, color: '#34d399', strokeWidth: 3 }} />
                        : isActive
                        ? <Loader2 style={{ width: 16, height: 16, color: '#22d3ee', animation: 'spin 1s linear infinite' }} />
                        : <Icon style={{ width: 13, height: 13, color: '#475569' }} />
                      }
                    </div>
                  </div>

                  {/* Label */}
                  <div className="flex flex-col items-center" style={{ width: 68 }}>
                    <span className="text-[8px] font-mono font-bold tracking-wider" style={{ color: isActive ? '#67e8f9' : done ? '#6ee7b7' : '#475569' }}>
                      {s.badge}
                    </span>
                    <span className="text-[10px] font-bold text-center leading-tight" style={{ color: isActive ? '#e0f2fe' : done ? '#cbd5e1' : '#475569' }}>
                      {s.short}
                    </span>
                  </div>
                </div>

                {/* Connector line */}
                {!isLast && (
                  <div className="flex-1 relative" style={{ height: 2, marginTop: -18, zIndex: 1 }}>
                    <div className="absolute inset-0 rounded-full" style={{ background: 'rgba(51,65,85,0.4)' }} />
                    {(done || isActive) && (
                      <div
                        className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                        style={{
                          width: done ? '100%' : '50%',
                          background: done
                            ? 'linear-gradient(90deg, #34d399, #34d399)'
                            : 'linear-gradient(90deg, #34d399, #22d3ee)',
                          boxShadow: `0 0 6px ${done ? 'rgba(52,211,153,0.5)' : 'rgba(34,211,238,0.4)'}`,
                        }}
                      />
                    )}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* ── ROW 3: ACTIVE MODULE INFO + PROGRESS BAR ────────────── */}
        <div className="shrink-0 flex items-center gap-3" style={{ padding: '0 4px' }}>
          {/* Active module card */}
          <div
            className="flex items-center gap-3 flex-1 animate-border-glow"
            style={{ padding: '10px 14px', borderRadius: 12, background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.2)' }}
          >
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: 'rgba(6,182,212,0.15)', border: '1px solid rgba(6,182,212,0.3)' }}
            >
              <ActiveIcon className="w-4 h-4 text-cyan-300 animate-pulse" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded" style={{ background: 'rgba(6,182,212,0.15)', border: '1px solid rgba(6,182,212,0.25)', color: '#67e8f9' }}>
                  {active.badge}
                </span>
                <span className="text-xs font-bold text-white truncate">{active.label}</span>
              </div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="flex-1">
            <div className="flex justify-between mb-1">
              <span className="text-[9px] font-mono font-bold uppercase tracking-wider text-slate-500">Progress</span>
              <span className="text-[9px] font-mono font-bold text-slate-400">{pct}%</span>
            </div>
            <div style={{ height: 8, borderRadius: 6, background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(51,65,85,0.5)', padding: 1.5 }}>
              <div
                className="h-full relative overflow-hidden transition-all duration-700"
                style={{
                  width: `${pct}%`,
                  borderRadius: 4,
                  background: isDone ? 'linear-gradient(90deg, #10b981, #34d399)' : 'linear-gradient(90deg, #0ea5e9, #6366f1)',
                  boxShadow: isDone ? '0 0 10px rgba(52,211,153,0.5)' : '0 0 10px rgba(6,182,212,0.5)',
                }}
              >
                <div className="absolute inset-0 animate-shimmer" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)' }} />
              </div>
            </div>
          </div>
        </div>

        {/* ── ROW 4: SCANNER + TERMINAL ────────────────────────────── */}
        <div className="flex-1 min-h-0 grid gap-3" style={{ gridTemplateColumns: '2fr 3fr' }}>

          {/* Document Scanner */}
          <div
            className="flex flex-col overflow-hidden"
            style={{ borderRadius: 14, border: '1px solid rgba(51,65,85,0.5)', background: 'rgba(2,6,23,0.9)' }}
          >
            {/* Scanner header */}
            <div className="flex items-center justify-between px-3.5 py-2 shrink-0" style={{ borderBottom: '1px solid rgba(51,65,85,0.4)' }}>
              <span className="flex items-center gap-1.5 text-[10px] font-mono font-bold uppercase tracking-wider text-cyan-400">
                <Eye className="w-3 h-3" />
                Document Scanner
              </span>
              <span className="text-[8px] font-mono font-bold px-1.5 py-0.5 rounded" style={{ color: '#67e8f9', background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.2)' }}>
                ICAO 9303
              </span>
            </div>

            {/* Canvas */}
            <div className="relative flex-1 overflow-hidden" style={{ background: '#020817' }}>
              {previewUrl ? (
                <>
                  <img src={previewUrl} alt="Scanned document" className="absolute inset-0 w-full h-full object-contain" style={{ opacity: 0.8 }} />
                  {/* Scan laser */}
                  {!isDone && (
                    <div className="absolute inset-x-0 animate-scanline" style={{ height: 2, background: '#22d3ee', boxShadow: '0 0 20px 4px rgba(34,211,238,0.4), 0 0 60px 10px rgba(34,211,238,0.15)', zIndex: 10 }} />
                  )}
                  {/* Blueprint grid */}
                  <div className="absolute inset-0 pointer-events-none" style={{ backgroundImage: 'linear-gradient(rgba(6,182,212,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.05) 1px, transparent 1px)', backgroundSize: '20px 20px' }} />
                  {/* Corner brackets — thicker, more visible */}
                  {[
                    { t: 8, l: 8, bT: true, bL: true },
                    { t: 8, r: 8, bT: true, bR: true },
                    { b: 8, l: 8, bB: true, bL: true },
                    { b: 8, r: 8, bB: true, bR: true },
                  ].map((c, i) => (
                    <div
                      key={i}
                      className="absolute"
                      style={{
                        width: 18, height: 18,
                        top: c.t, bottom: c.b, left: c.l, right: c.r,
                        borderTop: c.bT ? '2px solid rgba(34,211,238,0.7)' : 'none',
                        borderBottom: c.bB ? '2px solid rgba(34,211,238,0.7)' : 'none',
                        borderLeft: c.bL ? '2px solid rgba(34,211,238,0.7)' : 'none',
                        borderRight: c.bR ? '2px solid rgba(34,211,238,0.7)' : 'none',
                      }}
                    />
                  ))}
                  {/* Status badge */}
                  <div className="absolute bottom-2.5 left-2.5 flex items-center gap-1.5 px-2 py-1 rounded-md" style={{ background: 'rgba(2,6,23,0.85)', backdropFilter: 'blur(6px)', border: '1px solid rgba(6,182,212,0.2)' }}>
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                    <span className="text-[9px] font-mono font-bold text-cyan-300">
                      {isDone ? '✓ VERIFIED' : `${active.badge} SCANNING`}
                    </span>
                  </div>
                </>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
                  <ShieldCheck className="w-10 h-10 animate-pulse" style={{ color: 'rgba(6,182,212,0.2)' }} />
                  <span className="text-[11px] font-mono text-slate-600">Awaiting document frame</span>
                </div>
              )}
            </div>
          </div>

          {/* Terminal */}
          <div
            className="flex flex-col overflow-hidden"
            style={{ borderRadius: 14, border: '1px solid rgba(51,65,85,0.5)', background: 'rgba(2,6,23,0.95)' }}
          >
            {/* Terminal chrome */}
            <div className="flex items-center justify-between px-3.5 py-2 shrink-0" style={{ borderBottom: '1px solid rgba(30,41,59,0.6)', background: 'rgba(8,14,28,0.8)' }}>
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#ef4444' }} />
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#eab308' }} />
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#22c55e' }} />
                </div>
                <div className="w-px h-3.5 bg-slate-800 mx-0.5" />
                <Terminal className="w-3 h-3 text-slate-500" />
                <span className="text-[10px] font-mono text-slate-400">forensic-pipeline.log</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[9px] font-mono font-bold text-emerald-400 tracking-wider">STREAMING</span>
              </div>
            </div>

            {/* Logs */}
            <div
              className="flex-1 min-h-0 overflow-y-auto terminal-scroll p-3 space-y-1"
              style={{ fontFamily: "'Fira Code', 'Cascadia Code', 'Courier New', monospace", fontSize: 11, lineHeight: 1.65 }}
            >
              {logs.map((log, i) => {
                const isDoneLog = log.includes('[DONE]');
                const isModule = log.includes('[M-0');
                return (
                  <div
                    key={i}
                    className="animate-float-in"
                    style={{
                      animationDelay: `${i * 30}ms`,
                      color: isDoneLog ? '#34d399' : isModule ? '#7dd3fc' : '#64748b',
                      fontWeight: isDoneLog ? 700 : isModule ? 500 : 400,
                      paddingLeft: isModule ? 6 : 0,
                      borderLeft: isModule ? '2px solid rgba(14,165,233,0.3)' : 'none',
                    }}
                  >
                    {log}
                  </div>
                );
              })}
              {!isDone && (
                <div className="flex items-center gap-1.5 mt-1" style={{ color: '#22d3ee' }}>
                  <span className="text-xs">❯</span>
                  <span className="animate-pulse" style={{ width: 6, height: 13, background: '#22d3ee', display: 'inline-block', borderRadius: 1 }} />
                </div>
              )}
              <div ref={logEndRef} />
            </div>
          </div>

        </div>

        {/* ── ROW 5: FOOTER ────────────────────────────────────────── */}
        <div
          className="shrink-0 flex items-center justify-between"
          style={{ padding: '7px 16px', borderRadius: 10, background: 'rgba(15,23,42,0.5)', border: '1px solid rgba(51,65,85,0.35)' }}
        >
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
            <Lock className="w-3 h-3 text-slate-600" />
            Encrypted Sandbox · Zero-Storage · GDPR Audit Pipeline
          </div>
          <div className="text-[10px] font-mono font-bold" style={{ color: isDone ? '#34d399' : '#67e8f9' }}>
            {isDone ? '✓ Screening complete — preparing results...' : '⟳ Please remain on this screen'}
          </div>
        </div>

      </div>{/* end inner padding */}
      </div>{/* end centered card */}
    </div>,
    document.body
  );
}
