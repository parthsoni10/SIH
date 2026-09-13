import React, { useState } from 'react';
import { ArrowRight, ShieldCheck, Loader2, FileText, CheckCircle2 } from 'lucide-react';
import DocumentUploader from '../components/DocumentUploader';
import CameraWidget from '../components/CameraWidget';
import { verifyDocument } from '../services/api';

const DOCUMENT_TYPES = [
  { id: 'Passport', label: 'Passport (ICAO 9303 MRZ)' },
  { id: 'Visa', label: 'Visa Entry Permit' },
  { id: 'Aadhaar', label: 'Aadhaar Card (Verhoeff Checksum)' },
  { id: 'PAN Card', label: 'PAN Card (Alphanumeric Code)' },
  { id: 'Driving License', label: 'Driving License' },
];

const PROCESSING_STAGES = [
  'Pre-processing & image normalization...',
  'Running PaddleOCR & MRZ checksum extraction...',
  'Validating document rules & blacklist database...',
  'Analyzing ELA & image tampering forensics...',
  'Computing live face match similarity...',
  'Evaluating 12-feature Random Forest risk model...',
  'Generating Gemini AI officer summary...'
];

export default function UploadPage({ onVerificationComplete }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [liveBlob, setLiveBlob] = useState(null);
  const [documentType, setDocumentType] = useState('Passport');
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMsg('Please select or drag-and-drop a document file to evaluate.');
      return;
    }

    setErrorMsg(null);
    setIsProcessing(true);
    setCurrentStageIdx(0);

    // Simulate stage progress timer
    const interval = setInterval(() => {
      setCurrentStageIdx((prev) => {
        if (prev < PROCESSING_STAGES.length - 1) return prev + 1;
        return prev;
      });
    }, 450);

    try {
      const resultData = await verifyDocument(selectedFile, liveBlob, documentType);
      clearInterval(interval);
      setIsProcessing(false);
      onVerificationComplete(resultData);
    } catch (err) {
      clearInterval(interval);
      setIsProcessing(false);
      
      let errorMessage;
      if (err.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out — the server took too long to respond. Please try again.';
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      } else {
        errorMessage = 'Verification request failed. Please check that the backend server is running.';
      }
      setErrorMsg(errorMessage);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 py-6 px-4">
      {/* Header Banner */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-extrabold text-white tracking-tight">
          Border Identity & Document Verification
        </h2>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto">
          Upload any official identity document for real-time OCR extraction, forensic tampering detection, rule checksum validation, and AI risk prediction.
        </p>
      </div>

      {/* Form Container */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Document Type Selector */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-400 font-semibold block">
            1. Select Document Category
          </label>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {DOCUMENT_TYPES.map((dt) => (
              <button
                key={dt.id}
                type="button"
                onClick={() => setDocumentType(dt.id)}
                className={`p-3 rounded-xl border text-xs font-semibold text-center transition-all ${
                  documentType === dt.id
                    ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300 shadow-md shadow-cyan-500/10'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                {dt.id}
              </button>
            ))}
          </div>
        </div>

        {/* File Uploader & Live Camera Widget Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <DocumentUploader
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
          />

          <CameraWidget
            onCapture={(blob) => setLiveBlob(blob)}
            capturedBlob={liveBlob}
            setCapturedBlob={setLiveBlob}
          />
        </div>

        {/* Error Alert */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-medium text-center">
            {errorMsg}
          </div>
        )}

        {/* Submit Action Button */}
        <div className="flex justify-center pt-2">
          <button
            type="submit"
            disabled={isProcessing || !selectedFile}
            className={`w-full md:w-auto min-w-[280px] flex items-center justify-center space-x-3 px-8 py-4 rounded-2xl text-sm font-bold tracking-wide transition-all shadow-xl ${
              isProcessing || !selectedFile
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white shadow-cyan-500/25 hover:shadow-cyan-500/40 cursor-pointer transform hover:-translate-y-0.5'
            }`}
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin text-cyan-300" />
                <span>Running Screening Pipeline...</span>
              </>
            ) : (
              <>
                <ShieldCheck className="w-5 h-5" />
                <span>Screen Document Now</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Stage Loading Overlay Modal */}
      {isProcessing && (
        <div className="fixed inset-0 z-50 glass-panel bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-8 rounded-3xl border border-slate-800 text-center space-y-6 shadow-2xl">
            <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto text-cyan-400">
              <Loader2 className="w-8 h-8 animate-spin" />
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-bold text-white">Screening Pipeline Running</h3>
              <p className="text-xs font-mono text-cyan-400 h-6">
                {PROCESSING_STAGES[currentStageIdx]}
              </p>
            </div>

            {/* Stage Progress Bar */}
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div
                className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full transition-all duration-300"
                style={{ width: `${((currentStageIdx + 1) / PROCESSING_STAGES.length) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
