import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle, X, RotateCcw } from 'lucide-react';

export default function DocumentUploader({ selectedFile, setSelectedFile, backSideFile, setBackSideFile }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isBackDragging, setIsBackDragging] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleBackFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setBackSideFile(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleBackDrop = (e) => {
    e.preventDefault();
    setIsBackDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setBackSideFile(e.dataTransfer.files[0]);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    // Clearing front side also clears back side
    if (setBackSideFile) setBackSideFile(null);
  };

  const clearBackFile = () => {
    setBackSideFile(null);
  };

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
          <FileText className="w-4 h-4 text-cyan-400" />
          <span>Identity Document Upload</span>
        </label>
        <span className="text-xs text-slate-400 font-mono">Formats: JPEG, PNG, PDF (max 15MB)</span>
      </div>

      {/* Front Side Upload */}
      {!selectedFile ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-cyan-400 bg-cyan-500/10'
              : 'border-slate-800 hover:border-slate-700 bg-slate-900/50'
          }`}
          onClick={() => document.getElementById('doc-file-input').click()}
        >
          <input
            id="doc-file-input"
            type="file"
            accept="image/jpeg,image/png,application/pdf"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="w-12 h-12 rounded-full bg-cyan-500/10 flex items-center justify-center mb-3 border border-cyan-500/20 text-cyan-400">
            <UploadCloud className="w-6 h-6" />
          </div>
          <p className="text-sm font-medium text-slate-200 mb-1">
            Click to upload or drag & drop document image
          </p>
          <p className="text-xs text-slate-400">
            Passports, Visas, Aadhaar, PAN Cards, Driving Licenses
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Front side file confirmation */}
          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-900/90 border border-slate-800">
            <div className="flex items-center space-x-3 overflow-hidden">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
                <CheckCircle className="w-5 h-5" />
              </div>
              <div className="truncate">
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full">Front Side</span>
                </div>
                <p className="text-sm font-semibold text-slate-200 truncate mt-0.5">{selectedFile.name}</p>
                <p className="text-xs text-slate-400 font-mono">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || 'Document'}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={clearFile}
              className="p-2 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-all shrink-0"
              title="Remove selected file"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Back Side Upload — appears after front side is uploaded */}
          {!backSideFile ? (
            <div
              onDragOver={(e) => { e.preventDefault(); setIsBackDragging(true); }}
              onDragLeave={() => setIsBackDragging(false)}
              onDrop={handleBackDrop}
              className={`border-2 border-dashed rounded-xl p-5 flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
                isBackDragging
                  ? 'border-amber-400 bg-amber-500/10'
                  : 'border-slate-800 hover:border-slate-700 bg-slate-900/50'
              }`}
              onClick={() => document.getElementById('doc-back-file-input').click()}
            >
              <input
                id="doc-back-file-input"
                type="file"
                accept="image/jpeg,image/png,application/pdf"
                className="hidden"
                onChange={handleBackFileChange}
              />
              <div className="w-10 h-10 rounded-full bg-amber-500/10 flex items-center justify-center mb-2 border border-amber-500/20 text-amber-400">
                <RotateCcw className="w-5 h-5" />
              </div>
              <p className="text-sm font-medium text-slate-200 mb-0.5">
                Upload Back Side <span className="text-slate-400 font-normal">(Optional)</span>
              </p>
              <p className="text-xs text-slate-500">
                For documents with info on both sides (Aadhaar, Driving License, etc.)
              </p>
            </div>
          ) : (
            /* Back side file confirmation */
            <div className="flex items-center justify-between p-4 rounded-xl bg-slate-900/90 border border-amber-500/20">
              <div className="flex items-center space-x-3 overflow-hidden">
                <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 shrink-0">
                  <CheckCircle className="w-5 h-5" />
                </div>
                <div className="truncate">
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] uppercase tracking-wider font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full">Back Side</span>
                  </div>
                  <p className="text-sm font-semibold text-slate-200 truncate mt-0.5">{backSideFile.name}</p>
                  <p className="text-xs text-slate-400 font-mono">
                    {(backSideFile.size / (1024 * 1024)).toFixed(2)} MB • {backSideFile.type || 'Document'}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={clearBackFile}
                className="p-2 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-all shrink-0"
                title="Remove back side file"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
