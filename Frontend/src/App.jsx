import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import AuditTrailPage from './pages/AuditTrailPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'result' | 'audit'
  const [verificationResult, setVerificationResult] = useState(null);

  const handleVerificationComplete = (data) => {
    setVerificationResult(data);
    setActiveTab('result');
  };

  const handleBackToUpload = () => {
    setVerificationResult(null);
    setActiveTab('upload');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between selection:bg-cyan-500 selection:text-white">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Page Area */}
      <main className="flex-1">
        {activeTab === 'upload' && (
          <UploadPage onVerificationComplete={handleVerificationComplete} />
        )}

        {activeTab === 'result' && (
          <ResultPage
            resultData={verificationResult}
            onBackToUpload={handleBackToUpload}
          />
        )}

        {activeTab === 'audit' && <AuditTrailPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-6 text-center text-xs font-mono text-slate-500 bg-slate-950">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-2">
          <span>SENTINEL AI Border Screening & Identity Security System</span>
          <span>FastAPI + PaddleOCR + DeepFace + Mistral AI + RandomForest</span>
        </div>
      </footer>
    </div>
  );
}
