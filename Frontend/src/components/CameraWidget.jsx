import React, { useRef, useState, useEffect } from 'react';
import { Camera, RefreshCw, CheckCircle, VideoOff } from 'lucide-react';

export default function CameraWidget({ onCapture, capturedBlob, setCapturedBlob }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streamActive, setStreamActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      // videoRef is always in the DOM now, so this will always succeed
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Don't call setStreamActive here — the onPlaying event handles it
        // This avoids the race condition where videoRef was null
        videoRef.current.play().catch(() => {});
      }
    } catch (err) {
      setCameraError('Camera access denied or unavailable.');
      setStreamActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setStreamActive(false);
  };

  const takeSnapshot = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        setCapturedBlob(blob);
        const url = URL.createObjectURL(blob);
        setPreviewUrl(url);
        onCapture(blob);
        stopCamera();
      }
    }, 'image/jpeg', 0.92);
  };

  const clearSnapshot = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setCapturedBlob(null);
    setPreviewUrl(null);
    onCapture(null);
    startCamera();
  };

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2 text-sm font-semibold text-slate-200">
          <Camera className="w-4 h-4 text-cyan-400" />
          <span>Live Face Verification Capture</span>
        </div>
        <span className="text-xs text-slate-400 font-mono">(Optional for Face Match)</span>
      </div>

      <div className="relative w-full aspect-video bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
        {/* Hidden Canvas */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Video element is ALWAYS in the DOM so videoRef is never null */}
        <video
          ref={videoRef}
          className={`w-full h-full object-cover transform -scale-x-100 ${
            streamActive && !previewUrl ? '' : 'hidden'
          }`}
          muted
          playsInline
          autoPlay
          onPlaying={() => setStreamActive(true)}
        />

        {/* Snapshot Preview — overlays the video */}
        {previewUrl ? (
          <div className="relative w-full h-full">
            <img src={previewUrl} alt="Live face snapshot" className="w-full h-full object-cover" />
            <div className="absolute top-3 left-3 bg-emerald-500/90 text-white text-xs font-semibold px-2.5 py-1 rounded-full flex items-center space-x-1 shadow-md">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Face Captured</span>
            </div>
          </div>
        ) : !streamActive ? (
          /* Camera Disabled / Error — shown when stream is not active */
          <div className="flex flex-col items-center justify-center p-6 text-center text-slate-500">
            <VideoOff className="w-10 h-10 mb-2 opacity-50 text-slate-400" />
            <p className="text-xs">{cameraError || 'Camera stream offline'}</p>
            <button
              type="button"
              onClick={startCamera}
              className="mt-3 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-all"
            >
              Start Camera
            </button>
          </div>
        ) : null}
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-between">
        {previewUrl ? (
          <button
            type="button"
            onClick={clearSnapshot}
            className="w-full flex items-center justify-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-all"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Retake Live Snapshot</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={takeSnapshot}
            disabled={!streamActive}
            className={`w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-all ${
              streamActive
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/20'
                : 'bg-slate-800 text-slate-500 cursor-not-allowed'
            }`}
          >
            <Camera className="w-4 h-4" />
            <span>Capture Photo</span>
          </button>
        )}
      </div>
    </div>
  );
}
