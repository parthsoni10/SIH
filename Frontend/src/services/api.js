import axios from 'axios';

const API_BASE = '/api';

export const verifyDocument = async (documentFile, liveCaptureBlob, documentType) => {
  const formData = new FormData();
  formData.append('document_file', documentFile);
  formData.append('document_type', documentType);

  if (liveCaptureBlob) {
    formData.append('live_capture_file', liveCaptureBlob, 'live_capture.jpg');
  }

  const response = await axios.post(`${API_BASE}/documents/verify`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const fetchAuditTrail = async (page = 1, limit = 10, documentType = null, prediction = null) => {
  const params = { page, limit };
  if (documentType) params.document_type = documentType;
  if (prediction) params.prediction = prediction;

  const response = await axios.get(`${API_BASE}/audit`, { params });
  return response.data;
};

export const fetchAuditRecord = async (auditId) => {
  const response = await axios.get(`${API_BASE}/audit/${auditId}`);
  return response.data;
};
