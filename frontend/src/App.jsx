import React, { useState, useEffect } from 'react';
import axios from 'axios';
import DocumentSidebar from './DocumentSidebar';
import ChatPanel from './ChatPanel';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Welcome to **Multi-Doc RAG**. Upload your documents (PDF, DOCX, TXT, MD) and ask me anything about them.' }
  ]);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_BASE}/documents`);
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents', err);
    }
  };

  const handleToggleSelectDoc = (docName) => {
    setSelectedDocs(prev =>
      prev.includes(docName) ? prev.filter(d => d !== docName) : [...prev, docName]
    );
  };

  const handleSelectAllDocs = () => {
    if (selectedDocs.length === documents.length) {
      setSelectedDocs([]);
    } else {
      setSelectedDocs(documents.map(d => d.doc_name));
    }
  };

  const handleUpload = async (files) => {
    if (!files.length) return;
    setIsUploading(true);
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        await axios.post(`${API_BASE}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }
      await fetchDocuments();
    } catch (err) {
      console.error('Upload failed', err);
      const detail = err?.response?.data?.detail || 'Unknown error';
      alert(`Upload failed: ${detail}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docName) => {
    if (!window.confirm(`Delete "${docName}"?`)) return;
    try {
      await axios.delete(`${API_BASE}/documents/${encodeURIComponent(docName)}`);
      setSelectedDocs(prev => prev.filter(d => d !== docName));
      await fetchDocuments();
    } catch (err) {
      console.error('Delete failed', err);
      alert('Failed to delete document.');
    }
  };

  return (
    <div className="app-layout">
      <DocumentSidebar
        documents={documents}
        selectedDocs={selectedDocs}
        onToggleSelectDoc={handleToggleSelectDoc}
        onSelectAllDocs={handleSelectAllDocs}
        onUpload={handleUpload}
        onDelete={handleDelete}
        isUploading={isUploading}
      />
      <ChatPanel
        messages={messages}
        setMessages={setMessages}
        apiBase={API_BASE}
        selectedDocs={selectedDocs}
      />
    </div>
  );
}

export default App;
