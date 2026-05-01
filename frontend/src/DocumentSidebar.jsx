import React, { useRef } from 'react';
import { UploadCloud, FileText, Loader2, Database, Trash2 } from 'lucide-react';

function DocumentSidebar({ documents, onUpload, onDelete, isUploading }) {
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            onUpload(Array.from(e.target.files));
            e.target.value = null;
        }
    };

    const formatDate = (iso) => {
        if (!iso) return '';
        const d = new Date(iso);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    };

    return (
        <aside className="sidebar glass-panel">
            <div className="sidebar-header">
                <div className="logo-container">
                    <Database className="logo-icon" size={24} />
                    <h1 className="logo-text gradient-text">Multi-Doc RAG</h1>
                </div>
                <p className="sidebar-subtitle">Knowledge Base</p>
            </div>

            <div className="upload-section">
                <input
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt,.md"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />
                <button
                    className={`upload-btn ${isUploading ? 'uploading' : ''}`}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                >
                    {isUploading ? (
                        <>
                            <Loader2 className="spinner" size={18} />
                            <span>Processing...</span>
                        </>
                    ) : (
                        <>
                            <UploadCloud size={18} />
                            <span>Upload Documents</span>
                        </>
                    )}
                </button>
            </div>

            <div className="document-list-container">
                <h3 className="section-title">
                    Uploaded Files <span className="badge">{documents.length}</span>
                </h3>

                {documents.length === 0 ? (
                    <div className="empty-state">
                        <FileText size={32} className="empty-icon" />
                        <p>No documents uploaded yet.</p>
                    </div>
                ) : (
                    <ul className="document-list">
                        {documents.map((doc, idx) => (
                            <li key={idx} className="document-item">
                                <FileText size={16} className="doc-icon" />
                                <div className="doc-info">
                                    <span className="doc-name" title={doc.doc_name}>{doc.doc_name}</span>
                                    <span className="doc-meta">{doc.num_chunks} chunks &middot; {formatDate(doc.upload_time)}</span>
                                </div>
                                <button
                                    className="doc-delete-btn"
                                    title="Delete document"
                                    onClick={() => onDelete(doc.doc_name)}
                                >
                                    <Trash2 size={14} />
                                </button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </aside>
    );
}

export default DocumentSidebar;
