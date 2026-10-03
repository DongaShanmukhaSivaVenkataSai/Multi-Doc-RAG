import React, { useRef } from 'react';
import { UploadCloud, FileText, Loader2, Database, Trash2, CheckSquare, Square } from 'lucide-react';

function DocumentSidebar({
    documents,
    selectedDocs = [],
    onToggleSelectDoc,
    onSelectAllDocs,
    onUpload,
    onDelete,
    isUploading,
}) {
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
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 className="section-title" style={{ margin: 0 }}>
                        Uploaded Files <span className="badge">{documents.length}</span>
                    </h3>
                    {documents.length > 0 && (
                        <button
                            onClick={onSelectAllDocs}
                            style={{
                                background: 'transparent',
                                border: '1px solid rgba(255,255,255,0.2)',
                                color: '#ccc',
                                borderRadius: '4px',
                                fontSize: '11px',
                                padding: '2px 6px',
                                cursor: 'pointer',
                            }}
                        >
                            {selectedDocs.length === 0 ? 'All' : `${selectedDocs.length}/${documents.length}`}
                        </button>
                    )}
                </div>

                {documents.length === 0 ? (
                    <div className="empty-state">
                        <FileText size={32} className="empty-icon" />
                        <p>No documents uploaded yet.</p>
                    </div>
                ) : (
                    <ul className="document-list">
                        {documents.map((doc, idx) => {
                            const isSelected = selectedDocs.includes(doc.doc_name);
                            return (
                                <li key={idx} className={`document-item ${isSelected ? 'selected' : ''}`}>
                                    <button
                                        onClick={() => onToggleSelectDoc(doc.doc_name)}
                                        style={{ background: 'transparent', border: 'none', color: isSelected ? '#6366f1' : '#888', cursor: 'pointer', padding: '0 4px', display: 'flex', alignItems: 'center' }}
                                        title={isSelected ? "Selected for query filter" : "Click to filter queries to this document"}
                                    >
                                        {isSelected ? <CheckSquare size={16} /> : <Square size={16} />}
                                    </button>
                                    <div className="doc-info" onClick={() => onToggleSelectDoc(doc.doc_name)} style={{ cursor: 'pointer' }}>
                                        <span className="doc-name" title={doc.doc_name}>{doc.doc_name}</span>
                                        <span className="doc-meta">{doc.num_chunks} chunks &middot; {formatDate(doc.upload_time)}</span>
                                    </div>
                                    <button
                                        className="doc-delete-btn"
                                        title="Delete document"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDelete(doc.doc_name);
                                        }}
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </li>
                            );
                        })}
                    </ul>
                )}
            </div>
        </aside>
    );
}

export default DocumentSidebar;
