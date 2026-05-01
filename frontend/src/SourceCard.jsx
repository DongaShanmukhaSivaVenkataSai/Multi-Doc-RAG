import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';

function SourceCard({ source }) {
    const [expanded, setExpanded] = useState(false);

    const score = typeof source.score === 'number' ? source.score : 0;
    const scoreDisplay = score.toFixed(4);
    const isHighConfidence = score > 0.7;

    const previewLength = 150;
    const fullText = source.text || '';
    const isLong = fullText.length > previewLength;
    const displayText = expanded || !isLong ? fullText : fullText.slice(0, previewLength) + '...';

    return (
        <div className="source-card">
            <div className="source-card-header">
                <div className="source-card-title">
                    <FileText size={14} />
                    <span title={source.doc_name}>{source.doc_name || 'Document'}</span>
                    {source.page ? <span className="source-page">p.{source.page}</span> : null}
                </div>
                <div className={`source-card-score ${isHighConfidence ? 'high-confidence' : ''}`}>
                    {scoreDisplay}
                </div>
            </div>
            <div className="source-card-text">{displayText}</div>
            {isLong && (
                <button className="source-expand-btn" onClick={() => setExpanded(e => !e)}>
                    {expanded ? <><ChevronUp size={12} /> less</> : <><ChevronDown size={12} /> more</>}
                </button>
            )}
        </div>
    );
}

export default SourceCard;
