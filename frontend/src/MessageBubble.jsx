import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot } from 'lucide-react';
import SourceCard from './SourceCard';

function MessageBubble({ message }) {
    const isUser = message.role === 'user';

    return (
        <div className={`message-wrapper ${isUser ? 'user' : 'assistant'}`}>
            <div className="message-content">
                <div className={`avatar ${isUser ? 'user' : 'assistant'}`}>
                    {isUser ? <User size={20} /> : <Bot size={20} />}
                </div>

                <div className={`message-bubble ${!isUser ? 'glass-panel' : ''}`}>
                    <div className="markdown-body">
                        {isUser ? (
                            <p>{message.text}</p>
                        ) : (
                            <ReactMarkdown>{message.text}</ReactMarkdown>
                        )}
                    </div>

                    {!isUser && message.sources && message.sources.length > 0 && (
                        <div className="sources-container">
                            <div className="sources-header">
                                <span>Sources</span>
                            </div>
                            <div className="sources-grid">
                                {message.sources.map((src, idx) => (
                                    <SourceCard key={idx} source={src} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default MessageBubble;
