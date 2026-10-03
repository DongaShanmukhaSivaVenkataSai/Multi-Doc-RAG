import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import MessageBubble from './MessageBubble';

function ChatPanel({ messages, setMessages, apiBase, selectedDocs = [] }) {
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const textareaRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const handleInput = (e) => {
        setInput(e.target.value);
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
        }
    };

    const buildChatHistory = (msgs) => {
        return msgs
            .filter(m => m.role === 'user' || m.role === 'assistant')
            .map(m => ({ role: m.role, content: m.text }));
    };

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userText = input.trim();
        const userMsg = { role: 'user', text: userText };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }

        // Placeholder for the streaming assistant message
        const assistantMsgId = Date.now();
        setMessages(prev => [...prev, { id: assistantMsgId, role: 'assistant', text: '', sources: [] }]);

        try {
            const response = await fetch(`${apiBase}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: userText,
                    chat_history: buildChatHistory(messages),
                    top_k: 5,
                    doc_ids: selectedDocs.length > 0 ? selectedDocs : null,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const raw = line.slice(6).trim();
                    if (!raw) continue;

                    let event;
                    try { event = JSON.parse(raw); } catch { continue; }

                    if (event.type === 'sources') {
                        setMessages(prev => prev.map(m =>
                            m.id === assistantMsgId ? { ...m, sources: event.sources } : m
                        ));
                    } else if (event.type === 'token') {
                        setMessages(prev => prev.map(m =>
                            m.id === assistantMsgId ? { ...m, text: m.text + event.content } : m
                        ));
                    } else if (event.type === 'done') {
                        break;
                    }
                }
            }
        } catch (err) {
            console.error('Chat error:', err);
            setMessages(prev => prev.map(m =>
                m.id === assistantMsgId
                    ? { ...m, text: 'Sorry, I encountered an error. Please ensure the backend is running and documents are uploaded.' }
                    : m
            ));
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <main className="chat-panel">
            <div className="chat-history">
                {messages.map((msg, idx) => (
                    <MessageBubble key={idx} message={msg} />
                ))}

                {isLoading && (
                    <div className="message-wrapper assistant">
                        <div className="message-content">
                            <div className="avatar assistant">
                                <Loader2 size={20} className="spinner" />
                            </div>
                            <div className="message-bubble glass-panel message-loading">
                                Thinking...
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-container">
                {selectedDocs.length > 0 && (
                    <div style={{ padding: '0 8px 6px 8px', fontSize: '12px', color: '#818cf8', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span>🎯 Filtering query to {selectedDocs.length} selected document{selectedDocs.length > 1 ? 's' : ''}</span>
                    </div>
                )}
                <div className="input-box glass-panel">
                    <textarea
                        ref={textareaRef}
                        className="chat-input"
                        placeholder="Ask a question about your documents..."
                        rows={1}
                        value={input}
                        onChange={handleInput}
                        onKeyDown={handleKeyDown}
                        disabled={isLoading}
                    />
                    <button
                        className="send-btn"
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </main>
    );
}

export default ChatPanel;
