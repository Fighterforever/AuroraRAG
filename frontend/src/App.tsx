import { useMemo, useState } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  communities?: CommunityInsight[];
}

interface CommunityInsight {
  community_id: string;
  title: string;
  summary: string;
  keywords: string[];
  members: string[];
  top_entities?: Array<{ id: string; name: string; degree?: number }>;
  bridge_entities?: Array<{ id: string; name: string; external_degree?: number }>;
}

interface QueryResponse {
  answer: {
    answer: string;
    relevant_entities: string[];
    question_analysis: Record<string, unknown>;
    evidence_plan: {
      steps: Array<Record<string, unknown>>;
    };
    critic: Record<string, unknown>;
  };
}

const createId = () => Math.random().toString(36).slice(2);

function extractCommunityContext(answer: QueryResponse['answer']): CommunityInsight[] {
  const steps = answer?.evidence_plan?.steps || [];
  const block = steps.find((step) => step.step === 'community_context');
  if (!block || !Array.isArray(block.communities)) return [];
  return block.communities as CommunityInsight[];
}

function splitIntoParagraphs(text: string): string[] {
  return text.split(/\n+/).filter(Boolean);
}

const App = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);

  const currentSummary = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((msg) => msg.role === 'assistant');
    return lastAssistant?.communities?.[0];
  }, [messages]);

  const handleSend = async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || loading) return;

    const conversationId = activeConversation ?? createId();
    if (!activeConversation) {
      setActiveConversation(conversationId);
      setHistory((prev) => [conversationId, ...prev]);
    }

    const userMessage: Message = {
      id: createId(),
      role: 'user',
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data: QueryResponse = await response.json();
      const answerText = data.answer?.answer ?? '未能生成回答。';
      const communities = extractCommunityContext(data.answer);
      const assistantMessage: Message = {
        id: createId(),
        role: 'assistant',
        content: answerText,
        communities,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: unknown) {
      const assistantMessage: Message = {
        id: createId(),
        role: 'assistant',
        content: error instanceof Error ? `请求失败：${error.message}` : '请求失败',
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setLoading(false);
    }
  };

  const startNewConversation = () => {
    setMessages([]);
    setActiveConversation(null);
    setInputValue('');
  };

  return (
    <div className="app-shell">
      <aside className="glass-panel sidebar">
        <div className="logo">
          <span>🧭</span>
          <span>AuroraRAG</span>
        </div>
        <button onClick={startNewConversation}>新建对话</button>
        <div className="history">
          {history.map((id) => (
            <div
              key={id}
              className={`history-item ${id === activeConversation ? 'active' : ''}`}
              onClick={() => {
                if (id === activeConversation) return;
                setActiveConversation(id);
                setMessages([]);
              }}
            >
              {id.slice(0, 6)}
            </div>
          ))}
        </div>
      </aside>

      <main className="glass-panel chat-panel">
        <div className="chat-header">
          <h2>知识问答</h2>
          {loading && <span className="typing-indicator">生成中…</span>}
        </div>

        <div className="message-list">
          {messages.map((msg) => (
            <article key={msg.id} className={`message ${msg.role}`}>
              {splitIntoParagraphs(msg.content).map((paragraph, idx) => (
                <p key={idx} style={{ margin: idx === 0 ? 0 : '12px 0 0' }}>
                  {paragraph}
                </p>
              ))}
              {msg.communities && msg.communities.length > 0 && (
                <div className="community-block">
                  <strong>相关社区</strong>
                  {msg.communities.map((community) => (
                    <div key={community.community_id} style={{ marginTop: 8 }}>
                      <div style={{ fontWeight: 600 }}>{community.title}</div>
                      <div style={{ fontSize: 13, color: 'rgba(226,232,240,0.7)', marginTop: 4 }}>
                        {community.summary}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </article>
          ))}
          {messages.length === 0 && (
            <div className="placeholder">
              <div className="placeholder-icon">✨</div>
              <div className="placeholder-text">开始一个问题，探索图谱洞见。</div>
            </div>
          )}
        </div>

        <div className="input-bar">
          <textarea
            placeholder="在这里输入问题…"
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSend();
              }
            }}
          />
          <button type="button" onClick={handleSend} disabled={loading}>
            发送
          </button>
        </div>
      </main>

      <aside className="glass-panel insight-panel">
        <section className="insight-card">
          <h3>社区摘要</h3>
          {currentSummary ? (
            <div>
              <div className="insight-title">{currentSummary.title}</div>
              <p className="insight-body">{currentSummary.summary}</p>
              <div className="keyword-row">
                {currentSummary.keywords?.map((keyword) => (
                  <span key={keyword} className="keyword-chip">
                    {keyword}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <p className="insight-placeholder">暂无社区信息</p>
          )}
        </section>

        <section className="insight-card">
          <h3>核心成员</h3>
          {currentSummary?.members && currentSummary.members.length > 0 ? (
            <ul className="member-list">
              {currentSummary.members.slice(0, 6).map((member) => (
                <li key={member}>{member}</li>
              ))}
            </ul>
          ) : (
            <p className="insight-placeholder">暂无成员数据</p>
          )}
        </section>

        <section className="insight-card">
          <h3>关键指标</h3>
          <div className="metrics-grid">
            <div className="metric-box">
              <span>回答轮数</span>
              <strong>{messages.filter((msg) => msg.role === 'assistant').length}</strong>
            </div>
            <div className="metric-box">
              <span>社区规模</span>
              <strong>{currentSummary ? currentSummary.members.length : 0}</strong>
            </div>
          </div>
        </section>
      </aside>
    </div>
  );
};

export default App;
