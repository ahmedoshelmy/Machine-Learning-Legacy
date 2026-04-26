import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

const CHAT_MODELS = [
  { id: 'llama-3.1-8b-instant',                      label: 'Llama 3.1 8B',      speed: '560 t/s',  ctx: '131K', badge: 'Fast'     },
  { id: 'llama-3.3-70b-versatile',                   label: 'Llama 3.3 70B',     speed: '280 t/s',  ctx: '131K', badge: 'Smart'    },
  { id: 'openai/gpt-oss-120b',                       label: 'GPT-OSS 120B',      speed: '500 t/s',  ctx: '131K', badge: 'Powerful' },
  { id: 'openai/gpt-oss-20b',                        label: 'GPT-OSS 20B',       speed: '1000 t/s', ctx: '131K', badge: 'Fastest'  },
  { id: 'meta-llama/llama-4-scout-17b-16e-instruct', label: 'Llama 4 Scout 17B', speed: '750 t/s',  ctx: '131K', badge: 'Preview'  },
  { id: 'qwen/qwen3-32b',                            label: 'Qwen3 32B',         speed: '400 t/s',  ctx: '131K', badge: 'Preview'  },
  { id: 'groq/compound',                             label: 'Compound',          speed: '450 t/s',  ctx: '131K', badge: 'System'   },
];

const IMAGE_MODELS = [
  { id: 'flux-2-klein', label: 'Flux 2 Klein', desc: 'Fast & efficient',     color: '#6366f1' },
  { id: 'flux-2-pro',   label: 'Flux 2 Pro',   desc: 'High quality outputs', color: '#8b5cf6' },
  { id: 'mystic',       label: 'Mystic',        desc: 'Artistic & creative',  color: '#ec4899' },
];

const BADGE_COLORS = { Fast: '#0ea5e9', Smart: '#10b981', Powerful: '#f59e0b', Fastest: '#ef4444', Preview: '#8b5cf6', System: '#6366f1' };
const STATUS_META  = { completed: { color: '#4ade80', label: 'Completed' }, generating: { color: '#f59e0b', label: 'Generating' }, failed: { color: '#f87171', label: 'Failed' } };

const STORAGE_KEY = 'ai_studio_tasks';

function loadTasks() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch { return []; }
}
function saveTasks(tasks) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks)); } catch {}
}

export default function Home() {
  const [view, setView]                   = useState('chat');
  const [chatModel, setChatModel]         = useState(CHAT_MODELS[0]);
  const [imageModel, setImageModel]       = useState(IMAGE_MODELS[0]);
  const [messages, setMessages]           = useState([{ role: 'assistant', content: `👋 Hello! I'm powered by **${CHAT_MODELS[0].label}**. How can I help you today?`, meta: null }]);
  const [chatInput, setChatInput]         = useState('');
  const [imagePrompt, setImagePrompt]     = useState('');
  const [negPrompt, setNegPrompt]         = useState('');
  const [tasks, setTasks]                 = useState([]);
  const [loading, setLoading]             = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [imgSettings, setImgSettings]     = useState({ aspect_ratio: 'square_1_1', num_images: 1, guidance_scale: 1.0 });
  const [lightbox, setLightbox]           = useState(null); // { url, task }
  const [taskFilter, setTaskFilter]       = useState('all'); // all | completed | generating | failed
  const scrollRef = useRef(null);

  // Load persisted tasks on mount
  useEffect(() => { setTasks(loadTasks()); }, []);

  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  useEffect(() => {
    const handler = (e) => { if (!e.target.closest('#model-picker-root')) setShowModelPicker(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Close lightbox on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') setLightbox(null); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const upsertTask = (task) => {
    setTasks(prev => {
      const next = prev.find(t => t.id === task.id)
        ? prev.map(t => t.id === task.id ? task : t)
        : [task, ...prev];
      saveTasks(next);
      return next;
    });
  };

  const deleteTask = (id) => {
    setTasks(prev => { const next = prev.filter(t => t.id !== id); saveTasks(next); return next; });
  };

  const clearAllTasks = () => { setTasks([]); saveTasks([]); };

  const handleChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || loading) return;
    const userMsg = { role: 'user', content: chatInput, meta: null };
    const newMsgs = [...messages, userMsg];
    setMessages(newMsgs);
    setChatInput('');
    setLoading(true);
    const t0 = Date.now();
    try {
      const res     = await axios.post('/api/chat', { mode: 'chat', messages: newMsgs.map(m => ({ role: m.role, content: m.content })), model: chatModel.id });
      const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
      const choice  = res.data.choices[0];
      const usage   = res.data.usage;
      setMessages([...newMsgs, { ...choice.message, meta: { model: res.data.model, elapsed, tokens: usage?.total_tokens, prompt_tokens: usage?.prompt_tokens, completion_tokens: usage?.completion_tokens, finish_reason: choice.finish_reason } }]);
    } catch (err) {
      setMessages([...newMsgs, { role: 'assistant', content: '⚠️ ' + (err.response?.data?.error || 'Connection failed.'), meta: null }]);
    } finally { setLoading(false); }
  };

  const handleImage = async (e) => {
    e.preventDefault();
    if (!imagePrompt.trim() || loading) return;
    setLoading(true);

    const taskId = `task_${Date.now()}`;
    const task = {
      id: taskId,
      prompt: imagePrompt,
      negPrompt,
      model: imageModel.label,
      modelId: imageModel.id,
      modelColor: imageModel.color,
      aspect_ratio: imgSettings.aspect_ratio,
      num_images: imgSettings.num_images,
      guidance_scale: imgSettings.guidance_scale,
      status: 'generating',
      url: null,
      seed: null,
      ts: new Date().toISOString(),
      elapsed: null,
    };
    upsertTask(task);
    setView('tasks'); // jump to tasks panel so user sees progress

    const t0 = Date.now();
    try {
      const res = await axios.post('/api/chat', { mode: 'image', prompt: imagePrompt, negativePrompt: negPrompt, model: imageModel.id, ...imgSettings });
      if (res.data.url) {
        upsertTask({ ...task, status: 'completed', url: res.data.url, seed: res.data.seed, elapsed: ((Date.now() - t0) / 1000).toFixed(1) });
        setImagePrompt('');
      }
    } catch (err) {
      upsertTask({ ...task, status: 'failed', error: err.response?.data?.error || err.message });
    } finally { setLoading(false); }
  };

  const clearChat = () => setMessages([{ role: 'assistant', content: `👋 Chat cleared. Using **${chatModel.label}**. Ask me anything!`, meta: null }]);

  const filteredTasks = taskFilter === 'all' ? tasks : tasks.filter(t => t.status === taskFilter);

  return (
    <div style={s.app}>
      {/* Sidebar */}
      <nav style={s.sidebar}>
        <div style={s.brand}><span style={{ fontSize: 22 }}>⚡</span><span>AI Studio</span></div>
        <div style={s.navSection}>WORKSPACE</div>
        <NavBtn icon="💬" label="Chat"   active={view === 'chat'}   onClick={() => setView('chat')}   color="#3b82f6" />
        <NavBtn icon="🎨" label="Studio" active={view === 'studio'} onClick={() => setView('studio')} color="#8b5cf6" />
        <NavBtn icon="🗂️" label="Tasks"
          active={view === 'tasks'}
          onClick={() => setView('tasks')}
          color="#f59e0b"
          badge={tasks.filter(t => t.status === 'generating').length || null}
        />
        <div style={{ flex: 1 }} />
        <div style={s.sidebarFooter}>
          <div style={s.footerLabel}>Active Chat Model</div>
          <div style={s.footerModel}>{chatModel.label}</div>
          <div style={{ ...s.badge, background: BADGE_COLORS[chatModel.badge] }}>{chatModel.badge}</div>
        </div>
      </nav>

      {/* Main */}
      <main style={s.workspace}>

        {/* ── CHAT ── */}
        {view === 'chat' && (
          <div style={s.chatLayout}>
            <div style={s.chatHeader}>
              <div>
                <div style={s.chatTitle}>Chat</div>
                <div style={s.chatSub}>{messages.length - 1} messages</div>
              </div>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <div id="model-picker-root" style={{ position: 'relative' }}>
                  <button style={s.modelBtn} onClick={() => setShowModelPicker(v => !v)}>
                    <span style={{ ...s.badge, background: BADGE_COLORS[chatModel.badge], marginRight: 6 }}>{chatModel.badge}</span>
                    {chatModel.label}
                    <span style={{ marginLeft: 6, opacity: 0.6 }}>▾</span>
                  </button>
                  {showModelPicker && (
                    <div style={s.dropdown}>
                      <div style={s.dropdownTitle}>Select Chat Model</div>
                      {CHAT_MODELS.map(m => (
                        <button key={m.id} style={{ ...s.dropdownItem, background: chatModel.id === m.id ? '#1e3a5f' : 'transparent' }}
                          onClick={() => { setChatModel(m); setShowModelPicker(false); }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontWeight: 600, color: '#f1f5f9' }}>{m.label}</span>
                            <span style={{ ...s.badge, background: BADGE_COLORS[m.badge] }}>{m.badge}</span>
                          </div>
                          <div style={s.dropdownMeta}>{m.speed} · ctx {m.ctx}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button style={s.clearBtn} onClick={clearChat}>Clear</button>
              </div>
            </div>

            <div style={s.chatArea}>
              {messages.map((m, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: m.role === 'user' ? 'flex-end' : 'flex-start', gap: 4 }}>
                  <div style={s.roleLabel}>{m.role === 'user' ? '🧑 You' : '🤖 Assistant'}</div>
                  <div style={{ ...s.bubble, background: m.role === 'user' ? 'linear-gradient(135deg,#1d4ed8,#2563eb)' : '#1e293b', borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px' }}>
                    {m.role === 'user' ? m.content : <MarkdownMessage content={m.content} />}
                  </div>
                  {m.meta && (
                    <div style={s.metaBar}>
                      <MetaChip icon="🤖" label={m.meta.model} />
                      <MetaChip icon="⏱" label={`${m.meta.elapsed}s`} />
                      <MetaChip icon="🔤" label={`${m.meta.tokens} tokens`} />
                      <MetaChip icon="📥" label={`${m.meta.prompt_tokens} in`} />
                      <MetaChip icon="📤" label={`${m.meta.completion_tokens} out`} />
                      <MetaChip icon="✅" label={m.meta.finish_reason} />
                    </div>
                  )}
                </div>
              ))}
              {loading && view === 'chat' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 4 }}>
                  <div style={s.roleLabel}>🤖 Assistant</div>
                  <div style={{ ...s.bubble, background: '#1e293b' }}>
                    <span className="typing"><span /><span /><span /></span>
                  </div>
                </div>
              )}
              <div ref={scrollRef} />
            </div>

            <form onSubmit={handleChat} style={s.inputBar}>
              <input value={chatInput} onChange={e => setChatInput(e.target.value)} placeholder={`Message ${chatModel.label}...`} style={s.input} onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleChat(e)} />
              <button style={{ ...s.sendBtn, opacity: loading || !chatInput.trim() ? 0.5 : 1 }} disabled={loading || !chatInput.trim()}>
                {loading ? '⏳' : '➤'}
              </button>
            </form>
          </div>
        )}

        {/* ── STUDIO ── */}
        {view === 'studio' && (
          <div style={s.studioLayout}>
            <div style={s.studioHeader}>
              <div style={s.chatTitle}>Image Studio</div>
              <div style={s.chatSub}>{tasks.filter(t => t.status === 'completed').length} images generated</div>
            </div>

            <div style={s.modelTabs}>
              {IMAGE_MODELS.map(m => (
                <button key={m.id} onClick={() => setImageModel(m)}
                  style={{ ...s.modelTab, borderColor: imageModel.id === m.id ? m.color : 'transparent', background: imageModel.id === m.id ? m.color + '22' : '#0f172a' }}>
                  <div style={{ fontWeight: 700, color: imageModel.id === m.id ? m.color : '#94a3b8' }}>{m.label}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{m.desc}</div>
                </button>
              ))}
            </div>

            <div style={s.studioControls}>
              <form onSubmit={handleImage} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <textarea value={imagePrompt} onChange={e => setImagePrompt(e.target.value)} placeholder="Describe your image in detail..." style={s.textarea} rows={3} />
                <input value={negPrompt} onChange={e => setNegPrompt(e.target.value)} placeholder="Negative prompt (optional): blurry, low quality..." style={{ ...s.input, background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: '10px 14px' }} />
                <div style={s.settingsRow}>
                  <label style={s.settingLabel}>
                    Aspect Ratio
                    <select value={imgSettings.aspect_ratio} onChange={e => setImgSettings(p => ({ ...p, aspect_ratio: e.target.value }))} style={s.select}>
                      <option value="square_1_1">Square 1:1</option>
                      <option value="landscape_16_9">Landscape 16:9</option>
                      <option value="portrait_9_16">Portrait 9:16</option>
                      <option value="landscape_4_3">Landscape 4:3</option>
                      <option value="portrait_3_4">Portrait 3:4</option>
                    </select>
                  </label>
                  <label style={s.settingLabel}>
                    Images
                    <select value={imgSettings.num_images} onChange={e => setImgSettings(p => ({ ...p, num_images: Number(e.target.value) }))} style={s.select}>
                      {[1, 2, 3, 4].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </label>
                  <label style={s.settingLabel}>
                    Guidance {imgSettings.guidance_scale.toFixed(1)}
                    <input type="range" min="0" max="2" step="0.1" value={imgSettings.guidance_scale} onChange={e => setImgSettings(p => ({ ...p, guidance_scale: Number(e.target.value) }))} style={{ width: '100%' }} />
                  </label>
                </div>
                <button style={{ ...s.generateBtn, background: imageModel.color, opacity: loading || !imagePrompt.trim() ? 0.5 : 1 }} disabled={loading || !imagePrompt.trim()}>
                  {loading ? '⏳ Generating...' : `✨ Generate with ${imageModel.label}`}
                </button>
              </form>
            </div>

            {/* Recent completed images inline */}
            {tasks.filter(t => t.status === 'completed').length > 0 && (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#64748b', letterSpacing: 1, textTransform: 'uppercase' }}>Recent</div>
                  <button style={s.clearBtn} onClick={() => setView('tasks')}>View all tasks →</button>
                </div>
                <div style={s.gallery}>
                  {tasks.filter(t => t.status === 'completed').slice(0, 6).map(task => (
                    <div key={task.id} style={s.galleryCard} onClick={() => setLightbox({ url: task.url, task })} role="button">
                      <img src={task.url} style={s.galleryImg} alt={task.prompt} />
                      <div style={s.galleryMeta}>
                        <div style={s.galleryPrompt}>"{task.prompt}"</div>
                        <div style={s.galleryDetails}>
                          <span style={{ color: task.modelColor }}>● {task.model}</span>
                          {task.seed && <span>🌱 {task.seed}</span>}
                          {task.elapsed && <span>⏱ {task.elapsed}s</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── TASKS ── */}
        {view === 'tasks' && (
          <div style={s.studioLayout}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexShrink: 0 }}>
              <div>
                <div style={s.chatTitle}>Generation Tasks</div>
                <div style={s.chatSub}>{tasks.length} total · {tasks.filter(t => t.status === 'completed').length} completed · {tasks.filter(t => t.status === 'generating').length} in progress</div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button style={s.clearBtn} onClick={() => setView('studio')}>+ New</button>
                {tasks.length > 0 && <button style={{ ...s.clearBtn, color: '#f87171', borderColor: '#f8717144' }} onClick={clearAllTasks}>Clear all</button>}
              </div>
            </div>

            {/* Filter tabs */}
            <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
              {['all', 'completed', 'generating', 'failed'].map(f => (
                <button key={f} onClick={() => setTaskFilter(f)}
                  style={{ ...s.filterTab, background: taskFilter === f ? '#1e293b' : 'transparent', color: taskFilter === f ? '#f1f5f9' : '#475569', borderColor: taskFilter === f ? '#334155' : 'transparent' }}>
                  {f === 'all' ? `All (${tasks.length})` : f === 'completed' ? `✅ Completed (${tasks.filter(t => t.status === 'completed').length})` : f === 'generating' ? `⏳ In Progress (${tasks.filter(t => t.status === 'generating').length})` : `❌ Failed (${tasks.filter(t => t.status === 'failed').length})`}
                </button>
              ))}
            </div>

            {filteredTasks.length === 0 ? (
              <div style={s.emptyState}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🗂️</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#475569' }}>No tasks yet</div>
                <div style={{ fontSize: 13, color: '#334155', marginTop: 6 }}>Generate images in the Studio to see them here</div>
                <button style={{ ...s.generateBtn, background: '#8b5cf6', marginTop: 20, padding: '10px 24px', fontSize: 14 }} onClick={() => setView('studio')}>Open Studio</button>
              </div>
            ) : (
              <div style={s.taskList}>
                {filteredTasks.map(task => (
                  <div key={task.id} style={s.taskCard}>
                    {/* Thumbnail */}
                    <div style={{ ...s.taskThumb, background: task.status === 'completed' ? 'transparent' : '#0f172a', cursor: task.status === 'completed' ? 'zoom-in' : 'default' }}
                      onClick={() => task.status === 'completed' && setLightbox({ url: task.url, task })}>
                      {task.status === 'completed' && <img src={task.url} style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 10 }} alt="" />}
                      {task.status === 'generating' && <div style={s.taskSpinner}><span className="typing"><span /><span /><span /></span></div>}
                      {task.status === 'failed'     && <div style={s.taskSpinner}>❌</div>}
                    </div>

                    {/* Info */}
                    <div style={s.taskInfo}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                        <div style={s.taskPrompt}>"{task.prompt}"</div>
                        <div style={{ ...s.statusBadge, background: STATUS_META[task.status].color + '22', color: STATUS_META[task.status].color }}>
                          {STATUS_META[task.status].label}
                        </div>
                      </div>
                      {task.negPrompt && <div style={s.taskNeg}>− {task.negPrompt}</div>}
                      <div style={s.taskMeta}>
                        <MetaChip icon="🤖" label={task.model} />
                        <MetaChip icon="📐" label={task.aspect_ratio.replace(/_/g, ' ')} />
                        <MetaChip icon="🖼" label={`${task.num_images} img`} />
                        <MetaChip icon="🎚" label={`guidance ${task.guidance_scale}`} />
                        {task.seed    && <MetaChip icon="🌱" label={`seed ${task.seed}`} />}
                        {task.elapsed && <MetaChip icon="⏱" label={`${task.elapsed}s`} />}
                        <MetaChip icon="🕐" label={new Date(task.ts).toLocaleString()} />
                      </div>
                      {task.status === 'failed' && task.error && (
                        <div style={{ fontSize: 12, color: '#f87171', marginTop: 6, background: '#f8717111', padding: '6px 10px', borderRadius: 6 }}>⚠️ {task.error}</div>
                      )}
                    </div>

                    {/* Actions */}
                    <div style={s.taskActions}>
                      {task.status === 'completed' && (
                        <>
                          <button style={s.actionBtn} onClick={() => setLightbox({ url: task.url, task })} title="View full size">🔍</button>
                          <a href={task.url} download={`ai-studio-${task.id}.jpg`} style={{ ...s.actionBtn, textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }} title="Download">⬇️</a>
                        </>
                      )}
                      {task.status === 'failed' && (
                        <button style={s.actionBtn} title="Retry" onClick={() => { setImagePrompt(task.prompt); setNegPrompt(task.negPrompt || ''); setView('studio'); }}>🔄</button>
                      )}
                      <button style={{ ...s.actionBtn, color: '#f87171' }} onClick={() => deleteTask(task.id)} title="Delete">🗑</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Lightbox */}
      {lightbox && (
        <div style={s.lightboxOverlay} onClick={() => setLightbox(null)}>
          <div style={s.lightboxBox} onClick={e => e.stopPropagation()}>
            <button style={s.lightboxClose} onClick={() => setLightbox(null)}>✕</button>
            <img src={lightbox.url} style={s.lightboxImg} alt={lightbox.task.prompt} />
            <div style={s.lightboxMeta}>
              <div style={s.galleryPrompt}>"{lightbox.task.prompt}"</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                <MetaChip icon="🤖" label={lightbox.task.model} />
                <MetaChip icon="📐" label={lightbox.task.aspect_ratio.replace(/_/g, ' ')} />
                {lightbox.task.seed    && <MetaChip icon="🌱" label={`seed ${lightbox.task.seed}`} />}
                {lightbox.task.elapsed && <MetaChip icon="⏱" label={`${lightbox.task.elapsed}s`} />}
                <MetaChip icon="🕐" label={new Date(lightbox.task.ts).toLocaleString()} />
              </div>
              <a href={lightbox.url} download={`ai-studio-${lightbox.task.id}.jpg`}
                style={{ display: 'inline-block', marginTop: 14, background: '#2563eb', color: '#fff', padding: '8px 20px', borderRadius: 8, textDecoration: 'none', fontSize: 13, fontWeight: 600 }}>
                ⬇️ Download
              </a>
            </div>
          </div>
        </div>
      )}

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #020617; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        @keyframes blink { 0%,80%,100%{opacity:0} 40%{opacity:1} }
        .typing span { display:inline-block; width:7px; height:7px; background:#94a3b8; border-radius:50%; margin:0 2px; animation:blink 1.4s infinite; }
        .typing span:nth-child(2){animation-delay:.2s}
        .typing span:nth-child(3){animation-delay:.4s}
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

// ── Helper Components ──────────────────────────────────────────────────────────

function NavBtn({ icon, label, active, onClick, color, badge }) {
  return (
    <button onClick={onClick} style={{ position: 'relative', background: active ? color + '22' : 'none', border: active ? `1px solid ${color}44` : '1px solid transparent', borderRadius: 10, textAlign: 'left', padding: '10px 14px', cursor: 'pointer', fontSize: 14, width: '100%', color: active ? color : '#64748b', display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4, transition: 'all .2s' }}>
      <span>{icon}</span>
      <span style={{ fontWeight: active ? 700 : 400 }}>{label}</span>
      {badge > 0 && <span style={{ marginLeft: 'auto', background: '#f59e0b', color: '#000', borderRadius: 20, fontSize: 10, fontWeight: 800, padding: '1px 6px' }}>{badge}</span>}
    </button>
  );
}

function MetaChip({ icon, label }) {
  return (
    <span style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, padding: '2px 8px', fontSize: 11, color: '#64748b', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {icon} {label}
    </span>
  );
}

function MarkdownMessage({ content }) {
  const [copied, setCopied] = useState({});
  const copy = (code, id) => {
    navigator.clipboard.writeText(code);
    setCopied(p => ({ ...p, [id]: true }));
    setTimeout(() => setCopied(p => ({ ...p, [id]: false })), 2000);
  };
  return (
    <div style={md.root}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 style={md.h1}>{children}</h1>,
          h2: ({ children }) => <h2 style={md.h2}>{children}</h2>,
          h3: ({ children }) => <h3 style={md.h3}>{children}</h3>,
          p:  ({ children }) => <p  style={md.p}>{children}</p>,
          strong: ({ children }) => <strong style={md.strong}>{children}</strong>,
          em:     ({ children }) => <em style={md.em}>{children}</em>,
          code({ inline, className, children, ...props }) {
            const lang    = /language-(\w+)/.exec(className || '')?.[1] || '';
            const codeStr = String(children).replace(/\n$/, '');
            const id      = codeStr.slice(0, 20);
            if (inline) return <code style={md.inlineCode}>{children}</code>;
            return (
              <div style={md.codeBlock}>
                <div style={md.codeHeader}>
                  <span style={md.codeLang}>{lang || 'code'}</span>
                  <button style={{ ...md.copyBtn, color: copied[id] ? '#4ade80' : '#94a3b8' }} onClick={() => copy(codeStr, id)}>
                    {copied[id] ? '✓ Copied' : 'Copy'}
                  </button>
                </div>
                <SyntaxHighlighter style={oneDark} language={lang || 'text'} PreTag="div"
                  customStyle={{ margin: 0, borderRadius: '0 0 10px 10px', fontSize: 13, background: '#0d1117' }} {...props}>
                  {codeStr}
                </SyntaxHighlighter>
              </div>
            );
          },
          blockquote: ({ children }) => <blockquote style={md.blockquote}>{children}</blockquote>,
          ul: ({ children }) => <ul style={md.ul}>{children}</ul>,
          ol: ({ children }) => <ol style={md.ol}>{children}</ol>,
          li: ({ children }) => <li style={md.li}>{children}</li>,
          hr: () => <hr style={md.hr} />,
          a:  ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" style={md.a}>{children}</a>,
          table: ({ children }) => <div style={md.tableWrap}><table style={md.table}>{children}</table></div>,
          th: ({ children }) => <th style={md.th}>{children}</th>,
          td: ({ children }) => <td style={md.td}>{children}</td>,
          tr: ({ children }) => <tr style={md.tr}>{children}</tr>,
        }}
      >{content}</ReactMarkdown>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const s = {
  app:          { display: 'flex', height: '100vh', background: '#020617', color: '#f8fafc', fontFamily: "'Inter', system-ui, sans-serif", overflow: 'hidden' },
  sidebar:      { width: 220, background: '#0a0f1e', borderRight: '1px solid #1e293b', padding: '24px 14px', display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0 },
  brand:        { display: 'flex', alignItems: 'center', gap: 10, fontSize: 18, fontWeight: 800, color: '#3b82f6', marginBottom: 24, letterSpacing: '-0.5px' },
  navSection:   { fontSize: 10, color: '#334155', fontWeight: 700, letterSpacing: 1.5, padding: '8px 4px 4px', marginBottom: 4 },
  sidebarFooter:{ background: '#0f172a', borderRadius: 12, padding: 14, border: '1px solid #1e293b', marginTop: 8 },
  footerLabel:  { fontSize: 10, color: '#475569', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 },
  footerModel:  { fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 6 },
  badge:        { display: 'inline-block', fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 20, color: '#fff', letterSpacing: 0.5 },
  workspace:    { flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' },

  // Chat
  chatLayout:   { display: 'flex', flexDirection: 'column', height: '100%', padding: '24px 32px' },
  chatHeader:   { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexShrink: 0 },
  chatTitle:    { fontSize: 22, fontWeight: 800, color: '#f1f5f9' },
  chatSub:      { fontSize: 12, color: '#475569', marginTop: 2 },
  chatArea:     { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16, paddingRight: 8 },
  roleLabel:    { fontSize: 11, color: '#475569', fontWeight: 600, paddingLeft: 4 },
  bubble:       { padding: '12px 18px', maxWidth: '72%', fontSize: 14, lineHeight: 1.6, color: '#e2e8f0', wordBreak: 'break-word' },
  metaBar:      { display: 'flex', flexWrap: 'wrap', gap: 6, paddingLeft: 4, marginTop: 2 },
  inputBar:     { display: 'flex', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 14, padding: '8px 8px 8px 16px', marginTop: 16, flexShrink: 0, alignItems: 'center', gap: 8 },
  input:        { flex: 1, background: 'transparent', border: 'none', color: '#f1f5f9', outline: 'none', fontSize: 14, resize: 'none' },
  sendBtn:      { background: '#2563eb', color: '#fff', border: 'none', width: 40, height: 40, borderRadius: 10, cursor: 'pointer', fontSize: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'opacity .2s' },
  clearBtn:     { background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 8, padding: '6px 14px', cursor: 'pointer', fontSize: 13 },
  modelBtn:     { background: '#0f172a', color: '#e2e8f0', border: '1px solid #1e293b', borderRadius: 10, padding: '8px 14px', cursor: 'pointer', fontSize: 13, display: 'flex', alignItems: 'center', fontWeight: 600 },
  dropdown:     { position: 'absolute', top: '110%', right: 0, background: '#0f172a', border: '1px solid #1e293b', borderRadius: 14, padding: 8, zIndex: 100, minWidth: 280, boxShadow: '0 20px 60px rgba(0,0,0,.6)' },
  dropdownTitle:{ fontSize: 11, color: '#475569', fontWeight: 700, letterSpacing: 1, padding: '6px 10px 10px', textTransform: 'uppercase' },
  dropdownItem: { width: '100%', border: 'none', borderRadius: 10, padding: '10px 12px', cursor: 'pointer', color: '#94a3b8', textAlign: 'left', marginBottom: 2, transition: 'background .15s' },
  dropdownMeta: { fontSize: 11, color: '#475569', marginTop: 3 },

  // Studio
  studioLayout:   { display: 'flex', flexDirection: 'column', height: '100%', padding: '24px 32px', overflowY: 'auto', gap: 20 },
  studioHeader:   { flexShrink: 0 },
  modelTabs:      { display: 'flex', gap: 12, flexShrink: 0 },
  modelTab:       { flex: 1, background: '#0f172a', border: '2px solid transparent', borderRadius: 14, padding: '14px 18px', cursor: 'pointer', textAlign: 'left', transition: 'all .2s' },
  studioControls: { background: '#0a0f1e', border: '1px solid #1e293b', borderRadius: 16, padding: 20, flexShrink: 0 },
  textarea:       { width: '100%', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, color: '#f1f5f9', padding: '12px 14px', fontSize: 14, outline: 'none', resize: 'vertical', fontFamily: 'inherit', lineHeight: 1.6 },
  settingsRow:    { display: 'flex', gap: 16, flexWrap: 'wrap' },
  settingLabel:   { display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, color: '#64748b', fontWeight: 600, flex: 1, minWidth: 120 },
  select:         { background: '#0f172a', border: '1px solid #1e293b', color: '#e2e8f0', borderRadius: 8, padding: '8px 10px', fontSize: 13, outline: 'none' },
  generateBtn:    { color: '#fff', border: 'none', borderRadius: 12, padding: '14px', cursor: 'pointer', fontSize: 15, fontWeight: 700, transition: 'opacity .2s' },
  gallery:        { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 16 },
  galleryCard:    { background: '#0a0f1e', border: '1px solid #1e293b', borderRadius: 16, overflow: 'hidden', cursor: 'zoom-in', transition: 'border-color .2s' },
  galleryImg:     { width: '100%', display: 'block' },
  galleryMeta:    { padding: '12px 14px' },
  galleryPrompt:  { fontSize: 12, color: '#94a3b8', fontStyle: 'italic', marginBottom: 8, lineHeight: 1.5 },
  galleryDetails: { display: 'flex', gap: 10, fontSize: 11, color: '#475569', flexWrap: 'wrap' },

  // Tasks
  filterTab:    { background: 'transparent', border: '1px solid transparent', borderRadius: 8, padding: '6px 14px', cursor: 'pointer', fontSize: 12, fontWeight: 600, transition: 'all .15s' },
  taskList:     { display: 'flex', flexDirection: 'column', gap: 12 },
  taskCard:     { display: 'flex', gap: 16, background: '#0a0f1e', border: '1px solid #1e293b', borderRadius: 16, padding: 16, alignItems: 'flex-start' },
  taskThumb:    { width: 100, height: 100, borderRadius: 10, flexShrink: 0, overflow: 'hidden', border: '1px solid #1e293b' },
  taskSpinner:  { width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 },
  taskInfo:     { flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 8 },
  taskPrompt:   { fontSize: 14, color: '#e2e8f0', fontWeight: 600, lineHeight: 1.4 },
  taskNeg:      { fontSize: 12, color: '#475569', fontStyle: 'italic' },
  taskMeta:     { display: 'flex', flexWrap: 'wrap', gap: 6 },
  taskActions:  { display: 'flex', flexDirection: 'column', gap: 6, flexShrink: 0 },
  actionBtn:    { background: '#1e293b', border: '1px solid #334155', borderRadius: 8, width: 34, height: 34, cursor: 'pointer', fontSize: 15, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  statusBadge:  { fontSize: 11, fontWeight: 700, padding: '3px 10px', borderRadius: 20, flexShrink: 0 },
  emptyState:   { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, padding: 60, textAlign: 'center' },

  // Lightbox
  lightboxOverlay: { position: 'fixed', inset: 0, background: 'rgba(0,0,0,.85)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 },
  lightboxBox:     { background: '#0a0f1e', border: '1px solid #1e293b', borderRadius: 20, overflow: 'hidden', maxWidth: 800, width: '100%', position: 'relative' },
  lightboxClose:   { position: 'absolute', top: 12, right: 12, background: '#1e293b', border: 'none', color: '#94a3b8', width: 32, height: 32, borderRadius: 8, cursor: 'pointer', fontSize: 16, zIndex: 1 },
  lightboxImg:     { width: '100%', display: 'block', maxHeight: '70vh', objectFit: 'contain', background: '#000' },
  lightboxMeta:    { padding: '16px 20px' },
};

const md = {
  root:       { fontSize: 14, lineHeight: 1.7, color: '#e2e8f0' },
  h1:         { fontSize: 20, fontWeight: 800, color: '#f1f5f9', margin: '16px 0 8px', borderBottom: '1px solid #1e293b', paddingBottom: 6 },
  h2:         { fontSize: 17, fontWeight: 700, color: '#f1f5f9', margin: '14px 0 6px' },
  h3:         { fontSize: 15, fontWeight: 700, color: '#cbd5e1', margin: '12px 0 4px' },
  p:          { margin: '6px 0' },
  strong:     { color: '#f1f5f9', fontWeight: 700 },
  em:         { color: '#94a3b8', fontStyle: 'italic' },
  inlineCode: { background: '#0d1117', color: '#7dd3fc', padding: '2px 6px', borderRadius: 5, fontSize: 13, fontFamily: 'monospace', border: '1px solid #1e293b' },
  codeBlock:  { borderRadius: 10, overflow: 'hidden', border: '1px solid #1e293b', margin: '10px 0' },
  codeHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#161b22', padding: '6px 14px', borderBottom: '1px solid #1e293b' },
  codeLang:   { fontSize: 11, color: '#475569', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1 },
  copyBtn:    { background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600, transition: 'color .2s' },
  blockquote: { borderLeft: '3px solid #3b82f6', paddingLeft: 14, margin: '8px 0', color: '#94a3b8', fontStyle: 'italic' },
  ul:         { paddingLeft: 20, margin: '6px 0', display: 'flex', flexDirection: 'column', gap: 3 },
  ol:         { paddingLeft: 20, margin: '6px 0', display: 'flex', flexDirection: 'column', gap: 3 },
  li:         { color: '#cbd5e1' },
  hr:         { border: 'none', borderTop: '1px solid #1e293b', margin: '12px 0' },
  a:          { color: '#60a5fa', textDecoration: 'underline' },
  tableWrap:  { overflowX: 'auto', margin: '10px 0' },
  table:      { borderCollapse: 'collapse', width: '100%', fontSize: 13 },
  th:         { background: '#0f172a', color: '#94a3b8', padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #1e293b', fontWeight: 700 },
  td:         { padding: '7px 12px', borderBottom: '1px solid #0f172a', color: '#cbd5e1' },
  tr:         { transition: 'background .15s' },
};
