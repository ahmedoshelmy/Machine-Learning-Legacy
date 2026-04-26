import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function Home() {
  const [view, setView] = useState('chat');
  const [messages, setMessages] = useState([{ role: 'assistant', content: 'Llama 3.1 8B is ready. How can I help?' }]);
  const [chatInput, setChatInput] = useState('');
  const [imagePrompt, setImagePrompt] = useState('');
  const [gallery, setGallery] = useState([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || loading) return;
    const newMsg = [...messages, { role: 'user', content: chatInput }];
    setMessages(newMsg);
    setChatInput('');
    setLoading(true);
    try {
      const res = await axios.post('/api/chat', { mode: 'chat', messages: newMsg });
      setMessages([...newMsg, res.data.choices[0].message]);
    } catch (err) {
      setMessages([...newMsg, { role: 'assistant', content: '⚠️ Groq connection failed.' }]);
    } finally { setLoading(false); }
  };

  const handleImage = async (e) => {
    e.preventDefault();
    if (!imagePrompt.trim() || loading) return;
    setLoading(true);
    try {
      const res = await axios.post('/api/chat', { mode: 'image', prompt: imagePrompt });
      if (res.data.url) {
        setGallery([res.data.url, ...gallery]);
        setImagePrompt('');
      }
    } catch (err) {
      alert("Error: " + (err.response?.data?.error || err.message));
    } finally { setLoading(false); }
  };

  return (
    <div style={styles.app}>
      <nav style={styles.sidebar}>
        <div style={styles.brand}>AI STUDIO 2026</div>
        <button onClick={() => setView('chat')} style={{...styles.navBtn, color: view === 'chat' ? '#3b82f6' : '#94a3b8'}}>💬 Chat</button>
        <button onClick={() => setView('studio')} style={{...styles.navBtn, color: view === 'studio' ? '#8b5cf6' : '#94a3b8'}}>🎨 Studio</button>
      </nav>

      <main style={styles.workspace}>
        {view === 'chat' ? (
          <div style={styles.container}>
            <div style={styles.chatArea}>
              {messages.map((m, i) => (
                <div key={i} style={{...styles.bubble, alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', background: m.role === 'user' ? '#2563eb' : '#1e293b'}}>
                  {m.content}
                </div>
              ))}
              <div ref={scrollRef} />
            </div>
            <form onSubmit={handleChat} style={styles.bar}>
              <input value={chatInput} onChange={e => setChatInput(e.target.value)} placeholder="Type a message..." style={styles.input} />
              <button style={styles.btn} disabled={loading}>{loading ? '...' : 'Send'}</button>
            </form>
          </div>
        ) : (
          <div style={styles.container}>
            <h2>Freepik Image Studio</h2>
            <form onSubmit={handleImage} style={styles.bar}>
              <input value={imagePrompt} onChange={e => setImagePrompt(e.target.value)} placeholder="A cyberpunk cat in Tokyo..." style={styles.input} />
              <button style={{...styles.btn, background: '#7c3aed'}} disabled={loading}>
                {loading ? 'Processing...' : 'Generate'}
              </button>
            </form>
            <div style={styles.grid}>
              {gallery.map((url, i) => <img key={i} src={url} style={styles.img} alt="AI Art" />)}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

const styles = {
  app: { display: 'flex', height: '100vh', background: '#020617', color: '#f8fafc', fontFamily: 'Inter, sans-serif' },
  sidebar: { width: '220px', background: '#0f172a', padding: '30px 15px', borderRight: '1px solid #1e293b' },
  brand: { fontSize: '20px', fontWeight: 'bold', marginBottom: '30px', color: '#3b82f6' },
  navBtn: { background: 'none', border: 'none', textAlign: 'left', padding: '12px', cursor: 'pointer', fontSize: '16px', width: '100%' },
  workspace: { flex: 1, padding: '40px', overflowY: 'auto' },
  container: { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', height: '100%' },
  chatArea: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '15px', marginBottom: '20px' },
  bubble: { padding: '12px 18px', borderRadius: '15px', maxWidth: '75%', fontSize: '14px' },
  bar: { display: 'flex', background: '#1e293b', padding: '10px', borderRadius: '12px', border: '1px solid #334155' },
  input: { flex: 1, background: 'transparent', border: 'none', color: '#fff', outline: 'none', padding: '10px' },
  btn: { background: '#2563eb', color: '#fff', border: 'none', padding: '0 20px', borderRadius: '8px', cursor: 'pointer' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px', marginTop: '30px' },
  img: { width: '100%', borderRadius: '12px', border: '1px solid #334155' }
};