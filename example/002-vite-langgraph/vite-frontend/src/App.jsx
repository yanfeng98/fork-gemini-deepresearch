import { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!input) return;

    const userMsg = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input }),
    });

    const data = await res.json();

    setMessages(prev => [...prev, { role: "ai", content: data.reply }]);
  };

  return (
    <div style={{ padding: 24 }}>
      <h1>🤖 LangGraph Chat</h1>

      <div style={{ minHeight: 200 }}>
        {messages.map((m, i) => (
          <p key={i}><b>{m.role}:</b> {m.content}</p>
        ))}
      </div>

      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        placeholder="输入你的问题"
      />
      <button onClick={sendMessage}>发送</button>
    </div>
  );
}

export default App;