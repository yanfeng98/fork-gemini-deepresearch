// App.jsx
import { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    const userMsg = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg, { role: "ai", content: "" }]);
    setInput("");

    const res = await fetch("/api/chat-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMsg.content }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let aiText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      aiText += decoder.decode(value);

      setMessages(prev => {
        const copy = [...prev];
        copy[copy.length - 1] = { role: "ai", content: aiText };
        return copy;
      });
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h1>🤖 Streaming LangGraph Chat</h1>

      {messages.map((m, i) => (
        <p key={i}><b>{m.role}:</b> {m.content}</p>
      ))}

      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={sendMessage}>发送</button>
    </div>
  );
}

export default App;