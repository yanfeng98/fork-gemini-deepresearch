import { useState } from "react";
import ChatWindow from "./components/ChatWindow";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!input) return;

    setMessages(prev => [...prev, { role: "user", content: input }, { role: "ai", content: "" }]);
    setInput("");

    const res = await fetch("/api/chat-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input }),
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
        copy[copy.length - 1].content = aiText;
        return copy;
      });
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "0 auto" }}>
      <h1 style={{ textAlign: "center" }}>🤖 AI Chat</h1>

      <ChatWindow messages={messages} />

      <div style={{ display: "flex", gap: 8, padding: 16 }}>
        <input
          style={{ flex: 1, padding: 10 }}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="输入你的问题"
        />
        <button onClick={sendMessage}>发送</button>
      </div>
    </div>
  );
}

export default App;