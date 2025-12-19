import { useEffect, useRef } from "react";
import ChatBubble from "./ChatBubble";

export default function ChatWindow({ messages }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div
      style={{
        height: "70vh",
        overflowY: "auto",
        padding: 16,
        background: "#fafafa",
      }}
    >
      {messages.map((m, i) => (
        <ChatBubble key={i} role={m.role} content={m.content} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}