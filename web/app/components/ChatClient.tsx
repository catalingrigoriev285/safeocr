"use client";

import dynamic from "next/dynamic";

// Chats live in localStorage, so the chat UI renders only in the browser (no SSR / hydration mismatch).
const Chat = dynamic(() => import("./Chat"), {
  ssr: false,
  loading: () => <div className="h-dvh bg-[var(--background)]" />,
});

export default function ChatClient() {
  return <Chat />;
}
