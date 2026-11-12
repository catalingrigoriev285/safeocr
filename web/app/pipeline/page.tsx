import type { Metadata } from "next";
import Link from "next/link";
import Anonymizer from "../components/Anonymizer";

export const metadata: Metadata = { title: "SafeOCR · Pipeline" };

// The original step-by-step demo page, kept as a copy (linked from the chat sidebar).
export default function PipelinePage() {
  return (
    <>
      <div className="mx-auto w-full max-w-7xl px-4 pt-4 sm:px-6">
        <Link href="/" className="text-sm text-zinc-500 transition hover:text-zinc-900 dark:hover:text-zinc-100">
          ← Back to chat
        </Link>
      </div>
      <Anonymizer />
    </>
  );
}
