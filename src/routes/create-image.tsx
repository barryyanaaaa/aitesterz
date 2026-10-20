import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { ArrowLeft, Download, ImageIcon, Loader2, Sparkles } from "lucide-react";
import { streamImage } from "@/lib/stream-image";

export const Route = createFileRoute("/create-image")({
  head: () => ({
    meta: [
      { title: "Image Studio — EvolveAI by mdot" },
      { name: "description", content: "Describe any image and EvolveAI creates it for you to download." },
      { property: "og:title", content: "Image Studio — EvolveAI by mdot" },
      { property: "og:description", content: "Describe any image and EvolveAI creates it for you to download." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: CreateImage,
});

const IDEAS = ["A neon-lit Tokyo street in the rain, cinematic", "Cozy cabin in snowy mountains at dusk, oil painting", "Minimal logo of a fox made of geometric shapes"];

function friendlyError(message: string) {
  if (message.includes(" 402")) return "You're out of AI credits. Add more in your workspace settings to keep creating.";
  if (message.includes(" 429")) return "Too many requests right now. Wait a moment and try again.";
  if (message.includes(" 403")) return "Image creation isn't available for this workspace right now.";
  return "The image couldn't be created. Try again or change your description.";
}

function CreateImage() {
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [final, setFinal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const generate = async (e?: FormEvent) => {
    e?.preventDefault();
    const text = prompt.trim();
    if (!text || busy) return;
    setBusy(true); setError(""); setImage(null); setFinal(false);
    try {
      await streamImage("/api/generate-image", { prompt: text }, (url, isFinal) => { setImage(url); setFinal(isFinal); });
    } catch (err) {
      console.error(err);
      setError(friendlyError(err instanceof Error ? err.message : String(err)));
    } finally {
      setBusy(false);
    }
  };

  const download = () => {
    if (!image) return;
    const a = document.createElement("a");
    a.href = image;
    a.download = `evolveai-${Date.now()}.png`;
    document.body.appendChild(a); a.click(); a.remove();
  };

  return (
    <div className="app-shell thin-grid min-h-screen text-foreground">
      <div className="mx-auto max-w-3xl px-5 py-8">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"><ArrowLeft size={15} /> Back to chat</Link>
        <div className="mt-8 flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl border border-border bg-card text-primary"><ImageIcon size={18} /></div>
          <div>
            <div className="mono text-[10px] uppercase tracking-[.14em] text-muted-foreground">EvolveAI · Image Studio</div>
            <h1 className="text-3xl font-bold tracking-tight">Describe it. <span className="text-primary">See it.</span></h1>
          </div>
        </div>

        <form onSubmit={generate} className="mt-8 rounded-2xl border border-border bg-card p-4">
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} placeholder="A watercolor of a lighthouse at sunrise…"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void generate(); } }}
            className="w-full resize-none bg-transparent text-[15px] outline-none placeholder:text-muted-foreground" aria-label="Describe your image" />
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {IDEAS.map((idea) => (
              <button type="button" key={idea} onClick={() => setPrompt(idea)} className="rounded-full border border-border px-3 py-1 text-[11px] text-muted-foreground hover:text-foreground">{idea}</button>
            ))}
            <button type="submit" disabled={busy || !prompt.trim()} className="ml-auto inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:opacity-50">
              {busy ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />} {busy ? "Creating…" : "Create image"}
            </button>
          </div>
        </form>

        {error && <p className="mt-4 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</p>}

        <div className="mt-6 overflow-hidden rounded-2xl border border-border bg-card">
          {image ? (
            <img src={image} alt={prompt} className={`w-full transition-[filter] duration-700 ${final ? "blur-0" : "blur-2xl"}`} />
          ) : (
            <div className="grid aspect-square place-items-center text-sm text-muted-foreground">
              {busy ? <span className="flex items-center gap-2"><Loader2 size={16} className="animate-spin" /> Painting your image…</span> : "Your image will appear here"}
            </div>
          )}
        </div>
        {image && final && (
          <button onClick={download} className="mt-4 inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2 text-sm font-semibold hover:bg-accent">
            <Download size={15} /> Download image
          </button>
        )}
      </div>
    </div>
  );
}
