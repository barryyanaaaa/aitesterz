import { type FormEvent, type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { env, pipeline, TextStreamer } from '@huggingface/transformers';
import { Link } from '@tanstack/react-router';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  AlertCircle,
  ArrowUp,
  Check,
  ChevronDown,
  CircleHelp,
  Cpu,
  Download,
  FileText,
  HardDrive,
  Image as ImageIcon,
  Menu,
  MoreHorizontal,
  Plus,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  X,
  Zap,
} from 'lucide-react';

type ModelMeta = { id: string; label: string; webgpuSize: string; wasmSize: string; capability: string; detail: string };
type ChatMessage = { id: string; role: 'user' | 'assistant'; content: string; timestamp: number };
type Conversation = { id: string; title: string; createdAt: number; messages: ChatMessage[] };
type RuntimeStatus = 'idle' | 'loading' | 'ready' | 'error';
type Progress = { file?: string; progress?: number; loaded?: number; total?: number };

env.useBrowserCache = true;
// Always fetch models from the Hugging Face Hub automatically (never look for local files).
env.allowLocalModels = false;
env.allowRemoteModels = true;

const MODELS: ModelMeta[] = [
  { id: 'onnx-community/SmolLM2-360M-Instruct-ONNX', label: 'SmolLM2 · 360M', webgpuSize: '~260 MB', wasmSize: '~369 MB', capability: 'Quick replies', detail: 'Fastest · great for everyday questions' },
  { id: 'onnx-community/Qwen2.5-0.5B-Instruct', label: 'Qwen 2.5 · 0.5B', webgpuSize: '~461 MB', wasmSize: '~750 MB', capability: 'Reasoning', detail: 'More capable · larger download' },
  { id: 'onnx-community/SmolLM2-135M-Instruct-ONNX', label: 'SmolLM2 · 135M', webgpuSize: '~112 MB', wasmSize: '~172 MB', capability: 'Lightweight', detail: 'Smallest footprint · best for older devices' },
];

const STORAGE_KEY = 'local-ai-chat.conversations.v1';
const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
const formatTime = (time: number) => new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(time);

function readConversations(): Conversation[] {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return [];
    const parsed = JSON.parse(saved) as Conversation[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function bytesLabel(bytes?: number) {
  if (!bytes) return 'Preparing model files…';
  return `${(bytes / 1024 / 1024).toFixed(1)} MB downloaded`;
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>(readConversations);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState(MODELS[0]!.id);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>('idle');
  const [loadProgress, setLoadProgress] = useState<Progress>({});
  const [runtimeError, setRuntimeError] = useState('');
  const [backend, setBackend] = useState<'webgpu' | 'wasm' | null>(null);
  const [draft, setDraft] = useState('');
  const [generating, setGenerating] = useState(false);
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const generationAbort = useRef<AbortController | null>(null);
  const runtimeRef = useRef<any>(null);
  const runtimeModelId = useRef<string | null>(null);
  const loadPromiseRef = useRef<{ modelId: string; promise: Promise<any> } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const draftRef = useRef<HTMLTextAreaElement>(null);

  const selectedModel = MODELS.find((model) => model.id === selectedModelId) ?? MODELS[0]!;
  const activeConversation = conversations.find((chat) => chat.id === activeId) ?? null;
  const isLoaded = runtimeStatus === 'ready' && runtimeModelId.current === selectedModelId;
  const loadingPercent = typeof loadProgress.progress === 'number' ? Math.max(0, Math.min(100, loadProgress.progress)) : null;
  const storedMessageCount = useMemo(() => conversations.reduce((sum, chat) => sum + chat.messages.length, 0), [conversations]);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations)); } catch { /* Storage can be unavailable in private browser contexts. */ }
  }, [conversations]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [activeConversation?.messages.length, activeConversation?.messages.at(-1)?.content]);

  useEffect(() => () => { generationAbort.current?.abort(); }, []);

  const updateConversation = useCallback((id: string, updater: (chat: Conversation) => Conversation) => {
    setConversations((current) => current.map((chat) => chat.id === id ? updater(chat) : chat));
  }, []);

  const createConversation = useCallback(() => {
    const conversation: Conversation = { id: newId(), title: 'New conversation', createdAt: Date.now(), messages: [] };
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    setMobileSidebar(false);
    window.setTimeout(() => draftRef.current?.focus(), 80);
    return conversation;
  }, []);

  const loadModel = useCallback(async (modelId = selectedModelId) => {
    if (runtimeRef.current && runtimeModelId.current === modelId) {
      setRuntimeStatus('ready');
      return runtimeRef.current;
    }
    if (loadPromiseRef.current?.modelId === modelId) return loadPromiseRef.current.promise;

    setRuntimeStatus('loading');
    setRuntimeError('');
    setLoadProgress({});
    setBackend(null);
    const progressCallback = (event: Progress & { status?: string }) => {
      if (event.status === 'progress' || event.status === 'download' || event.progress !== undefined) setLoadProgress(event);
    };
    const load = async (device: 'webgpu' | 'wasm') => pipeline('text-generation', modelId, {
      device,
      dtype: device === 'webgpu' ? 'q4f16' : 'q4',
      progress_callback: progressCallback,
    });
    const promise = (async () => {
      let webgpuAvailable = false;
      try {
        // Only use WebGPU when a real GPU adapter exists, otherwise go straight to WASM.
        const gpu = typeof navigator !== 'undefined' ? (navigator as any).gpu : undefined;
        const adapter = gpu && localStorage.getItem('evolve.no-webgpu') !== '1' ? await gpu.requestAdapter() : null;
        webgpuAvailable = !!(adapter && (await adapter.requestDevice()));
      } catch { webgpuAvailable = false; }
      if (webgpuAvailable) {
        setBackend('webgpu');
        try {
          return await load('webgpu');
        } catch {
          // WebGPU failed on this device. The runtime can't switch backends in the same page,
          // so remember it and reload once — the next load goes straight to WASM using cached files.
          try { localStorage.setItem('evolve.no-webgpu', '1'); window.location.reload(); return new Promise(() => {}); } catch { /* fall through */ }
          setLoadProgress({});
          setBackend('wasm');
        }
      } else {
        setBackend('wasm');
      }
      try {
        return await load('wasm');
      } catch {
        setLoadProgress({});
        // Fallback precision for browsers/devices that reject 4-bit weights.
        return pipeline('text-generation', modelId, { device: 'wasm', dtype: 'q8', progress_callback: progressCallback });
      }
    })();
    loadPromiseRef.current = { modelId, promise };
    try {
      const loadedRuntime = await promise;
      runtimeRef.current = loadedRuntime;
      runtimeModelId.current = modelId;
      setRuntimeStatus('ready');
      return loadedRuntime;
    } catch (error) {
      console.error('Model load failed', error);
      const message = error instanceof Error ? error.message : typeof error === 'string' ? error : 'The model could not be loaded in this browser.';
      setRuntimeError(message);
      setRuntimeStatus('error');
      return null;
    } finally {
      if (loadPromiseRef.current?.promise === promise) loadPromiseRef.current = null;
    }
  }, [selectedModelId]);

  // Automatically prepare the selected model on open (cached after the first visit).
  useEffect(() => {
    if (runtimeModelId.current !== selectedModelId) void loadModel(selectedModelId);
  }, [selectedModelId, loadModel]);

  const selectModel = (id: string) => {
    if (generating || runtimeStatus === 'loading') return;
    setSelectedModelId(id);
    setModelMenuOpen(false);
    setRuntimeError('');
    if (runtimeModelId.current === id) setRuntimeStatus('ready');
    else setRuntimeStatus('idle');
  };

  const sendMessage = useCallback(async (rawText = draft) => {
    const text = rawText.trim();
    if (!text || generating) return;
    let conversation = activeConversation;
    if (!conversation) conversation = createConversation();
    const conversationId = conversation.id;
    const userMessage: ChatMessage = { id: newId(), role: 'user', content: text, timestamp: Date.now() };
    const assistantMessage: ChatMessage = { id: newId(), role: 'assistant', content: '', timestamp: Date.now() };
    updateConversation(conversationId, (chat) => ({
      ...chat,
      title: chat.messages.length === 0 ? text.slice(0, 44).split('\n')[0] || 'New conversation' : chat.title,
      messages: [...chat.messages, userMessage, assistantMessage],
    }));
    setDraft('');
    setGenerating(true);
    setRuntimeError('');
    const activeRuntime = await loadModel();
    if (!activeRuntime) {
      updateConversation(conversationId, (chat) => ({ ...chat, messages: chat.messages.filter((message) => message.id !== assistantMessage.id) }));
      setGenerating(false);
      return;
    }

    const aborter = new AbortController();
    generationAbort.current = aborter;
    try {
      const history = [...conversation.messages, userMessage].map(({ role, content }) => ({ role, content }));
      const tokenizer = (activeRuntime as any).tokenizer;
      const prompt = tokenizer.apply_chat_template(history, { tokenize: false, add_generation_prompt: true });
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        callback_function: (fragment: string) => {
          if (!fragment || aborter.signal.aborted) return;
          updateConversation(conversationId, (chat) => ({
            ...chat,
            messages: chat.messages.map((message) => message.id === assistantMessage.id ? { ...message, content: message.content + fragment } : message),
          }));
        },
      });
      await (activeRuntime as any)(prompt, {
        max_new_tokens: 512,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.08,
        streamer,
        signal: aborter.signal,
      });
    } catch (error) {
      const aborted = aborter.signal.aborted || (error instanceof Error && error.name === 'AbortError');
      if (!aborted) {
        const message = error instanceof Error ? error.message : 'Generation stopped unexpectedly.';
        setRuntimeError(message);
        updateConversation(conversationId, (chat) => ({
          ...chat,
          messages: chat.messages.map((item) => item.id === assistantMessage.id && !item.content ? { ...item, content: 'I couldn’t finish that response. Please try again.' } : item),
        }));
      }
    } finally {
      generationAbort.current = null;
      setGenerating(false);
    }
  }, [activeConversation, createConversation, draft, generating, loadModel, updateConversation]);

  const stopGeneration = () => generationAbort.current?.abort();
  const deleteConversation = (id: string) => {
    if (generating && activeId === id) stopGeneration();
    setConversations((current) => current.filter((chat) => chat.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const renameConversation = (chat: Conversation) => {
    const title = window.prompt('Name this conversation', chat.title);
    if (title?.trim()) updateConversation(chat.id, (current) => ({ ...current, title: title.trim().slice(0, 70) }));
  };

  const copyMessage = async (message: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedId(message.id);
      window.setTimeout(() => setCopiedId(null), 1400);
    } catch {
      setRuntimeError('Clipboard access is unavailable in this browser.');
    }
  };

  const removeCurrent = () => activeId && deleteConversation(activeId);
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void sendMessage();
  };
  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  const statusLabel = runtimeStatus === 'ready' && isLoaded
    ? `Ready · ${backend === 'webgpu' ? 'WebGPU' : 'WASM'}`
    : runtimeStatus === 'loading' ? 'Loading model'
      : runtimeStatus === 'error' ? 'Needs attention' : 'Model not loaded';

  return (
    <div className="app-shell flex h-[100dvh] min-h-0 overflow-hidden text-foreground" data-testid="app-local-ai-chat">
      {mobileSidebar && <button aria-label="Close conversation list" className="fixed inset-0 z-30 bg-black/60 md:hidden" onClick={() => setMobileSidebar(false)} data-testid="button-close-sidebar-overlay" />}
      <aside className={`sidebar-scroll fixed inset-y-0 left-0 z-40 flex w-[286px] flex-col border-r border-white/[.07] bg-[#101714] transition-transform duration-200 md:static md:translate-x-0 ${mobileSidebar ? 'translate-x-0' : '-translate-x-full'}`} data-testid="sidebar-conversations">
        <div className="flex h-[76px] items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-xl border border-[#587b62]/40 bg-[#1c2b21] text-[#a4d4af]"><Sparkles size={17} strokeWidth={1.8} /></div>
            <div>
              <div className="text-[14px] font-extrabold tracking-[-.035em] text-[#e1e9e3]"><span>Evolve</span><span className="text-[#9fd0a8]">AI</span></div>
              <div className="mono mt-0.5 text-[9px] tracking-[.13em] text-[#79887d]">BY MDOT · PRIVATE BY DESIGN</div>
            </div>
          </div>
          <button onClick={() => setMobileSidebar(false)} className="grid size-8 place-items-center rounded-lg text-[#7c8980] hover:bg-white/[.06] md:hidden" aria-label="Close sidebar" data-testid="button-close-sidebar"><X size={16} /></button>
        </div>
        <div className="px-3">
          <button onClick={createConversation} className="group flex h-10 w-full items-center gap-2.5 rounded-xl border border-[#3b5c45]/55 bg-[#1b2a20] px-3 text-left text-[12px] font-semibold text-[#bdd8c2] transition-colors hover:bg-[#223629]" data-testid="button-new-chat">
            <Plus size={15} className="text-[#a4d4af]" /> New conversation <span className="ml-auto mono text-[10px] text-[#728777]">⌘ K</span>
          </button>
          <Link to="/create-image" className="mt-2 flex h-10 w-full items-center gap-2.5 rounded-xl border border-white/[.08] px-3 text-[12px] font-semibold text-[#9fb3a4] transition-colors hover:bg-white/[.04]" data-testid="link-image-studio">
            <ImageIcon size={15} className="text-[#a4d4af]" /> Create an image
          </Link>
        </div>
        <div className="mt-7 flex items-center justify-between px-5">
          <span className="mono text-[9px] font-medium uppercase tracking-[.16em] text-[#6e7d72]">Your conversations</span>
          <span className="mono text-[10px] text-[#59665d]" data-testid="text-chat-count">{conversations.length.toString().padStart(2, '0')}</span>
        </div>
        <div className="sidebar-scroll mt-2 flex-1 overflow-y-auto px-2.5 pb-3">
          {conversations.length === 0 ? (
            <div className="mx-2 mt-3 rounded-xl border border-dashed border-white/[.09] px-3 py-4">
              <p className="text-[11px] leading-5 text-[#78867c]">Your chats live only on this device. Start one whenever you’re ready.</p>
            </div>
          ) : conversations.map((chat) => (
            <div key={chat.id} className={`group mb-1 flex items-center gap-1 rounded-xl px-2 py-2 transition-colors ${activeId === chat.id ? 'bg-[#202d24]' : 'hover:bg-white/[.035]'}`} data-testid={`conversation-row-${chat.id}`}>
              <button onClick={() => { setActiveId(chat.id); setMobileSidebar(false); }} className="min-w-0 flex-1 text-left" data-testid={`button-open-conversation-${chat.id}`}>
                <span className={`block truncate text-[11px] ${activeId === chat.id ? 'font-semibold text-[#d3e3d6]' : 'text-[#9ba79d]'}`}>{chat.title || 'New conversation'}</span>
                <span className="mono mt-1 block text-[9px] text-[#647168]">{chat.messages.length} {chat.messages.length === 1 ? 'message' : 'messages'}</span>
              </button>
              <div className="flex opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                <button onClick={() => renameConversation(chat)} aria-label={`Rename ${chat.title}`} className="grid size-7 place-items-center rounded-md text-[#78877b] hover:bg-white/[.07] hover:text-[#d4e3d7]" data-testid={`button-rename-${chat.id}`}><MoreHorizontal size={14} /></button>
                <button onClick={() => deleteConversation(chat.id)} aria-label={`Delete ${chat.title}`} className="grid size-7 place-items-center rounded-md text-[#78877b] hover:bg-[#3b2524] hover:text-[#f0a29b]" data-testid={`button-delete-${chat.id}`}><Trash2 size={13} /></button>
              </div>
            </div>
          ))}
        </div>
        <div className="border-t border-white/[.07] p-4">
          <div className="flex items-center gap-2.5 rounded-xl bg-[#18211b] p-3">
            <ShieldCheck size={16} className="shrink-0 text-[#91bd9d]" />
            <div className="min-w-0">
              <div className="text-[10px] font-semibold text-[#bdcbbf]">On-device, always</div>
              <div className="mt-1 text-[9px] leading-4 text-[#748177]">No account. No conversation leaves here.</div>
            </div>
          </div>
          <div className="mono mt-3 flex justify-between px-1 text-[9px] text-[#58665d]"><span>LOCAL STORAGE</span><span>{storedMessageCount} MSG</span></div>
        </div>
      </aside>

      <main className="relative flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="relative z-20 flex h-[76px] shrink-0 items-center justify-between border-b border-white/[.07] px-4 sm:px-7">
          <div className="flex min-w-0 items-center gap-3">
            <button className="grid size-9 shrink-0 place-items-center rounded-lg border border-white/[.08] text-[#9aa89d] hover:bg-white/[.05] md:hidden" aria-label="Open conversations" onClick={() => setMobileSidebar(true)} data-testid="button-open-sidebar"><Menu size={17} /></button>
              <div className="hidden size-8 place-items-center rounded-lg bg-[#18231c] text-[#8fb99a] sm:grid"><Sparkles size={15} /></div>
            <div className="min-w-0">
               <div className="truncate text-[12px] font-bold tracking-[-.02em] text-[#dce5de]"><span>Evolve</span><span className="text-[#9fd0a8]">AI</span><span className="ml-1.5 text-[9px] font-medium uppercase tracking-[.12em] text-[#7c8b80]">by mdot</span></div>
              <div className="mt-1 flex items-center gap-1.5 text-[10px] text-[#77847a]"><span className={`size-1.5 rounded-full ${isLoaded ? 'bg-[#93cb9e]' : runtimeStatus === 'error' ? 'bg-[#d98479]' : 'bg-[#7d8a80]'}`} />{statusLabel}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <button onClick={() => setModelMenuOpen((open) => !open)} className="flex h-9 max-w-[190px] items-center gap-2 rounded-lg border border-white/[.09] bg-[#151d18] px-2.5 text-left hover:border-white/[.16] sm:max-w-none" aria-expanded={modelMenuOpen} data-testid="button-model-picker">
                <Cpu size={14} className="shrink-0 text-[#8fac94]" />
                <span className="truncate text-[10px] font-semibold text-[#c2cec4] sm:text-[11px]">{selectedModel.label}</span>
                <ChevronDown size={13} className={`shrink-0 text-[#78857b] transition-transform ${modelMenuOpen ? 'rotate-180' : ''}`} />
              </button>
              {modelMenuOpen && (
                <div className="absolute right-0 top-11 z-50 w-[282px] rounded-xl border border-white/[.1] bg-[#151d18] p-1.5 shadow-[0_18px_50px_rgba(0,0,0,.45)]" data-testid="menu-model-options">
                  <div className="px-2.5 pb-2 pt-1.5 mono text-[9px] uppercase tracking-[.14em] text-[#6f7c72]">Choose a local model</div>
                  {MODELS.map((model) => (
                    <button key={model.id} onClick={() => selectModel(model.id)} className={`flex w-full items-start gap-2.5 rounded-lg p-2.5 text-left hover:bg-white/[.05] ${selectedModelId === model.id ? 'bg-[#1c2920]' : ''}`} data-testid={`option-model-${model.id.split('/').at(-1)}`}>
                      <div className={`mt-0.5 grid size-6 shrink-0 place-items-center rounded-md ${selectedModelId === model.id ? 'bg-[#2a4130] text-[#a8d7b2]' : 'bg-[#212822] text-[#819087]'}`}>{selectedModelId === model.id ? <Check size={13} /> : <Cpu size={13} />}</div>
                      <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between gap-2 text-[11px] font-semibold text-[#d2ded4]"><span>{model.label}</span><span className="mono text-[8px] font-normal text-[#748177]">{model.webgpuSize} / {model.wasmSize}</span></div>
                        <div className="mt-1 text-[10px] text-[#7e8b81]">{model.detail}</div>
                      </div>
                    </button>
                  ))}
                  <div className="mt-1 border-t border-white/[.07] px-2.5 py-2 text-[9px] leading-4 text-[#758279]">Download sizes show WebGPU / WASM. Weights are cached in your browser after download.</div>
                </div>
              )}
            </div>
            {activeConversation && (
              <button onClick={removeCurrent} aria-label="Delete current conversation" className="grid size-9 place-items-center rounded-lg text-[#77847a] hover:bg-[#352320] hover:text-[#e3a198]" data-testid="button-delete-current-chat"><Trash2 size={15} /></button>
            )}
          </div>
        </header>

        {modelMenuOpen && <button className="fixed inset-0 z-10 cursor-default" aria-label="Dismiss model menu" onClick={() => setModelMenuOpen(false)} data-testid="button-dismiss-model-menu" />}

        <section className="message-scroll min-h-0 flex-1 overflow-y-auto" data-testid="conversation-view">
          {activeConversation && activeConversation.messages.length > 0 ? (
            <div className="mx-auto w-full max-w-[790px] px-4 pb-8 pt-9 sm:px-8 sm:pt-12">
              <div className="mb-9 flex items-center gap-3">
                <span className="mono text-[9px] uppercase tracking-[.15em] text-[#6b796f]">Conversation</span>
                <span className="h-px flex-1 bg-white/[.07]" />
                <span className="mono text-[9px] text-[#627066]">{new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(activeConversation.createdAt)}</span>
              </div>
              <div className="space-y-8">
                {activeConversation.messages.map((message, index) => (
                  <article key={message.id} className={`entry ${message.role === 'user' ? 'flex justify-end' : ''}`} data-testid={`message-${message.role}-${message.id}`}>
                    {message.role === 'user' ? (
                      <div className="max-w-[88%] rounded-2xl rounded-br-md border border-[#344a38] bg-[#1e2b22] px-4 py-3 text-[13px] leading-6 text-[#dce6dd] sm:max-w-[76%]">
                        <div className="mb-1.5 flex items-center justify-between gap-4"><span className="mono text-[9px] uppercase tracking-[.1em] text-[#87ac8e]">You</span><time className="mono text-[9px] text-[#748277]">{formatTime(message.timestamp)}</time></div>
                        <div className="whitespace-pre-wrap">{message.content}</div>
                      </div>
                    ) : (
                      <div className="flex max-w-full gap-3.5 sm:gap-4">
                        <div className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg border border-[#46614b]/40 bg-[#1c2a20] text-[#a5d0ac]"><Sparkles size={14} /></div>
                        <div className="min-w-0 flex-1">
                          <div className="mb-2 flex items-center gap-2">
                            <span className="text-[11px] font-bold text-[#cfdbd1]">EvolveAI</span>
                            <span className="mono text-[9px] text-[#657268]">{formatTime(message.timestamp)}</span>
                            {message.content && <button onClick={() => void copyMessage(message)} className="ml-auto rounded-md p-1 text-[#66746a] hover:bg-white/[.06] hover:text-[#bdcabe]" aria-label="Copy assistant response" data-testid={`button-copy-message-${message.id}`}>{copiedId === message.id ? <Check size={13} /> : <FileText size={13} />}</button>}
                          </div>
                          {message.content ? (
                            <div className="message-body prose prose-invert max-w-none text-[13px] leading-7 text-[#b7c3b9]" data-testid={`content-message-${message.id}`}>
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                            </div>
                          ) : generating && index === activeConversation.messages.length - 1 ? (
                            <div className="flex h-8 items-center gap-1.5" aria-label="Assistant is thinking" data-testid="indicator-typing">
                              <span className="typing-dot size-1.5 rounded-full bg-[#95c6a0]" /><span className="typing-dot size-1.5 rounded-full bg-[#95c6a0]" /><span className="typing-dot size-1.5 rounded-full bg-[#95c6a0]" />
                              <span className="ml-2 text-[10px] text-[#79877d]">Thinking locally</span>
                            </div>
                          ) : null}
                        </div>
                      </div>
                    )}
                  </article>
                ))}
              </div>
              <div ref={messagesEndRef} />
            </div>
          ) : (
            <div className="thin-grid flex min-h-full items-center justify-center px-5 py-12">
              <div className="entry w-full max-w-[710px]">
                <div className="mb-8 flex items-center gap-3">
                  <div className="grid size-10 place-items-center rounded-[14px] border border-[#45624b]/45 bg-[#1b2a20] text-[#acd5b3]"><Sparkles size={18} /></div>
                  <div><div className="mono text-[9px] uppercase tracking-[.16em] text-[#8eb496]">EvolveAI · A private space to think</div><div className="mt-1 text-[10px] text-[#6f7d73]">Your browser is the whole system.</div></div>
                </div>
                <h1 className="max-w-[650px] text-[36px] font-semibold leading-[1.12] tracking-[-.055em] text-[#e0e9e2] sm:text-[51px]">Think forward.<br /><span className="text-[#8fb799]">Evolve every day.</span></h1>
                <p className="mt-5 max-w-[500px] text-[13px] leading-6 text-[#8b988e]">EvolveAI by mdot brings capable open-source assistants right to this device. No sign-in, no cloud conversation, no strings attached.</p>

                <div className="mt-9 grid gap-2.5 sm:grid-cols-3">
                  {[
                    { icon: <FileText size={15} />, title: 'Shape a thought', prompt: 'Help me organize a messy idea into a clear outline.' },
                    { icon: <Sparkles size={15} />, title: 'Make something', prompt: 'Give me a small creative writing prompt for today.' },
                    { icon: <CircleHelp size={15} />, title: 'Learn a concept', prompt: 'Explain a useful idea from cognitive science in plain language.' },
                  ].map((item) => (
                    <button key={item.title} onClick={() => { setDraft(item.prompt); draftRef.current?.focus(); }} className="group rounded-xl border border-white/[.08] bg-[#131b16]/80 p-3.5 text-left transition-colors hover:border-[#506d56]/65 hover:bg-[#18231b]" data-testid={`button-prompt-${item.title.toLowerCase().replaceAll(' ', '-')}`}>
                      <div className="mb-3 text-[#8aad91]">{item.icon}</div>
                      <div className="text-[11px] font-semibold text-[#c8d5ca]">{item.title}</div>
                      <div className="mt-1.5 text-[10px] leading-[1.55] text-[#77847a]">{item.prompt}</div>
                    </button>
                  ))}
                </div>

                <div className="mt-8 rounded-2xl border border-white/[.08] bg-[#111814]/90 p-4 sm:p-5" data-testid="panel-model-readiness">
                  <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                    <div className="flex gap-3">
                      <div className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-[#1c2920] text-[#94be9d]"><HardDrive size={15} /></div>
                      <div>
                        <div className="text-[11px] font-bold text-[#ced9d0]">{runtimeStatus === 'ready' ? 'Model ready on this device' : runtimeStatus === 'loading' ? 'Preparing your model' : runtimeStatus === 'error' ? 'Could not prepare model' : 'One small download, then it’s yours'}</div>
                        <div className="mt-1 text-[10px] leading-5 text-[#7e8a80]">
                          {runtimeStatus === 'loading' ? `${bytesLabel(loadProgress.loaded)}${loadProgress.file ? ` · ${loadProgress.file.split('/').at(-1)}` : ''}` : runtimeStatus === 'error' ? 'Check your connection or available memory, then try again.' : `${selectedModel.label} · ${selectedModel.webgpuSize} WebGPU / ${selectedModel.wasmSize} WASM · cached locally`}
                        </div>
                      </div>
                    </div>
                    {runtimeStatus !== 'ready' && (
                      <button disabled={runtimeStatus === 'loading'} onClick={() => void loadModel()} className="flex h-9 shrink-0 items-center justify-center gap-2 rounded-lg bg-[#a5d2ad] px-3.5 text-[10px] font-bold text-[#152219] transition-colors hover:bg-[#b9dfc0] disabled:cursor-wait disabled:opacity-70" data-testid="button-load-model">
                        {runtimeStatus === 'loading' ? <><Download size={13} className="animate-pulse" /> Downloading</> : runtimeStatus === 'error' ? <><AlertCircle size={13} /> Try again</> : <><Download size={13} /> Load model</>}
                      </button>
                    )}
                  </div>
                  {runtimeStatus === 'loading' && (
                    <div className="mt-4">
                      <div className="h-1 overflow-hidden rounded-full bg-[#26312a]" data-testid="progress-model-download">
                        <div className={`h-full rounded-full bg-[#96c6a0] transition-[width] duration-300 ${loadingPercent === null ? 'progress-indeterminate' : ''}`} style={{ width: loadingPercent === null ? '30%' : `${loadingPercent}%` }} />
                      </div>
                      <div className="mono mt-2 flex justify-between text-[9px] text-[#6e7c71]"><span>{loadProgress.file?.split('/').at(-1) ?? 'Fetching model weights'}</span><span>{loadingPercent === null ? 'Working…' : `${loadingPercent.toFixed(0)}%`}</span></div>
                    </div>
                  )}
                  {runtimeStatus === 'ready' && <div className="mt-3 flex items-center gap-2 text-[10px] text-[#8eb496]" data-testid="status-model-ready"><Check size={13} /> Running with {backend === 'webgpu' ? 'WebGPU acceleration' : 'WASM in your browser'}</div>}
                  {runtimeStatus === 'error' && <div className="mt-3 break-words text-[10px] leading-5 text-[#dc968d]" role="alert" data-testid="status-model-error">{runtimeError}</div>}
                </div>

                <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-[9px] text-[#68766c]">
                  <span className="flex items-center gap-1.5"><ShieldCheck size={12} className="text-[#82a88b]" /> Chats stored on this device only</span>
                  <span className="flex items-center gap-1.5"><Zap size={12} className="text-[#82a88b]" /> WebGPU when available, WASM otherwise</span>
                </div>
              </div>
            </div>
          )}
        </section>

        <footer className="shrink-0 px-4 pb-4 pt-2 sm:px-8 sm:pb-6">
          <div className="mx-auto max-w-[790px]">
            {runtimeError && runtimeStatus !== 'error' && <div className="mb-2 flex items-center gap-2 text-[10px] text-[#dd9a90]" role="alert" data-testid="status-generation-error"><AlertCircle size={13} />{runtimeError}<button onClick={() => setRuntimeError('')} className="ml-auto text-[#a98d87] hover:text-white" aria-label="Dismiss error"><X size={13} /></button></div>}
            <form onSubmit={handleSubmit} className="relative rounded-2xl border border-white/[.12] bg-[#141b16] shadow-[0_12px_45px_rgba(0,0,0,.22)] transition-colors focus-within:border-[#55745b]/70" data-testid="form-chat-composer">
              <textarea
                ref={draftRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Ask something worth thinking about…"
                rows={2}
                className="block max-h-40 min-h-[72px] w-full resize-none bg-transparent px-4 pb-1 pt-4 text-[12px] leading-6 text-[#d3ded5] outline-none placeholder:text-[#66736a] sm:px-5"
                aria-label="Message"
                data-testid="input-chat-message"
              />
              <div className="flex items-center justify-between px-3 pb-2.5 sm:px-4">
                <div className="flex items-center gap-2 text-[9px] text-[#637066]">
                  <span className="mono hidden sm:inline">SHIFT + ENTER FOR A NEW LINE</span>
                  <span className="mono sm:hidden">ON-DEVICE INFERENCE</span>
                  <span className="hidden size-1 rounded-full bg-[#536458] sm:block" />
                  <span className="hidden sm:inline">{selectedModel.capability}</span>
                </div>
                {generating ? (
                  <button type="button" onClick={stopGeneration} className="flex h-8 items-center gap-2 rounded-lg border border-[#604340] bg-[#382522] px-3 text-[10px] font-semibold text-[#e1aaa1] hover:bg-[#462d29]" data-testid="button-stop-generation"><Square size={12} fill="currentColor" /> Stop</button>
                ) : (
                  <button type="submit" disabled={!draft.trim()} className="grid size-8 place-items-center rounded-lg bg-[#9ac9a3] text-[#142018] transition-colors hover:bg-[#b2d9b8] disabled:cursor-not-allowed disabled:bg-[#2b3730] disabled:text-[#657168]" aria-label="Send message" data-testid="button-send-message"><ArrowUp size={16} strokeWidth={2.4} /></button>
                )}
              </div>
            </form>
            <div className="mt-2.5 flex items-center justify-center gap-2 text-center text-[9px] text-[#56635a]" data-testid="text-privacy-footer"><ShieldCheck size={11} /> Your message is processed by this browser. No server receives it.</div>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
