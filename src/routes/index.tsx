import { createFileRoute, ClientOnly } from "@tanstack/react-router";
import { lazy, Suspense } from "react";

const EvolveApp = lazy(() => import("@/components/evolve/EvolveApp"));

const DESC =
  "Private, on-device AI chat by mdot. Run lightweight open-source language models in your browser with no account, API keys, or cloud inference.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "EvolveAI — By mdot" },
      { name: "description", content: DESC },
      { property: "og:title", content: "EvolveAI — By mdot" },
      { property: "og:description", content: DESC },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
      { name: "twitter:title", content: "EvolveAI — By mdot" },
      { name: "twitter:description", content: DESC },
    ],
  }),
  component: Index,
});

function Index() {
  const fallback = <div className="app-shell" />;
  return (
    <ClientOnly fallback={fallback}>
      <Suspense fallback={fallback}>
        <EvolveApp />
      </Suspense>
    </ClientOnly>
  );
}
