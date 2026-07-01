import { FC, useEffect, useRef } from "react";
import { Session } from "@/lib/chat-utils";
import { useChatStream } from "@/hooks/use-chat-stream"; // Reuse your existing hook
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Loader2,
  FileText,
  Download,
  CheckCircle2,
  PlayCircle,
} from "lucide-react";
import { MarkdownRenderer } from "./markdown-renderer"; // Assuming you have one, or use simple text
import { ChatHeader } from "@/components/chat/header";

interface AnalyzePanelProps {
  session: Session | undefined;
  onUpdateSession: (s: Session) => void;
  onReset: () => void; // Function to go back to "NoSessionState"
}

export const AnalyzePanel: FC<AnalyzePanelProps> = ({
  session,
  onUpdateSession,
  onReset,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // 1. Reuse the robust stream logic we just fixed
  const { isStreaming, loadingStatus, connectToStream } = useChatStream(
    session,
    onUpdateSession
  );

  // 2. Auto-connect on mount (since it's "Analyze Only")
  useEffect(() => {
    if (!session.messages.length && !isStreaming) {
      // If session is new/empty, connect immediately
      connectToStream(session.id);
    } else if (session.id && !isStreaming && !session.graphState) {
      // Re-connect logic if needed
      connectToStream(session.id);
    }
  }, [
    session.id,
    connectToStream,
    isStreaming,
    session.messages.length,
    session.graphState,
  ]);

  // 3. Auto-scroll logs
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [session.messages, loadingStatus]);

  // 4. Extract Final Report (The last bot message)
  const finalReport = session.messages.filter((m) => m.role === "bot").pop();
  const isFinished = !isStreaming && finalReport;

  // 5. Prepare Graph Data
  const graphStructure = session.graphState
    ? {
        nodes: session.graphState.nodes,
        edges: session.graphState.edges,
      }
    : null;

  return (
    <div className="flex flex-col h-full w-full bg-neutral-100/50">
      {/* Header with Actions */}
      <ChatHeader title={session.title || "Automated Analysis"} id={session.id}>
        <div className="flex items-center gap-2">
          {isFinished && (
            <Button
              variant="outline"
              size="sm"
              className="gap-2 text-emerald-600 border-emerald-200 bg-emerald-50 hover:bg-emerald-100"
            >
              <Download className="h-4 w-4" /> Export Results
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onReset}>
            New Analysis
          </Button>
        </div>
      </ChatHeader>

      {/* Main Split Layout */}
      <div className="flex flex-1 overflow-hidden p-4 gap-4">
        {/* LEFT COLUMN: The Workflow Graph (Always Visible) */}
        <Card className="flex-1 max-w-[50%] h-full flex flex-col border-neutral-200 shadow-sm">
          <CardHeader className="py-3 px-4 border-b bg-white rounded-t-xl flex flex-row justify-between items-center">
            <CardTitle className="text-sm font-semibold uppercase tracking-wide text-neutral-500 flex items-center gap-2">
              <PlayCircle className="h-4 w-4" /> Execution Plan
            </CardTitle>
            {isStreaming && (
              <Badge
                variant="secondary"
                className="bg-blue-50 text-blue-700 gap-1 animate-pulse"
              >
                <Loader2 className="h-3 w-3 animate-spin" /> Running
              </Badge>
            )}
            {isFinished && (
              <Badge
                variant="secondary"
                className="bg-emerald-50 text-emerald-700 gap-1"
              >
                <CheckCircle2 className="h-3 w-3" /> Complete
              </Badge>
            )}
          </CardHeader>
        </Card>

        {/* RIGHT COLUMN: Logs & Final Report */}
        <Card className="flex-1 h-full flex flex-col border-neutral-200 shadow-sm bg-white">
          <CardHeader className="py-3 px-4 border-b flex flex-row justify-between items-center">
            <CardTitle className="text-sm font-semibold uppercase tracking-wide text-neutral-500 flex items-center gap-2">
              <FileText className="h-4 w-4" /> Analysis Output
            </CardTitle>
          </CardHeader>

          <ScrollArea className="flex-1 p-6">
            <div className="max-w-prose mx-auto space-y-6">
              {/* 1. If still streaming, show progress logs */}
              {isStreaming && (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-4 bg-blue-50/50 border border-blue-100 rounded-lg text-blue-700 text-sm">
                    <Loader2 className="h-4 w-4 animate-spin shrink-0" />
                    <span>{loadingStatus || "Processing data..."}</span>
                  </div>
                  {/* Optional: Show intermediate logs here if you stored them in session */}
                </div>
              )}

              {/* 2. Final Report (Markdown) */}
              {finalReport ? (
                <div className="prose prose-neutral prose-sm max-w-none animate-in fade-in slide-in-from-bottom-2 duration-500">
                  {/* Render the markdown content */}
                  <MarkdownRenderer content={finalReport.content} />
                </div>
              ) : (
                !isStreaming && (
                  <div className="text-center py-20 text-neutral-400">
                    Waiting for results...
                  </div>
                )
              )}

              <div ref={scrollRef} />
            </div>
          </ScrollArea>
        </Card>
      </div>
    </div>
  );
};
