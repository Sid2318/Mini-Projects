import React from 'react';
import { Brain } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Upload } from '@/components/Upload';
import { Chat } from '@/components/Chat';

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-bg">
      {/* Header */}
      <header className="border-b bg-card/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-primary rounded-lg">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">RAG Assistant</h1>
              <p className="text-sm text-muted-foreground">
                Upload documents and ask intelligent questions
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-6 h-[calc(100vh-88px)]">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-full">
          {/* Left Panel - Upload */}
          <div className="lg:col-span-2">
            <Card className="h-full shadow-card bg-gradient-card">
              <Upload />
            </Card>
          </div>

          {/* Right Panel - Chat */}
          <div className="lg:col-span-3">
            <Card className="h-full shadow-card bg-gradient-card">
              <Chat />
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
