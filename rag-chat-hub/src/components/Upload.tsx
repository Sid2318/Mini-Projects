import React, { useState, useRef } from 'react';
import { Upload as UploadIcon, File, CheckCircle, AlertCircle, X } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { apiClient, ApiError } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface UploadedFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
}

export const Upload = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const acceptedTypes = '.pdf,.txt,.docx';
  const maxFileSize = 10 * 1024 * 1024; // 10MB

  const validateFile = (file: File): string | null => {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(extension)) {
      return 'File type not supported. Please upload PDF, TXT, or DOCX files.';
    }
    if (file.size > maxFileSize) {
      return 'File size exceeds 10MB limit.';
    }
    return null;
  };

  const handleFiles = (files: FileList) => {
    const validFiles: File[] = [];
    const invalidFiles: string[] = [];

    Array.from(files).forEach((file) => {
      const error = validateFile(file);
      if (error) {
        invalidFiles.push(`${file.name}: ${error}`);
      } else {
        validFiles.push(file);
      }
    });

    if (invalidFiles.length > 0) {
      toast({
        title: 'Invalid Files',
        description: invalidFiles.join('\n'),
        variant: 'destructive',
      });
    }

    if (validFiles.length > 0) {
      const newFiles: UploadedFile[] = validFiles.map((file) => ({
        id: Date.now() + Math.random().toString(),
        file,
        status: 'pending',
      }));

      setUploadedFiles((prev) => [...prev, ...newFiles]);
      uploadFiles(newFiles);
    }
  };

  const uploadFiles = async (filesToUpload: UploadedFile[]) => {
    const files = filesToUpload.map((f) => f.file);
    
    // Update status to uploading
    setUploadedFiles((prev) =>
      prev.map((f) =>
        filesToUpload.some((upload) => upload.id === f.id)
          ? { ...f, status: 'uploading' }
          : f
      )
    );

    try {
      setUploadProgress(0);
      // Simulate progress (since we can't track real progress with FormData)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      const response = await apiClient.uploadFiles(files);
      
      clearInterval(progressInterval);
      setUploadProgress(100);

      // Update successful files
      setUploadedFiles((prev) =>
        prev.map((f) =>
          filesToUpload.some((upload) => upload.id === f.id)
            ? { ...f, status: 'success' }
            : f
        )
      );

      toast({
        title: 'Upload Successful',
        description: `${response.files_processed} files processed successfully.`,
      });

      setTimeout(() => setUploadProgress(0), 1000);
    } catch (error) {
      // Update failed files
      const errorMessage = error instanceof ApiError ? error.message : 'Upload failed';
      
      setUploadedFiles((prev) =>
        prev.map((f) =>
          filesToUpload.some((upload) => upload.id === f.id)
            ? { ...f, status: 'error', error: errorMessage }
            : f
        )
      );

      toast({
        title: 'Upload Failed',
        description: errorMessage,
        variant: 'destructive',
      });

      setUploadProgress(0);
    }
  };

  const removeFile = (id: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-success" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case 'uploading':
        return <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />;
      default:
        return <File className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-6 border-b">
        <h2 className="text-xl font-semibold mb-2">Document Upload</h2>
        <p className="text-sm text-muted-foreground">
          Upload PDF, TXT, or DOCX files to add to your knowledge base
        </p>
      </div>

      <div className="flex-1 p-6 space-y-6">
        {/* Upload Area */}
        <Card
          className={`
            border-2 border-dashed transition-all duration-200 cursor-pointer
            ${isDragging 
              ? 'border-primary bg-primary/5 shadow-upload' 
              : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-primary/2'
            }
          `}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="p-8 text-center">
            <UploadIcon className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-medium mb-2">
              Drag & drop files here
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              or click to select files (PDF, TXT, DOCX)
            </p>
            <Button variant="outline" type="button">
              Select Files
            </Button>
          </div>
        </Card>

        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedTypes}
          multiple
          onChange={handleFileInput}
          className="hidden"
        />

        {/* Upload Progress */}
        {uploadProgress > 0 && (
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Uploading...</span>
              <span className="text-sm text-muted-foreground">{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="h-2" />
          </Card>
        )}

        {/* File List */}
        {uploadedFiles.length > 0 && (
          <div className="space-y-3">
            <h3 className="font-medium text-sm text-muted-foreground uppercase tracking-wide">
              Uploaded Files ({uploadedFiles.length})
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {uploadedFiles.map((uploadedFile) => (
                <Card key={uploadedFile.id} className="p-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3 flex-1 min-w-0">
                      {getStatusIcon(uploadedFile.status)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">
                          {uploadedFile.file.name}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatFileSize(uploadedFile.file.size)}
                        </p>
                        {uploadedFile.error && (
                          <p className="text-xs text-destructive mt-1">
                            {uploadedFile.error}
                          </p>
                        )}
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(uploadedFile.id)}
                      className="h-8 w-8 p-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};