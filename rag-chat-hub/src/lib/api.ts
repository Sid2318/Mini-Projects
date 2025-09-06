const API_BASE_URL = 'http://localhost:8000';

export interface UploadResponse {
  status: string;
  files_processed: number;
}

export interface AskResponse {
  answer: string;
  context: Array<{
    content: string;
    filename?: string;
    page?: number;
  }>;
}

export class ApiError extends Error {
  status?: number;
  
  constructor(options: { message: string; status?: number }) {
    super(options.message);
    this.name = 'ApiError';
    this.status = options.status;
  }
}

class ApiClient {
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorText = await response.text();
      throw new ApiError({
        message: `API Error: ${response.status} - ${errorText}`,
        status: response.status,
      });
    }
    return response.json();
  }

  async uploadFiles(files: File[]): Promise<UploadResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/upload/`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<UploadResponse>(response);
  }

  async askQuestion(question: string): Promise<AskResponse> {
    const response = await fetch(
      `${API_BASE_URL}/ask/?q=${encodeURIComponent(question)}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    return this.handleResponse<AskResponse>(response);
  }
}

export const apiClient = new ApiClient();