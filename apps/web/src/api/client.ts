/**
 * TAIME API Client
 * Type-safe API calls to the backend
 */

const API_BASE = '/api/v1';

// Types
export interface Dataset {
    id: number;
    name: string;
    filename: string;
    file_size: number;
    sut_type: string;
    description?: string;
    row_count?: number;
    version: number;
    created_at: string;
    updated_at: string;
}

export interface DatasetPreview {
    id: number;
    name: string;
    columns: string[];
    rows: Record<string, unknown>[];
    total_rows: number;
    statistics?: Record<string, unknown> | null;
}

export interface Job {
    id: number;
    dataset_id: number;
    sut_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    current_epoch?: number;
    total_epochs?: number;
    config?: Record<string, unknown>;
    metrics?: Record<string, unknown>;
    error_message?: string;
    started_at?: string;
    completed_at?: string;
    created_at: string;
    updated_at: string;
}

export interface Model {
    id: number;
    name: string;
    job_id: number;
    sut_type: string;
    version: number;
    file_path: string;
    file_size: number;
    metrics?: Record<string, unknown>;
    config?: Record<string, unknown>;
    created_at: string;
}

export interface RuntimeInfo {
    version: string;
    image: string;
    cuda_available: boolean;
    device_name?: string | null;
    torch_version?: string | null;
}

// API functions
async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, {
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
        ...options,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(error.detail || 'Request failed');
    }

    return response.json();
}

// Datasets
export const datasetsApi = {
    list: () => fetchJSON<Dataset[]>(`${API_BASE}/datasets`),
    get: (id: number) => fetchJSON<Dataset>(`${API_BASE}/datasets/${id}`),
    preview: (id: number) =>
        fetchJSON<DatasetPreview>(`${API_BASE}/datasets/${id}/preview`),
    upload: async (file: File, name: string, sutType: string, description?: string) => {
        const formData = new FormData();
        formData.append('file', file, file.name);
        formData.append('name', name);
        formData.append('sut_type', sutType);
        if (description) formData.append('description', description);

        const response = await fetch(`${API_BASE}/datasets`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const payload = await response.json().catch(() => null);
            const detail = payload?.detail ?? 'Upload failed';
            throw new Error(detail);
        }
        return response.json() as Promise<Dataset>;
    },
    delete: (id: number) =>
        fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' }),
};

// Jobs
export const jobsApi = {
    list: () => fetchJSON<Job[]>(`${API_BASE}/jobs`),
    get: (id: number) => fetchJSON<Job>(`${API_BASE}/jobs/${id}`),
    create: (data: { dataset_id: number; sut_type: string; config?: Record<string, unknown> }) =>
        fetchJSON<Job>(`${API_BASE}/jobs`, {
            method: 'POST',
            body: JSON.stringify(data),
        }),
    cancel: (id: number) =>
        fetchJSON<Job>(`${API_BASE}/jobs/${id}/cancel`, { method: 'POST' }),
};

// Models
export const modelsApi = {
    list: () => fetchJSON<Model[]>(`${API_BASE}/models`),
    get: (id: number) => fetchJSON<Model>(`${API_BASE}/models/${id}`),
    downloadUrl: (id: number) => `${API_BASE}/models/${id}/download`,
};

// Runtime
export const runtimeApi = {
    get: () => fetchJSON<RuntimeInfo>(`${API_BASE}/runtime`),
};
