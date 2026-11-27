import type { AnnotationDTO, BBox, ImageMeta, ImageStatus, Insights, LabelClass, ModelStatus, Project, Pt, SamMode, SmartPoint } from './types';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  listProjects: () => request<Project[]>('/api/projects'),
  getProject: (id: number) => request<Project>(`/api/projects/${id}`),
  createProject: (name: string, classes: { name: string }[], description = '') =>
    request<Project>('/api/projects', json('POST', { name, description, classes })),
  deleteProject: (id: number) => request<void>(`/api/projects/${id}`, { method: 'DELETE' }),

  addClass: (projectId: number, name: string, color?: string) =>
    request<LabelClass>(`/api/projects/${projectId}/classes`, json('POST', { name, color })),
  updateClass: (id: number, patch: Partial<Pick<LabelClass, 'name' | 'color'>>) =>
    request<LabelClass>(`/api/classes/${id}`, json('PATCH', patch)),
  deleteClass: (id: number) => request<void>(`/api/classes/${id}`, { method: 'DELETE' }),

  listImages: (projectId: number) => request<ImageMeta[]>(`/api/projects/${projectId}/images`),
  uploadImages: (projectId: number, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f, f.name));
    return request<{ created: ImageMeta[]; errors: { filename: string; error: string }[] }>(
      `/api/projects/${projectId}/images`,
      { method: 'POST', body: fd },
    );
  },
  deleteImage: (id: number) => request<void>(`/api/images/${id}`, { method: 'DELETE' }),
  setImageStatus: (id: number, status: ImageStatus) =>
    request<ImageMeta>(`/api/images/${id}`, json('PATCH', { status })),
  imageUrl: (id: number) => `/api/images/${id}/file`,

  getAnnotations: (imageId: number) =>
    request<{ image: ImageMeta; annotations: AnnotationDTO[] }>(`/api/images/${imageId}/annotations`),
  saveAnnotations: (imageId: number, annotations: AnnotationDTO[], status?: ImageStatus) =>
    request<{ image: ImageMeta; annotations: AnnotationDTO[] }>(
      `/api/images/${imageId}/annotations`,
      json('PUT', { annotations, status }),
    ),

  exportUrl: (projectId: number, format: string, includeImages: boolean, onlyDone: boolean) =>
    `/api/projects/${projectId}/export?format=${encodeURIComponent(format)}` +
    `&include_images=${includeImages}&only_done=${onlyDone}`,
  logsUrl: (projectId: number) => `/api/projects/${projectId}/logs`,

  // ---------- AI assistance ----------
  aiStatus: () => request<{ enabled: boolean; device?: string; reason?: string }>('/api/ai/status'),
  samPrepare: (imageId: number) => request<{ encode_ms: number }>(`/api/images/${imageId}/sam/prepare`, { method: 'POST' }),
  samPredict: (imageId: number, points: SmartPoint[], box: BBox | null, mode: SamMode) =>
    request<{
      points: Pt[] | null;
      score: number;
      focus_window: [number, number, number, number] | null;
      encode_ms: number;
      decode_ms: number;
    }>(`/api/images/${imageId}/sam/predict`, json('POST', { points, box, mode })),
  samBoxes: (imageId: number, boxes: BBox[]) =>
    request<{ polygons: (Pt[] | null)[]; scores: number[] }>(`/api/images/${imageId}/sam/boxes`, json('POST', { boxes })),
  proposals: (imageId: number, threshold: number) =>
    request<{
      version: number;
      proposals: { bbox: BBox; score: number; class_id: number; source: 'trained' | 'zero-shot' }[];
      zero_shot_classes: string[];
      ms: number;
    }>(
      `/api/images/${imageId}/proposals?threshold=${threshold}`,
    ),
  rank: (projectId: number) => request<{ started: boolean; n_images: number }>(`/api/projects/${projectId}/model/rank`, { method: 'POST' }),
  next: (projectId: number, current: number, order: 'upload' | 'uncertainty') =>
    request<{ image_id: number | null; order: string; score: number | null }>(
      `/api/projects/${projectId}/next?current=${current}&order=${order}`,
    ),
  insights: (projectId: number) => request<Insights>(`/api/projects/${projectId}/insights`),
  modelStatus: (projectId: number) => request<ModelStatus>(`/api/projects/${projectId}/model`),
  train: (projectId: number, epochs: number) =>
    request<{ started: boolean; n_images: number }>(`/api/projects/${projectId}/model/train?epochs=${epochs}`, {
      method: 'POST',
    }),
};
