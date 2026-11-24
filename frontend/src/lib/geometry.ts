import type { Annotation, AnnotationDTO, BBox, Pt } from './types';

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export function bboxFromPoints(points: Pt[]): BBox {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs);
  const y0 = Math.min(...ys);
  return [x0, y0, Math.max(...xs) - x0, Math.max(...ys) - y0];
}

export function clipBox(b: BBox, w: number, h: number): BBox {
  const x0 = clamp(b[0], 0, w);
  const y0 = clamp(b[1], 0, h);
  const x1 = clamp(b[0] + b[2], 0, w);
  const y1 = clamp(b[1] + b[3], 0, h);
  return [x0, y0, x1 - x0, y1 - y0];
}

export function boxFromCorners(a: Pt, b: Pt): BBox {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])];
}

export function dist(a: Pt, b: Pt) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

let counter = 0;
export const newUid = () => `a${Date.now().toString(36)}_${(counter++).toString(36)}`;

export function fromDTO(d: AnnotationDTO): Annotation {
  return { uid: newUid(), class_id: d.class_id, kind: d.kind, bbox: d.bbox, points: d.points, source: d.source, score: d.score };
}

export function toDTO(a: Annotation): AnnotationDTO {
  return { class_id: a.class_id, kind: a.kind, bbox: a.bbox, points: a.points, source: a.source, score: a.score };
}

/** hex "#rrggbb" + alpha -> rgba() */
export function withAlpha(hex: string, alpha: number) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function bboxIou(a: BBox, b: BBox): number {
  const iw = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}
