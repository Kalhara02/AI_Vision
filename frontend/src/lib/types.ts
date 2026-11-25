export type Kind = 'bbox' | 'polygon';
export type Source = 'manual' | 'sam' | 'detector';
export type ImageStatus = 'unlabeled' | 'in_progress' | 'done';
export type StudyMode = 'manual' | 'assisted';
export type Tool = 'select' | 'bbox' | 'polygon' | 'smart';

export type Pt = [number, number];
export type BBox = [number, number, number, number]; // x, y, w, h in image pixels

export interface LabelClass {
  id: number;
  name: string;
  color: string;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  created_at: string;
  classes: LabelClass[];
  image_count: number;
  done_count: number;
  annotation_count: number;
  cover_image_id: number | null;
}

export interface ImageMeta {
  id: number;
  project_id: number;
  filename: string;
  width: number;
  height: number;
  status: ImageStatus;
  annotation_count: number;
}

/** Annotation as the editor holds it. `uid` is a client-side key only. */
export interface Annotation {
  uid: string;
  class_id: number;
  kind: Kind;
  bbox: BBox;
  points: Pt[] | null;
  source: Source;
  score: number | null;
}

export interface AnnotationDTO {
  id?: number;
  class_id: number;
  kind: Kind;
  bbox: BBox;
  points: Pt[] | null;
  source: Source;
  score: number | null;
}

/** A detector suggestion that the user has not accepted yet (not saved). */
export interface Proposal {
  uid: string;
  bbox: BBox;
  score: number;
}

export interface TrainingRound {
  round: number;
  trigger: 'auto' | 'manual';
  n_finished: number;
  n_train: number;
  n_holdout: number;
  accepted: boolean;
  version?: number;
  previous_version?: number;
  ap50_previous?: number;
  ap50_new?: number;
  train_seconds: number;
  created: string;
}

export interface ModelStatus {
  version: number;
  state: 'idle' | 'training' | 'scoring' | 'error';
  progress: number;
  message: string;
  history: { version: number; n_images: number; train_seconds: number; created: string; ap50_holdout?: number | null }[];
  rounds: TrainingRound[];
  ranked_version: number | null;
  n_ranked: number;
}

export interface Insights {
  modes: { mode: string; images: number; active_s_mean: number; active_s_sd: number; clicks_per_object: number | null; objects_per_image: number }[];
  time_saved_percent: number | null;
  acceptance: { version: number; shown: number; accepted: number; rejected: number; discarded: number; acceptance_rate: number | null }[];
  rounds: TrainingRound[];
  n_events: number;
}

/** A SAM click: [x, y, 1 = object | 0 = background] */
export type SmartPoint = [number, number, 0 | 1];
