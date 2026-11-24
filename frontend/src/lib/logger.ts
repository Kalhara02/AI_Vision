/**
 * Interaction logger. Every meaningful user action is recorded with a timestamp
 * and sent to the backend in batches. These logs are the raw data for the
 * effort evaluation (time per image, clicks per object, manual vs. assisted).
 */
import type { StudyMode } from './types';

interface Event {
  session_id: string;
  mode: StudyMode;
  event: string;
  t: number;
  project_id: number | null;
  image_id: number | null;
  payload: Record<string, unknown>;
}

const MODE_KEY = 'labelloop.mode';
const PARTICIPANT_KEY = 'labelloop.participant';

function randomId(): string {
  // crypto.randomUUID is only available in secure contexts; fall back otherwise
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

class InteractionLogger {
  readonly sessionId = randomId();
  private queue: Event[] = [];
  private ctx: { project_id: number | null; image_id: number | null } = { project_id: null, image_id: null };

  constructor() {
    window.setInterval(() => void this.flush(), 3000);
    window.addEventListener('pagehide', () => this.flushBeacon());
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') this.flushBeacon();
      this.log(document.visibilityState === 'hidden' ? 'tab_hidden' : 'tab_visible');
    });
  }

  get mode(): StudyMode {
    return (localStorage.getItem(MODE_KEY) as StudyMode) || 'assisted';
  }

  set mode(m: StudyMode) {
    localStorage.setItem(MODE_KEY, m);
    this.log('mode_change', { mode: m });
  }

  /** Who is labeling (user study). Stored per browser, added to every event. */
  get participant(): string {
    return localStorage.getItem(PARTICIPANT_KEY) || '';
  }

  set participant(p: string) {
    localStorage.setItem(PARTICIPANT_KEY, p);
  }

  setContext(project_id: number | null, image_id: number | null) {
    this.ctx = { project_id, image_id };
  }

  log(event: string, payload: Record<string, unknown> = {}) {
    const p = this.participant;
    this.queue.push({
      session_id: this.sessionId, mode: this.mode, event, t: Date.now(), ...this.ctx,
      payload: p ? { ...payload, participant: p } : payload,
    });
    if (this.queue.length >= 200) void this.flush();
  }

  async flush() {
    if (!this.queue.length) return;
    const events = this.queue.splice(0, this.queue.length);
    try {
      const res = await fetch('/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ events }),
        keepalive: true,
      });
      if (!res.ok) throw new Error(String(res.status));
    } catch {
      this.queue.unshift(...events); // retry on next tick
    }
  }

  private flushBeacon() {
    if (!this.queue.length) return;
    const events = this.queue.splice(0, this.queue.length);
    const blob = new Blob([JSON.stringify({ events })], { type: 'application/json' });
    if (!navigator.sendBeacon('/api/logs', blob)) this.queue.unshift(...events);
  }
}

export const logger = new InteractionLogger();
