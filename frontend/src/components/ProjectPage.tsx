import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api } from '../lib/api';
import { logger } from '../lib/logger';
import { navigate } from '../lib/router';
import type { ImageMeta, Insights, Project, StudyMode } from '../lib/types';
import { Progress } from './ProjectList';

const FORMATS = [
  { id: 'coco', label: 'COCO JSON (boxes + polygons)' },
  { id: 'yolo', label: 'YOLO detection (txt)' },
  { id: 'yolo-seg', label: 'YOLO segmentation (txt)' },
  { id: 'voc', label: 'Pascal VOC (xml, boxes)' },
];

export default function ProjectPage({ projectId }: { projectId: number }) {
  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [newClass, setNewClass] = useState('');
  const [fmt, setFmt] = useState('coco');
  const [includeImages, setIncludeImages] = useState(true);
  const [onlyDone, setOnlyDone] = useState(false);
  const [mode, setMode] = useState<StudyMode>(logger.mode);
  const [participant, setParticipant] = useState(logger.participant);
  const [insights, setInsights] = useState<Insights | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const reload = useCallback(async () => {
    try {
      const [p, ims] = await Promise.all([api.getProject(projectId), api.listImages(projectId)]);
      setProject(p);
      setImages(ims);
      api.insights(projectId).then(setInsights).catch(() => {});
    } catch (e) {
      setError((e as Error).message);
    }
  }, [projectId]);

  useEffect(() => {
    void reload();
    logger.setContext(projectId, null);
  }, [reload, projectId]);

  const upload = async (files: File[]) => {
    const imgs = files.filter((f) => f.type.startsWith('image/') || /\.(tiff?|png|jpe?g|webp|bmp)$/i.test(f.name));
    if (!imgs.length) return;
    const batchSize = 20;
    const errors: string[] = [];
    for (let i = 0; i < imgs.length; i += batchSize) {
      setUploading(`Uploading ${Math.min(i + batchSize, imgs.length)} / ${imgs.length}…`);
      try {
        const res = await api.uploadImages(projectId, imgs.slice(i, i + batchSize));
        res.errors.forEach((e) => errors.push(`${e.filename}: ${e.error}`));
      } catch (e) {
        errors.push((e as Error).message);
      }
    }
    setUploading(null);
    setError(errors.length ? errors.join('\n') : null);
    logger.log('images_uploaded', { count: imgs.length, errors: errors.length });
    await reload();
  };

  const addClass = async (e: FormEvent) => {
    e.preventDefault();
    if (!newClass.trim()) return;
    try {
      await api.addClass(projectId, newClass.trim());
      setNewClass('');
      await reload();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const removeClass = async (id: number) => {
    try {
      await api.deleteClass(id);
      await reload();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const recolor = async (id: number, color: string) => {
    await api.updateClass(id, { color });
    await reload();
  };

  const removeProject = async () => {
    if (!project || !confirm(`Delete project "${project.name}" with all images and annotations? This cannot be undone.`)) return;
    await api.deleteProject(projectId);
    navigate('/');
  };

  const changeMode = (m: StudyMode) => {
    logger.mode = m;
    setMode(m);
  };

  const firstOpen = images.find((i) => i.status !== 'done') ?? images[0];

  if (!project) return <div className="page">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <a href="#/" className="back">← Projects</a>
        <h1>{project.name}</h1>
        <Progress done={project.done_count} total={project.image_count} />
        {firstOpen && (
          <button className="primary" onClick={() => navigate(`/projects/${projectId}/images/${firstOpen.id}`)}>
            {project.done_count ? 'Continue annotating' : 'Start annotating'}
          </button>
        )}
      </header>

      {error && <pre className="error">{error}</pre>}

      <div className="project-layout">
        <section>
          <div
            className={`dropzone ${dragOver ? 'over' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              void upload(Array.from(e.dataTransfer.files));
            }}
            onClick={() => fileInput.current?.click()}
          >
            {uploading ?? 'Drop images here or click to upload (JPEG, PNG, WebP, TIFF)'}
            <input
              ref={fileInput}
              type="file"
              accept="image/*,.tif,.tiff"
              multiple
              hidden
              onChange={(e) => {
                void upload(Array.from(e.target.files ?? []));
                e.target.value = '';
              }}
            />
          </div>

          <div className="thumb-grid">
            {images.map((im) => (
              <a key={im.id} className={`thumb status-${im.status}`} href={`#/projects/${projectId}/images/${im.id}`} title={im.filename}>
                <img src={api.imageUrl(im.id)} loading="lazy" alt={im.filename} />
                <span className="thumb-meta">
                  <span className={`dot ${im.status}`} />
                  {im.annotation_count} obj
                </span>
              </a>
            ))}
          </div>
        </section>

        <aside className="side">
          <div className="card">
            <h3>Classes</h3>
            <ul className="class-list">
              {project.classes.map((c, i) => (
                <li key={c.id}>
                  <input type="color" value={c.color} onChange={(e) => void recolor(c.id, e.target.value)} />
                  <span className="kbd">{i < 9 ? i + 1 : ''}</span>
                  <span className="grow">{c.name}</span>
                  <button className="icon" title="Delete class" onClick={() => void removeClass(c.id)}>✕</button>
                </li>
              ))}
            </ul>
            <form onSubmit={addClass} className="row">
              <input value={newClass} onChange={(e) => setNewClass(e.target.value)} placeholder="new class" />
              <button type="submit">Add</button>
            </form>
          </div>

          <div className="card">
            <h3>Session mode</h3>
            <p className="muted small">
              Used for the effort study. <b>Manual</b> hides all AI assistance; <b>assisted</b> enables it.
              Every logged action is tagged with this mode.
            </p>
            <div className="segmented">
              {(['manual', 'assisted'] as StudyMode[]).map((m) => (
                <button key={m} className={mode === m ? 'active' : ''} onClick={() => changeMode(m)}>{m}</button>
              ))}
            </div>
            <label className="small">
              Participant ID (user study)
              <input
                value={participant}
                placeholder="e.g. P1"
                onChange={(e) => {
                  setParticipant(e.target.value);
                  logger.participant = e.target.value.trim();
                }}
              />
            </label>
          </div>

          {insights && (insights.modes.length > 0 || insights.rounds.length > 0) && (
            <div className="card">
              <h3>Insights</h3>
              {insights.time_saved_percent != null && (
                <p className="headline">
                  <b>{insights.time_saved_percent > 0 ? `${insights.time_saved_percent}% less time` : `${-insights.time_saved_percent}% more time`}</b>{' '}
                  per image with AI assistance
                </p>
              )}
              {insights.modes.length > 0 && (
                <table className="mini-table">
                  <thead><tr><th>mode</th><th>images</th><th>s / image</th><th>clicks / obj</th></tr></thead>
                  <tbody>
                    {insights.modes.map((m) => (
                      <tr key={m.mode}>
                        <td>{m.mode}</td><td>{m.images}</td>
                        <td>{m.active_s_mean} ± {m.active_s_sd}</td><td>{m.clicks_per_object ?? 'n/a'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {insights.acceptance.length > 0 && (
                <table className="mini-table">
                  <thead><tr><th>detector</th><th>suggested</th><th>kept</th></tr></thead>
                  <tbody>
                    {insights.acceptance.map((a) => (
                      <tr key={a.version}>
                        <td>v{a.version}</td><td>{a.shown}</td>
                        <td>{a.acceptance_rate != null ? `${Math.round(a.acceptance_rate * 100)}%` : 'n/a'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {insights.rounds.length > 0 && (
                <p className="muted small">
                  Training rounds: {insights.rounds.map((r) => `#${r.round} ${r.accepted ? `v${r.version}` : 'kept old'}${r.ap50_new != null ? ` (AP50 ${r.ap50_new.toFixed(2)})` : ''}`).join(' · ')}
                </p>
              )}
              <p className="muted small">Active time per finished image; pauses over 60 s are not counted.</p>
            </div>
          )}

          <div className="card">
            <h3>Export</h3>
            <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
              {FORMATS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
            </select>
            <label className="check">
              <input type="checkbox" checked={includeImages} onChange={(e) => setIncludeImages(e.target.checked)} /> include images
            </label>
            <label className="check">
              <input type="checkbox" checked={onlyDone} onChange={(e) => setOnlyDone(e.target.checked)} /> only images marked done
            </label>
            <a
              className="button primary"
              href={api.exportUrl(projectId, fmt, includeImages, onlyDone)}
              onClick={() => logger.log('export', { format: fmt, include_images: includeImages, only_done: onlyDone })}
            >
              Download zip
            </a>
            <a className="small" href={api.logsUrl(projectId)} target="_blank" rel="noreferrer">interaction logs (JSON)</a>
          </div>

          <button className="danger" onClick={() => void removeProject()}>Delete project</button>
        </aside>
      </div>
    </div>
  );
}
