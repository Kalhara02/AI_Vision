import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { api } from '../lib/api';
import { logger } from '../lib/logger';
import { navigate } from '../lib/router';
import type { ImageMeta, Insights, ModelStatus, Project, StudyMode } from '../lib/types';
import { Icon, Ring, type IconName } from './Icons';
import ModelChart from './ModelChart';
import TopNav from './TopNav';

const FORMATS = [
  { id: 'coco', label: 'COCO JSON (boxes + polygons)' },
  { id: 'yolo', label: 'YOLO detection (txt)' },
  { id: 'yolo-seg', label: 'YOLO segmentation (txt)' },
  { id: 'voc', label: 'Pascal VOC (xml, boxes)' },
];

type Filter = 'all' | 'todo' | 'done';

function Stat({ icon, label, value, sub }: { icon: IconName; label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <span className="stat-icon"><Icon name={icon} size={18} /></span>
      <div>
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
        {sub && <div className="stat-sub">{sub}</div>}
      </div>
    </div>
  );
}

function Card({ icon, title, children, extra }: { icon: IconName; title: string; children: ReactNode; extra?: ReactNode }) {
  return (
    <div className="card side-card">
      <div className="card-head">
        <h3><Icon name={icon} size={16} /> {title}</h3>
        {extra}
      </div>
      {children}
    </div>
  );
}

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
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const fileInput = useRef<HTMLInputElement>(null);

  const reload = useCallback(async () => {
    try {
      const [p, ims] = await Promise.all([api.getProject(projectId), api.listImages(projectId)]);
      setProject(p);
      setImages(ims);
      api.insights(projectId).then(setInsights).catch(() => {});
      api.modelStatus(projectId).then(setModel).catch(() => {});
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

  if (!project) {
    return (
      <div className="app-shell">
        <TopNav />
        <main className="page">{error ? <p className="error">{error}</p> : <div className="empty">Loading…</div>}</main>
      </div>
    );
  }

  const firstOpen = images.find((i) => i.status !== 'done') ?? images[0];
  const todo = images.filter((i) => i.status !== 'done');
  const shown = filter === 'all' ? images : filter === 'done' ? images.filter((i) => i.status === 'done') : todo;
  const lastAccepted = model?.rounds.filter((r) => r.accepted).at(-1);

  return (
    <div className="app-shell">
      <TopNav crumbs={[{ label: project.name }]} />
      <main className="page">
        <header className="project-header">
          <div>
            <h1>{project.name}</h1>
            <div className="class-chips">
              {project.classes.map((c) => (
                <span key={c.id} className="chip"><span className="swatch" style={{ background: c.color }} />{c.name}</span>
              ))}
            </div>
          </div>
          <div className="header-actions">
            <button onClick={() => fileInput.current?.click()}><Icon name="upload" /> Upload images</button>
            {firstOpen && (
              <button className="primary big" onClick={() => navigate(`/projects/${projectId}/images/${firstOpen.id}`)}>
                <Icon name="sparkle" /> {project.done_count ? 'Continue annotating' : 'Start annotating'}
              </button>
            )}
          </div>
        </header>

        {error && <pre className="error banner">{error}</pre>}

        <section className="stat-row">
          <div className="stat">
            <Ring value={project.image_count ? project.done_count / project.image_count : 0} size={46} />
            <div>
              <div className="stat-label">Progress</div>
              <div className="stat-value">{project.done_count} / {project.image_count}</div>
              <div className="stat-sub">images done</div>
            </div>
          </div>
          <Stat icon="tag" label="Objects labeled" value={project.annotation_count.toLocaleString()} sub={`${todo.length} images to go`} />
          <Stat
            icon="cpu"
            label="Detector"
            value={model ? `v${model.version}` : '–'}
            sub={lastAccepted?.ap50_new != null ? `AP50 ${lastAccepted.ap50_new.toFixed(2)} on held-out` : model ? 'pretrained (COCO)' : 'AI off'}
          />
          <Stat
            icon="clock"
            label="Time saved by AI"
            value={insights?.time_saved_percent != null ? `${insights.time_saved_percent}%` : '–'}
            sub={insights?.time_saved_percent != null ? 'per image vs. manual' : 'label in both modes to measure'}
          />
        </section>

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
              <Icon name="upload" size={22} />
              <span>{uploading ?? 'Drop images here or click to upload'}</span>
              <span className="muted small">JPEG, PNG, WebP, TIFF · EXIF rotation and 16-bit images handled</span>
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

            <div className="grid-toolbar">
              <div className="tabs">
                {([['all', 'All', images.length], ['todo', 'To do', todo.length], ['done', 'Done', project.done_count]] as [Filter, string, number][]).map(
                  ([id, label, n]) => (
                    <button key={id} className={filter === id ? 'active' : ''} onClick={() => setFilter(id)}>
                      {label} <span className="count">{n}</span>
                    </button>
                  ),
                )}
              </div>
            </div>

            {shown.length === 0 ? (
              <div className="empty">
                <Icon name="image" size={30} />
                <p>{images.length ? 'Nothing here.' : 'Upload images to get started.'}</p>
              </div>
            ) : (
              <div className="thumb-grid">
                {shown.map((im) => (
                  <a key={im.id} className={`thumb status-${im.status}`} href={`#/projects/${projectId}/images/${im.id}`} title={im.filename}>
                    <img src={api.imageUrl(im.id)} loading="lazy" alt={im.filename} />
                    <span className={`thumb-status ${im.status}`}>
                      {im.status === 'done' ? <Icon name="check" size={12} /> : null}
                      {im.status.replace('_', ' ')}
                    </span>
                    <span className="thumb-meta">
                      <span className="thumb-name">{im.filename}</span>
                      <span><Icon name="tag" size={12} /> {im.annotation_count}</span>
                    </span>
                  </a>
                ))}
              </div>
            )}
          </section>

          <aside className="side">
            {insights && (insights.modes.length > 0 || insights.rounds.length > 0 || insights.acceptance.length > 0) && (
              <Card icon="chart" title="Insights">
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
                          <td><span className={`mode-dot ${m.mode}`} />{m.mode}</td>
                          <td>{m.images}</td>
                          <td>{m.active_s_mean} ± {m.active_s_sd}</td>
                          <td>{m.clicks_per_object ?? '–'}</td>
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
                          <td>v{a.version}</td>
                          <td>{a.shown}</td>
                          <td>{a.acceptance_rate != null ? `${Math.round(a.acceptance_rate * 100)}%` : '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {model && model.rounds.length > 0 && <ModelChart rounds={model.rounds} />}
                <p className="muted small">Active time per finished image; pauses over 60 s are not counted.</p>
              </Card>
            )}

            <Card icon="tag" title="Classes">
              <ul className="class-list">
                {project.classes.map((c, i) => (
                  <li key={c.id}>
                    <input type="color" value={c.color} onChange={(e) => void recolor(c.id, e.target.value)} title="Change color" />
                    <span className="grow">{c.name}</span>
                    {i < 9 && <span className="kbd">{i + 1}</span>}
                    <button className="icon" title="Delete class" onClick={() => void removeClass(c.id)}><Icon name="trash" size={15} /></button>
                  </li>
                ))}
              </ul>
              <form onSubmit={addClass} className="row">
                <input value={newClass} onChange={(e) => setNewClass(e.target.value)} placeholder="New class" />
                <button type="submit"><Icon name="plus" size={16} /></button>
              </form>
            </Card>

            <Card icon="user" title="Session">
              <div className="segmented">
                {(['manual', 'assisted'] as StudyMode[]).map((m) => (
                  <button key={m} className={mode === m ? 'active' : ''} onClick={() => changeMode(m)}>
                    <Icon name={m === 'manual' ? 'polygon' : 'sparkle'} size={15} /> {m}
                  </button>
                ))}
              </div>
              <p className="muted small">
                <b>Manual</b> hides all AI help, <b>assisted</b> turns it on. Every action is logged with this mode, which is how the
                effort saved is measured.
              </p>
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
            </Card>

            <Card icon="download" title="Export">
              <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
                {FORMATS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
              </select>
              <label className="check">
                <input type="checkbox" checked={includeImages} onChange={(e) => setIncludeImages(e.target.checked)} /> Include images
              </label>
              <label className="check">
                <input type="checkbox" checked={onlyDone} onChange={(e) => setOnlyDone(e.target.checked)} /> Only images marked done
              </label>
              <a
                className="button primary"
                href={api.exportUrl(projectId, fmt, includeImages, onlyDone)}
                onClick={() => logger.log('export', { format: fmt, include_images: includeImages, only_done: onlyDone })}
              >
                <Icon name="download" /> Download zip
              </a>
              <a className="small link-quiet" href={api.logsUrl(projectId)} target="_blank" rel="noreferrer">Interaction logs (JSON)</a>
            </Card>

            <button className="danger ghost" onClick={() => void removeProject()}><Icon name="trash" size={15} /> Delete project</button>
          </aside>
        </div>
      </main>
    </div>
  );
}
