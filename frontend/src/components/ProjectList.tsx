import { useEffect, useState, type FormEvent } from 'react';
import { api } from '../lib/api';
import { navigate } from '../lib/router';
import type { Project } from '../lib/types';
import { Icon, Ring, type IconName } from './Icons';
import TopNav from './TopNav';

const STEPS: { icon: IconName; title: string; text: string }[] = [
  { icon: 'upload', title: 'Upload', text: 'Drop in your images. JPEG, PNG, WebP and 16-bit TIFF.' },
  { icon: 'zap', title: 'AI suggests', text: 'Type your own classes: suggestions start zero-shot, SAM outlines anything with a click.' },
  { icon: 'check', title: 'You correct', text: 'Accept, fix or delete. The detector retrains on your corrections.' },
  { icon: 'download', title: 'Export', text: 'COCO, YOLO, YOLO-seg or Pascal VOC, with the images.' },
];

export default function ProjectList() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState('');
  const [classes, setClasses] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listProjects().then(setProjects).catch((e) => setError(String(e.message)));
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const names = classes.split(',').map((c) => c.trim()).filter(Boolean);
    if (!name.trim()) return;
    try {
      const p = await api.createProject(name.trim(), names.map((n) => ({ name: n })));
      navigate(`/projects/${p.id}`);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="app-shell">
      <TopNav />
      <main className="page">
        <section className="hero">
          <div className="hero-text">
            <span className="eyebrow"><Icon name="sparkle" size={14} /> AI-assisted image annotation</span>
            <h1>Label images faster with an AI that learns from you</h1>
            <p>
              Click an object and SAM 2 outlines it. A detector suggests the rest, and every correction you make
              trains it further. Everything runs locally; export in standard formats when you are done.
            </p>
          </div>

          <form className="card create-card" onSubmit={create}>
            <h3><Icon name="plus" /> New project</h3>
            <label>
              Name
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Street scenes" required />
            </label>
            <label>
              Classes <span className="muted">(comma separated, editable later)</span>
              <input value={classes} onChange={(e) => setClasses(e.target.value)} placeholder="car, person, traffic light" />
            </label>
            <button type="submit" className="primary big">Create project</button>
            {error && <p className="error small">{error}</p>}
          </form>
        </section>

        <section className="steps">
          {STEPS.map((s, i) => (
            <div key={s.title} className="step">
              <span className="step-icon"><Icon name={s.icon} size={20} /></span>
              <div>
                <div className="step-title"><span className="step-num">{i + 1}</span>{s.title}</div>
                <div className="muted small">{s.text}</div>
              </div>
            </div>
          ))}
        </section>

        <div className="section-head">
          <h2>Projects</h2>
          {projects && <span className="muted">{projects.length} total</span>}
        </div>
        {projects === null && <div className="empty">Loading…</div>}
        {projects?.length === 0 && (
          <div className="empty">
            <Icon name="layers" size={32} />
            <p>No projects yet. Create one above to start labeling.</p>
          </div>
        )}
        <div className="project-grid">
          {projects?.map((p) => (
            <a key={p.id} className="card project-card" href={`#/projects/${p.id}`}>
              <div className="cover">
                {p.cover_image_id != null ? (
                  <img src={api.imageUrl(p.cover_image_id)} alt="" loading="lazy" />
                ) : (
                  <span className="cover-empty"><Icon name="image" size={28} /></span>
                )}
              </div>
              <div className="project-body">
                <div className="project-title">
                  <div>
                    <h3>{p.name}</h3>
                    <div className="class-chips">
                      {p.classes.map((c) => (
                        <span key={c.id} className="chip"><span className="swatch" style={{ background: c.color }} />{c.name}</span>
                      ))}
                    </div>
                  </div>
                  <Ring value={p.image_count ? p.done_count / p.image_count : 0} />
                </div>
                <div className="project-meta muted small">
                  <span><Icon name="image" size={14} /> {p.image_count} images</span>
                  <span><Icon name="check" size={14} /> {p.done_count} done</span>
                  <span><Icon name="tag" size={14} /> {p.annotation_count.toLocaleString()} objects</span>
                </div>
              </div>
            </a>
          ))}
        </div>
      </main>
    </div>
  );
}
