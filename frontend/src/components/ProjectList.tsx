import { useEffect, useState, type FormEvent } from 'react';
import { api } from '../lib/api';
import { navigate } from '../lib/router';
import type { Project } from '../lib/types';

export default function ProjectList() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState('');
  const [classes, setClasses] = useState('');
  const [error, setError] = useState<string | null>(null);

  const load = () => api.listProjects().then(setProjects).catch((e) => setError(String(e.message)));
  useEffect(() => {
    void load();
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
    <div className="page">
      <header className="page-header">
        <h1>LabelLoop</h1>
        <span className="muted">AI-assisted image annotation</span>
      </header>

      <form className="card create-form" onSubmit={create}>
        <h2>New project</h2>
        <label>
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Aerial vehicles" required />
        </label>
        <label>
          Classes <span className="muted">(comma separated, editable later)</span>
          <input value={classes} onChange={(e) => setClasses(e.target.value)} placeholder="car, truck, bus" />
        </label>
        <button type="submit" className="primary">Create project</button>
        {error && <p className="error">{error}</p>}
      </form>

      <h2>Projects</h2>
      {projects === null && <p className="muted">Loading…</p>}
      {projects?.length === 0 && <p className="muted">No projects yet.</p>}
      <div className="project-grid">
        {projects?.map((p) => (
          <a key={p.id} className="card project-card" href={`#/projects/${p.id}`}>
            <h3>{p.name}</h3>
            <div className="class-chips">
              {p.classes.map((c) => (
                <span key={c.id} className="chip" style={{ borderColor: c.color }}>{c.name}</span>
              ))}
            </div>
            <Progress done={p.done_count} total={p.image_count} />
          </a>
        ))}
      </div>
    </div>
  );
}

export function Progress({ done, total }: { done: number; total: number }) {
  const pct = total ? Math.round((100 * done) / total) : 0;
  return (
    <div className="progress" title={`${done} of ${total} images done`}>
      <div className="progress-bar" style={{ width: `${pct}%` }} />
      <span>{done}/{total} done</span>
    </div>
  );
}
