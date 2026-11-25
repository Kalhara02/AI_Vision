import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { Icon, Logo } from './Icons';

/** App header for the project pages: brand, navigation and the AI engine status. */
export default function TopNav({ crumbs = [] }: { crumbs?: { label: string; href?: string }[] }) {
  const [ai, setAi] = useState<{ enabled: boolean; device?: string } | null>(null);
  useEffect(() => {
    api.aiStatus().then(setAi).catch(() => setAi({ enabled: false }));
  }, []);

  return (
    <header className="topnav">
      <a href="#/" className="brand">
        <Logo />
        <span className="brand-name">LabelLoop</span>
      </a>
      <nav className="crumbs">
        <a href="#/">Projects</a>
        {crumbs.map((c) => (
          <span key={c.label} className="crumb">
            <Icon name="chevronRight" size={14} />
            {c.href ? <a href={c.href}>{c.label}</a> : <span>{c.label}</span>}
          </span>
        ))}
      </nav>
      <span className="grow" />
      {ai && (
        <span className={`engine-pill ${ai.enabled ? 'on' : 'off'}`} title="Models run locally inside the backend container">
          <span className="dot" />
          {ai.enabled ? `AI engine ready · ${ai.device?.toUpperCase()}` : 'AI engine off'}
        </span>
      )}
    </header>
  );
}
