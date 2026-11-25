/** Small inline SVG icon set (drawn for this project, stroke style), plus the LabelLoop logo. */
import type { ReactNode } from 'react';

const P: Record<string, ReactNode> = {
  cursor: <path d="M5 3l13 7.5-5.5 1.6L10 18z" />,
  box: <rect x="4" y="5" width="16" height="14" rx="2" />,
  polygon: <path d="M12 3.5l8 5.8-3 10.2H7L4 9.3z" />,
  sparkle: (
    <>
      <path d="M11 3l1.9 5.1L18 10l-5.1 1.9L11 17l-1.9-5.1L4 10l5.1-1.9z" />
      <path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z" />
    </>
  ),
  undo: <path d="M9 14L4 9l5-5M4 9h10.5a5.5 5.5 0 010 11H11" />,
  redo: <path d="M15 14l5-5-5-5M20 9H9.5a5.5 5.5 0 000 11H13" />,
  fit: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
  eye: (
    <>
      <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  eyeOff: (
    <>
      <path d="M2 12s3.6-7 10-7c2 0 3.7.7 5.1 1.6M22 12s-3.6 7-10 7c-2 0-3.7-.7-5.1-1.6" />
      <path d="M4 4l16 16" />
    </>
  ),
  help: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.6 9.3a2.5 2.5 0 014.8.9c0 1.7-2.4 2.2-2.4 3.8M12 17.2v.1" />
    </>
  ),
  arrowLeft: <path d="M19 12H5M11 6l-6 6 6 6" />,
  chevronLeft: <path d="M15 6l-6 6 6 6" />,
  chevronRight: <path d="M9 6l6 6-6 6" />,
  check: <path d="M5 12.5l4.5 4.5L19 7" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  upload: <path d="M12 15V4M7.5 8.5L12 4l4.5 4.5M4 15v4a1 1 0 001 1h14a1 1 0 001-1v-4" />,
  download: <path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M4 15v4a1 1 0 001 1h14a1 1 0 001-1v-4" />,
  zap: <path d="M13 2L4.5 13.5H11L10 22l8.5-11.5H12z" />,
  cpu: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <path d="M9 2.5V6M15 2.5V6M9 18v3.5M15 18v3.5M2.5 9H6M2.5 15H6M18 9h3.5M18 15h3.5" />
    </>
  ),
  image: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="8.5" cy="9.5" r="1.8" />
      <path d="M21 16l-5-5-9 9" />
    </>
  ),
  tag: (
    <>
      <path d="M3 12V4a1 1 0 011-1h8l9 9-9 9z" />
      <circle cx="7.5" cy="7.5" r="1.5" />
    </>
  ),
  chart: <path d="M4 20V11M10 20V5M16 20v-6M2.5 20h19" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="12" cy="12" r="4.5" />
      <circle cx="12" cy="12" r="1" />
    </>
  ),
  layers: <path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5" />,
  trash: <path d="M4 7h16M9 7V4h6v3M6.5 7l1 13h9l1-13" />,
  plus: <path d="M12 5v14M5 12h14" />,
  user: (
    <>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21a8 8 0 0116 0" />
    </>
  ),
  shuffle: <path d="M3 7h4l10 10h4M3 17h4l3-3M14 10l3-3h4M18 4l3 3-3 3M18 14l3 3-3 3" />,
  keyboard: (
    <>
      <rect x="2.5" y="6" width="19" height="12" rx="2" />
      <path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10" />
    </>
  ),
  loop: <path d="M17 2.5l3 3-3 3M4 11.5v-1a5 5 0 015-5h11M7 21.5l-3-3 3-3M20 12.5v1a5 5 0 01-5 5H4" />,
};

export type IconName = keyof typeof P;

export function Icon({ name, size = 18, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      className={`icon-svg ${className ?? ''}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {P[name]}
    </svg>
  );
}

/** Logo: a labeled object (box with a corner handle) inside a loop. */
export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-label="LabelLoop">
      <defs>
        <linearGradient id="ll-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#5b8cff" />
          <stop offset="1" stopColor="#9b6bff" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="url(#ll-grad)" />
      <path d="M22.5 9.5A8.5 8.5 0 108.2 19.6" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
      <path d="M22.8 5.8l.2 4.4-4.3.3" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="12.5" y="12.5" width="9" height="9" rx="1.5" fill="none" stroke="#fff" strokeWidth="1.8" strokeDasharray="2.4 1.6" />
      <circle cx="21.5" cy="21.5" r="1.9" fill="#fff" />
    </svg>
  );
}

/** Circular progress (0..1). */
export function Ring({ value, size = 44, stroke = 5 }: { value: number; size?: number; stroke?: number }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value || 0));
  return (
    <svg width={size} height={size} className="ring">
      <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--surface-3)" strokeWidth={stroke} fill="none" />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        stroke="var(--ok)"
        strokeWidth={stroke}
        fill="none"
        strokeDasharray={`${c * v} ${c}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="50%" dominantBaseline="central" textAnchor="middle" fontSize={size * 0.26} fill="var(--text)" fontWeight={600}>
        {Math.round(v * 100)}%
      </text>
    </svg>
  );
}
