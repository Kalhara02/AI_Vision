import type { TrainingRound } from '../lib/types';

/**
 * AP50 on held-out images per training round. The first bar is the model that was in use before
 * round 1 (usually the pretrained one); each later bar is the newly trained model, green when it
 * was deployed, grey when the guard kept the old one.
 */
export default function ModelChart({ rounds }: { rounds: TrainingRound[] }) {
  const scored = rounds.filter((r) => r.ap50_new != null);
  if (!scored.length) return null;
  const bars = [
    { label: `v${scored[0].previous_version ?? 0}`, value: scored[0].ap50_previous ?? 0, kind: 'base' },
    ...scored.map((r) => ({
      label: r.accepted ? `v${r.version}` : `r${r.round}`,
      value: r.ap50_new ?? 0,
      kind: r.accepted ? 'ok' : 'kept',
    })),
  ].slice(-8);

  const W = 260;
  const H = 92;
  const top = 14;
  const bottom = 18;
  const gap = 6;
  const bw = Math.min(34, (W - gap * (bars.length - 1)) / bars.length);
  const scale = (v: number) => (H - top - bottom) * Math.max(0, Math.min(1, v));

  return (
    <div className="model-chart">
      <div className="model-chart-title">
        <span>AP50 on held-out images per round</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Detector AP50 per training round">
        <line x1={0} x2={W} y1={H - bottom} y2={H - bottom} stroke="var(--line)" />
        {bars.map((b, i) => {
          const x = i * (bw + gap);
          const h = scale(b.value);
          const fill = b.kind === 'ok' ? 'var(--ok)' : b.kind === 'kept' ? 'var(--surface-3)' : 'var(--accent)';
          return (
            <g key={i}>
              <rect x={x} y={H - bottom - h} width={bw} height={h} rx={3} fill={fill} opacity={b.kind === 'base' ? 0.75 : 1} />
              <text x={x + bw / 2} y={H - bottom - h - 4} textAnchor="middle" fontSize={9} fill="var(--text)">
                {b.value.toFixed(2)}
              </text>
              <text x={x + bw / 2} y={H - 5} textAnchor="middle" fontSize={9} fill="var(--muted)">
                {b.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend small muted">
        <span><i className="lg base" />before</span>
        <span><i className="lg ok" />deployed</span>
        <span><i className="lg kept" />rejected by guard</span>
      </div>
    </div>
  );
}
