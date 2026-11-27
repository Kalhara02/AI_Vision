import type { TrainingRound } from '../lib/types';

/**
 * One pair of bars per training round: the model in use before the round and the newly trained one,
 * both scored on that round's held-out images. Rounds use different held-out sets (more images are
 * done each time), so only the two bars inside a pair are comparable, not bars across rounds.
 */
export default function ModelChart({ rounds }: { rounds: TrainingRound[] }) {
  const scored = rounds.filter((r) => r.ap50_new != null).slice(-5);
  if (!scored.length) return null;

  const W = 260;
  const H = 104;
  const top = 16;
  const bottom = 26;
  const pairGap = 14;
  const pw = Math.min(56, (W - pairGap * (scored.length - 1)) / scored.length);
  const bw = (pw - 4) / 2;
  const scale = (v: number) => (H - top - bottom) * Math.max(0, Math.min(1, v));

  return (
    <div className="model-chart">
      <div className="model-chart-title">
        <span>AP50 before vs after each round</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Detector AP50 before and after each training round">
        <line x1={0} x2={W} y1={H - bottom} y2={H - bottom} stroke="var(--line)" />
        {scored.map((r, i) => {
          const x0 = i * (pw + pairGap);
          const before = r.ap50_previous ?? 0;
          const after = r.ap50_new ?? 0;
          const delta = after - before;
          const bars = [
            { v: before, fill: 'var(--accent)', op: 0.6 },
            { v: after, fill: r.accepted ? 'var(--ok)' : 'var(--surface-3)', op: 1 },
          ];
          return (
            <g key={r.round}>
              {bars.map((b, j) => {
                const x = x0 + j * (bw + 4);
                const h = scale(b.v);
                return (
                  <g key={j}>
                    <rect x={x} y={H - bottom - h} width={bw} height={h} rx={3} fill={b.fill} opacity={b.op} />
                    <text x={x + bw / 2} y={H - bottom - h - 4} textAnchor="middle" fontSize={8.5} fill="var(--text)">
                      {b.v.toFixed(2)}
                    </text>
                  </g>
                );
              })}
              <text x={x0 + pw / 2} y={H - 14} textAnchor="middle" fontSize={9} fill="var(--muted)">
                {`R${r.round} · n=${r.n_holdout}`}
              </text>
              <text x={x0 + pw / 2} y={H - 3} textAnchor="middle" fontSize={9} fill={delta >= 0 ? 'var(--ok)' : 'var(--danger, #e5484d)'}>
                {`${delta >= 0 ? '+' : ''}${delta.toFixed(2)}`}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend small muted">
        <span><i className="lg base" />before</span>
        <span><i className="lg ok" />new, deployed</span>
        <span><i className="lg kept" />new, rejected</span>
      </div>
    </div>
  );
}
