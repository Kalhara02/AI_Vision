/**
 * The drawing surface. Everything is stored in image pixel coordinates; the
 * Konva stage applies the zoom/pan transform, so annotations never drift when
 * the view changes.
 *
 * Interaction:
 *   select tool   click to select, drag to move, handles to resize,
 *                 polygon: drag vertices, click a midpoint to insert a vertex,
 *                 right-click a vertex to delete it
 *   bbox tool     click-drag
 *   polygon tool  click to add points, click the first point or press Enter to close
 *   smart tool    click = SAM positive point, right/Shift click = negative point,
 *                 drag = SAM box prompt; dashed detector proposals: click accepts, right click rejects
 *   pan           hold Space and drag, or drag with the middle mouse button
 *   zoom          mouse wheel (around the cursor)
 */
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Circle, Image as KImage, Label, Layer, Line, Rect, Stage, Tag, Text, Transformer } from 'react-konva';
import { boxFromCorners, clamp, clipBox, dist, withAlpha, bboxFromPoints } from '../lib/geometry';
import { logger } from '../lib/logger';
import type { Annotation, BBox, Kind, LabelClass, Proposal, Pt, SmartPoint, Tool } from '../lib/types';

export interface NewShape {
  kind: Kind;
  bbox: BBox;
  points: Pt[] | null;
}

export interface CanvasHandle {
  closePolygon: () => boolean;
  popPoint: () => boolean;
  cancelDraft: () => boolean;
  fit: () => void;
  zoomBy: (factor: number) => void;
}

interface Props {
  imageUrl: string;
  width: number;
  height: number;
  annotations: Annotation[];
  classes: Map<number, LabelClass>;
  selected: string | null;
  tool: Tool;
  hidden: boolean;
  drawColor: string;
  onSelect: (uid: string | null) => void;
  onCreate: (shape: NewShape) => void;
  onChange: (uid: string, patch: Partial<Annotation>, record?: boolean) => void;
  onCheckpoint: () => void;
  // AI assistance (all optional so manual mode needs none of it)
  proposals?: Proposal[];
  onProposal?: (uid: string, accept: boolean) => void;
  smartPoints?: SmartPoint[];
  smartPreview?: Pt[] | null;
  onSmartClick?: (pt: Pt, positive: boolean) => void;
  onSmartBox?: (b: BBox) => void;
  showLabels?: boolean;
}

interface View {
  scale: number;
  x: number;
  y: number;
}

const MIN_BOX_SCREEN_PX = 4;
const HINT_BAR_PX = 48;
const CLOSE_RADIUS_SCREEN_PX = 10;

const AnnotationCanvas = forwardRef<CanvasHandle, Props>(function AnnotationCanvas(props, ref) {
  const { imageUrl, width: iw, height: ih, annotations, classes, selected, tool, hidden, drawColor } = props;
  const proposals = props.proposals ?? [];
  const smartPoints = props.smartPoints ?? [];
  const smartPreview = props.smartPreview ?? null;
  const labelText = (a: Annotation): string => {
    const score = a.score != null ? ` ${Math.round(a.score * 100)}%` : '';
    if (classes.size > 1) return `${nameOf(a)}${a.source !== 'manual' ? ' · AI' : ''}${score}`;
    return a.source !== 'manual' ? `AI${score}` : '';
  };
  const smartShift = useRef(false);

  const wrapRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);

  const [size, setSize] = useState({ w: 0, h: 0 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 });
  const [boxDraft, setBoxDraft] = useState<{ a: Pt; b: Pt } | null>(null);
  const [polyDraft, setPolyDraft] = useState<Pt[]>([]);
  const [cursor, setCursor] = useState<Pt | null>(null);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [panning, setPanning] = useState(false);
  const panStart = useRef<{ px: number; py: number; vx: number; vy: number } | null>(null);

  // ---------- sizing, image loading, fit ----------
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) =>
      setSize({ w: Math.floor(entry.contentRect.width), h: Math.floor(entry.contentRect.height) }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setImg(null);
    const im = new window.Image();
    im.onload = () => setImg(im);
    im.src = imageUrl;
    return () => {
      im.onload = null;
    };
  }, [imageUrl]);

  const fit = () => {
    if (!size.w || !size.h) return;
    // keep the bottom strip free for the shortcut bar so it never covers the image
    const availH = Math.max(1, size.h - HINT_BAR_PX);
    const s = Math.min(size.w / iw, availH / ih) * 0.96;
    setView({ scale: s, x: (size.w - iw * s) / 2, y: (availH - ih * s) / 2 });
  };

  useEffect(() => {
    if (img && size.w && size.h) fit();
    // fit once per image and when the panel is resized
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [img, size.w, size.h]);

  // drafts belong to the image and the tool they were started with
  useEffect(() => {
    setBoxDraft(null);
    setPolyDraft([]);
  }, [imageUrl, tool]);

  // ---------- transformer follows the selected box ----------
  useEffect(() => {
    const tr = trRef.current;
    const stage = stageRef.current;
    if (!tr || !stage) return;
    const sel = annotations.find((a) => a.uid === selected);
    const node = sel && sel.kind === 'bbox' && tool === 'select' && !hidden ? stage.findOne(`#${sel.uid}`) : undefined;
    tr.nodes(node ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selected, annotations, tool, hidden]);

  // ---------- space to pan ----------
  useEffect(() => {
    const isTyping = (e: KeyboardEvent) => (e.target as HTMLElement)?.closest?.('input, select, textarea');
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !isTyping(e)) {
        e.preventDefault();
        setSpaceHeld(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') setSpaceHeld(false);
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);

  // ---------- coordinate helpers ----------
  const toImage = (p: { x: number; y: number }): Pt => [
    clamp((p.x - view.x) / view.scale, 0, iw),
    clamp((p.y - view.y) / view.scale, 0, ih),
  ];
  const px = (screenPx: number) => screenPx / view.scale; // screen px -> image px

  // ---------- drafts ----------
  const closePolygon = () => {
    if (polyDraft.length < 3) return false;
    props.onCreate({ kind: 'polygon', points: polyDraft, bbox: bboxFromPoints(polyDraft) });
    logger.log('poly_closed', { n_points: polyDraft.length });
    setPolyDraft([]);
    return true;
  };

  const finishBox = () => {
    if (!boxDraft) return;
    const b = clipBox(boxFromCorners(boxDraft.a, boxDraft.b), iw, ih);
    setBoxDraft(null);
    if (tool === 'smart') {
      // a short press is a click (point prompt), a drag is a box prompt
      if (b[2] * view.scale < 6 && b[3] * view.scale < 6) props.onSmartClick?.(boxDraft.a, !smartShift.current);
      else props.onSmartBox?.(b);
      return;
    }
    if (b[2] * view.scale >= MIN_BOX_SCREEN_PX && b[3] * view.scale >= MIN_BOX_SCREEN_PX) {
      props.onCreate({ kind: 'bbox', bbox: b, points: null });
    } else {
      logger.log('box_too_small');
    }
  };

  useImperativeHandle(ref, () => ({
    closePolygon,
    popPoint: () => {
      if (!polyDraft.length) return false;
      setPolyDraft((d) => d.slice(0, -1));
      logger.log('poly_point_removed');
      return true;
    },
    cancelDraft: () => {
      const had = polyDraft.length > 0 || boxDraft !== null;
      setPolyDraft([]);
      setBoxDraft(null);
      if (had) logger.log('draft_cancelled');
      return had;
    },
    fit,
    zoomBy: (factor: number) => {
      const c = { x: size.w / 2, y: size.h / 2 };
      zoomAt(c, factor);
    },
  }));

  // ---------- mouse ----------
  const zoomAt = (p: { x: number; y: number }, factor: number) => {
    setView((v) => {
      const minS = Math.min(size.w / iw, size.h / ih) * 0.2;
      const s = clamp(v.scale * factor, minS, 50);
      const mx = (p.x - v.x) / v.scale;
      const my = (p.y - v.y) / v.scale;
      return { scale: s, x: p.x - mx * s, y: p.y - my * s };
    });
  };

  const onWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const p = stageRef.current?.getPointerPosition();
    if (p) zoomAt(p, e.evt.deltaY > 0 ? 1 / 1.15 : 1.15);
  };

  const onDown = (e: KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    const p = stage?.getPointerPosition();
    if (!stage || !p) return;
    logger.log('pointer_down', { tool, button: e.evt.button });

    if (e.evt.button === 1 || (e.evt.button === 0 && spaceHeld)) {
      e.evt.preventDefault();
      panStart.current = { px: p.x, py: p.y, vx: view.x, vy: view.y };
      setPanning(true);
      return;
    }
    if (tool === 'smart' && e.evt.button === 2) {
      props.onSmartClick?.(toImage(p), false);
      return;
    }
    if (e.evt.button !== 0) return;
    const ip = toImage(p);

    if (tool === 'smart') {
      smartShift.current = e.evt.shiftKey;
      setBoxDraft({ a: ip, b: ip });
      return;
    }

    if (tool === 'bbox') {
      setBoxDraft({ a: ip, b: ip });
      return;
    }
    if (tool === 'polygon') {
      if (polyDraft.length >= 3 && dist(polyDraft[0], ip) * view.scale < CLOSE_RADIUS_SCREEN_PX) {
        closePolygon();
      } else {
        setPolyDraft((d) => [...d, ip]);
        logger.log('poly_point', { n: polyDraft.length + 1 });
      }
      return;
    }
    // select tool: clicking empty space clears the selection
    if (e.target === stage) props.onSelect(null);
  };

  const onMove = () => {
    const p = stageRef.current?.getPointerPosition();
    if (!p) return;
    if (panning && panStart.current) {
      const s = panStart.current;
      setView((v) => ({ ...v, x: s.vx + (p.x - s.px), y: s.vy + (p.y - s.py) }));
      return;
    }
    const ip = toImage(p);
    setCursor(ip);
    if (boxDraft) setBoxDraft({ a: boxDraft.a, b: ip });
  };

  const onUp = () => {
    if (panning) {
      setPanning(false);
      panStart.current = null;
      logger.log('pan');
      return;
    }
    finishBox();
  };

  // finish a box or pan even if the button is released outside the canvas
  const upRef = useRef(onUp);
  upRef.current = onUp;
  useEffect(() => {
    const h = () => upRef.current();
    window.addEventListener('mouseup', h);
    return () => window.removeEventListener('mouseup', h);
  }, []);

  // ---------- rendering helpers ----------
  const editable = tool === 'select' && !spaceHeld;
  const colorOf = (a: Annotation) => classes.get(a.class_id)?.color ?? '#ffffff';
  const nameOf = (a: Annotation) => classes.get(a.class_id)?.name ?? '?';
  const sel = annotations.find((a) => a.uid === selected) ?? null;

  const cursorStyle = panning ? 'grabbing' : spaceHeld ? 'grab' : tool === 'select' ? 'default' : 'crosshair';

  const updatePolyPoints = (a: Annotation, points: Pt[], record: boolean) =>
    props.onChange(a.uid, { points, bbox: bboxFromPoints(points) }, record);

  return (
    <div ref={wrapRef} className="canvas-wrap" style={{ cursor: cursorStyle }} onContextMenu={(e) => e.preventDefault()}>
      {size.w > 0 && (
        <Stage
          ref={stageRef}
          width={size.w}
          height={size.h}
          x={view.x}
          y={view.y}
          scaleX={view.scale}
          scaleY={view.scale}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={() => setCursor(null)}
          onWheel={onWheel}
        >
          <Layer listening={false}>
            {img && <KImage image={img} width={iw} height={ih} />}
          </Layer>

          <Layer visible={!hidden}>
            {/* detector proposals: dashed, not saved until accepted */}
            {proposals.map((q) => (
              <Rect
                key={q.uid}
                x={q.bbox[0]}
                y={q.bbox[1]}
                width={q.bbox[2]}
                height={q.bbox[3]}
                stroke="#ffd400"
                strokeWidth={1.5}
                strokeScaleEnabled={false}
                dash={[6, 4]}
                fill="rgba(255, 212, 0, 0.07)"
                listening={(tool === 'select' || tool === 'smart') && !spaceHeld}
                onMouseDown={(e) => {
                  e.cancelBubble = true;
                  if (e.evt.button === 0) props.onProposal?.(q.uid, true);
                  else if (e.evt.button === 2) props.onProposal?.(q.uid, false);
                }}
              />
            ))}
            {proposals.map((q) => (
              <Label key={`pl${q.uid}`} x={q.bbox[0]} y={q.bbox[1] - px(14)} listening={false} scaleX={1 / view.scale} scaleY={1 / view.scale}>
                <Tag fill="#ffd400" opacity={0.85} />
                <Text text={`${Math.round(q.score * 100)}%`} fontSize={10} padding={1} fill="#111" />
              </Label>
            ))}

            {annotations.map((a) => {
              const color = colorOf(a);
              const isSel = a.uid === selected;
              const proposal = a.source !== 'manual';
              const common = {
                stroke: color,
                strokeWidth: isSel ? 3 : 2,
                strokeScaleEnabled: false,
                dash: a.source === 'detector' ? [6, 4] : undefined,
                fill: withAlpha(color, isSel ? 0.28 : proposal ? 0.08 : 0.15),
                listening: editable,
                onMouseDown: (e: KonvaEventObject<MouseEvent>) => {
                  if (e.evt.button !== 0) return;
                  if (!isSel) logger.log('select', { kind: a.kind, source: a.source });
                  props.onSelect(a.uid);
                },
              };
              if (a.kind === 'bbox') {
                return (
                  <Rect
                    key={a.uid}
                    id={a.uid}
                    x={a.bbox[0]}
                    y={a.bbox[1]}
                    width={a.bbox[2]}
                    height={a.bbox[3]}
                    draggable={editable && isSel}
                    onDragEnd={(e) => {
                      const n = e.target;
                      props.onChange(a.uid, { bbox: clipBox([n.x(), n.y(), a.bbox[2], a.bbox[3]], iw, ih) });
                      logger.log('ann_moved', { kind: 'bbox' });
                    }}
                    onTransformEnd={(e) => {
                      const n = e.target;
                      const b: BBox = [n.x(), n.y(), n.width() * n.scaleX(), n.height() * n.scaleY()];
                      n.scaleX(1);
                      n.scaleY(1);
                      props.onChange(a.uid, { bbox: clipBox(b, iw, ih) });
                      logger.log('ann_resized', { kind: 'bbox' });
                    }}
                    {...common}
                  />
                );
              }
              const pts = a.points ?? [];
              return (
                <Line
                  key={a.uid}
                  id={a.uid}
                  points={pts.flat()}
                  closed
                  draggable={editable && isSel}
                  onDragEnd={(e) => {
                    const n = e.target;
                    const dx = n.x();
                    const dy = n.y();
                    n.position({ x: 0, y: 0 });
                    const moved = pts.map(([x, y]) => [clamp(x + dx, 0, iw), clamp(y + dy, 0, ih)] as Pt);
                    updatePolyPoints(a, moved, true);
                    logger.log('ann_moved', { kind: 'polygon' });
                  }}
                  {...common}
                />
              );
            })}

            {/* vertex + midpoint handles for the selected polygon */}
            {sel && sel.kind === 'polygon' && editable && sel.points && (
              <>
                {sel.points.map((p, i) => {
                  const q = sel.points![(i + 1) % sel.points!.length];
                  return (
                    <Circle
                      key={`m${i}`}
                      x={(p[0] + q[0]) / 2}
                      y={(p[1] + q[1]) / 2}
                      radius={px(4)}
                      fill={withAlpha('#ffffff', 0.5)}
                      stroke={colorOf(sel)}
                      strokeWidth={1}
                      strokeScaleEnabled={false}
                      onMouseDown={(e) => {
                        e.cancelBubble = true;
                        const next = [...sel.points!];
                        next.splice(i + 1, 0, [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]);
                        updatePolyPoints(sel, next, true);
                        logger.log('vertex_added');
                      }}
                    />
                  );
                })}
                {sel.points.map((p, i) => (
                  <Circle
                    key={`v${i}`}
                    x={p[0]}
                    y={p[1]}
                    radius={px(5)}
                    fill="#ffffff"
                    stroke={colorOf(sel)}
                    strokeWidth={2}
                    strokeScaleEnabled={false}
                    draggable
                    onMouseDown={(e) => {
                      e.cancelBubble = true;
                      if (e.evt.button === 2 && sel.points!.length > 3) {
                        updatePolyPoints(sel, sel.points!.filter((_, j) => j !== i), true);
                        logger.log('vertex_removed');
                      }
                    }}
                    onDragStart={() => props.onCheckpoint()}
                    onDragMove={(e) => {
                      const n = e.target;
                      const next = [...sel.points!];
                      next[i] = [clamp(n.x(), 0, iw), clamp(n.y(), 0, ih)];
                      updatePolyPoints(sel, next, false);
                    }}
                    onDragEnd={() => logger.log('vertex_moved')}
                  />
                ))}
              </>
            )}

            {/* class labels: with one class the colour already says the class, so only AI
                annotations get a short tag (dense images stay readable); L toggles them */}
            {props.showLabels !== false && annotations.map((a) => {
              const text = labelText(a);
              if (!text) return null; // react-konva cannot render a bare '' string
              return (
              // labels are drawn in screen pixels (inverse scale) so they stay readable at any zoom
              <Label
                key={`l${a.uid}`}
                x={a.bbox[0]}
                y={a.bbox[1] - px(16)}
                listening={false}
                scaleX={1 / view.scale}
                scaleY={1 / view.scale}
              >
                <Tag fill={colorOf(a)} />
                <Text text={text} fontSize={classes.size > 1 ? 11 : 9} padding={classes.size > 1 ? 2 : 1} fill="#111" />
              </Label>
              );
            })}

            <Transformer
              ref={trRef}
              rotateEnabled={false}
              flipEnabled={false}
              keepRatio={false}
              ignoreStroke
              anchorSize={8}
              borderStroke="#ffffff"
              anchorStroke="#333"
              boundBoxFunc={(oldBox, newBox) => (newBox.width < 3 || newBox.height < 3 ? oldBox : newBox)}
            />
          </Layer>

          {/* drafts and guides */}
          <Layer listening={false}>
            {tool !== 'select' && cursor && !spaceHeld && (
              <>
                <Line points={[cursor[0], 0, cursor[0], ih]} stroke="#ffffff" opacity={0.45} strokeWidth={1} strokeScaleEnabled={false} dash={[4, 4]} />
                <Line points={[0, cursor[1], iw, cursor[1]]} stroke="#ffffff" opacity={0.45} strokeWidth={1} strokeScaleEnabled={false} dash={[4, 4]} />
              </>
            )}
            {/* SAM preview + its clicks */}
            {smartPreview && smartPreview.length >= 3 && (
              <Line
                points={smartPreview.flat()}
                closed
                stroke={drawColor}
                strokeWidth={2}
                strokeScaleEnabled={false}
                fill={withAlpha(drawColor, 0.35)}
              />
            )}
            {smartPoints.map(([x, y, l], i) => (
              <Circle
                key={`sp${i}`}
                x={x}
                y={y}
                radius={px(5)}
                fill={l === 1 ? '#22c55e' : '#ef4444'}
                stroke="#ffffff"
                strokeWidth={1.5}
                strokeScaleEnabled={false}
              />
            ))}
            {boxDraft && (
              <Rect
                x={Math.min(boxDraft.a[0], boxDraft.b[0])}
                y={Math.min(boxDraft.a[1], boxDraft.b[1])}
                width={Math.abs(boxDraft.a[0] - boxDraft.b[0])}
                height={Math.abs(boxDraft.a[1] - boxDraft.b[1])}
                stroke={drawColor}
                strokeWidth={2}
                strokeScaleEnabled={false}
                fill={withAlpha(drawColor, 0.15)}
              />
            )}
            {polyDraft.length > 0 && (
              <>
                <Line
                  points={[...polyDraft.flat(), ...(cursor ?? [])]}
                  stroke={drawColor}
                  strokeWidth={2}
                  strokeScaleEnabled={false}
                  fill={withAlpha(drawColor, 0.12)}
                  closed={polyDraft.length >= 2}
                />
                {polyDraft.map((p, i) => (
                  <Circle
                    key={i}
                    x={p[0]}
                    y={p[1]}
                    radius={px(i === 0 && polyDraft.length >= 3 ? 7 : 4)}
                    fill={i === 0 ? drawColor : '#ffffff'}
                    stroke={drawColor}
                    strokeWidth={1}
                    strokeScaleEnabled={false}
                  />
                ))}
              </>
            )}
          </Layer>
        </Stage>
      )}
      <div className="zoom-badge">{Math.round(view.scale * 100)}%{cursor ? `  ·  ${Math.round(cursor[0])}, ${Math.round(cursor[1])}` : ''}</div>
      {!img && <div className="canvas-loading">Loading image…</div>}
    </div>
  );
});

export default AnnotationCanvas;
