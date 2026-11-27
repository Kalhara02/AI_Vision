import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../lib/api';
import { bboxFromPoints, bboxIou, fromDTO, newUid, toDTO } from '../lib/geometry';
import { logger } from '../lib/logger';
import { navigate } from '../lib/router';
import type {
  Annotation,
  BBox,
  ImageMeta,
  ImageStatus,
  ModelStatus,
  Project,
  Proposal,
  Pt,
  SamMode,
  SmartPoint,
  Source,
  Tool,
} from '../lib/types';
import { useHistory } from '../lib/useHistory';
import AnnotationCanvas, { type CanvasHandle, type NewShape } from './AnnotationCanvas';
import { Icon, type IconName } from './Icons';
import ModelChart from './ModelChart';

type SaveState = 'saved' | 'dirty' | 'saving' | 'error';

const TOOL_ICON: Record<Tool, IconName> = { select: 'cursor', bbox: 'box', polygon: 'polygon', smart: 'sparkle' };
const TOOL_SHORT: Record<Tool, string> = { select: 'Edit', bbox: 'Box', polygon: 'Polygon', smart: 'Smart' };

const TOOLS: { id: Tool; label: string; key: string }[] = [
  { id: 'select', label: 'Select / edit', key: 'V' },
  { id: 'bbox', label: 'Box', key: 'B' },
  { id: 'polygon', label: 'Polygon', key: 'P' },
  { id: 'smart', label: 'Smart (SAM): click, right-click, drag a box', key: 'S' },
];

const SHORTCUTS: [string, string][] = [
  ['V / B / P / S', 'select, box, polygon, smart (SAM) tool'],
  ['Smart: click / right-click', 'add object / background point, drag = box prompt'],
  ['Smart: Enter', 'accept the SAM outline'],
  ['click / right-click proposal', 'accept / reject one detector suggestion'],
  ['A', 'accept all detector suggestions'],
  ['R', 'boxes to polygons with SAM (selected box, or all boxes)'],
  ['1 – 9', 'choose class (also re-labels the selected object)'],
  ['Enter', 'close polygon'],
  ['Backspace', 'remove last polygon point, otherwise delete selected'],
  ['Delete', 'delete selected'],
  ['Esc', 'cancel drawing / deselect'],
  ['Tab / Shift+Tab', 'select next / previous object'],
  ['Ctrl+Z / Ctrl+Shift+Z', 'undo / redo'],
  ['Ctrl+S', 'save now'],
  ['Ctrl+Enter', 'mark image done and go to next'],
  ['← / →', 'previous / next image'],
  ['Space + drag, middle drag', 'pan'],
  ['Wheel, + / −', 'zoom'],
  ['F', 'fit image to screen'],
  ['H', 'hide / show annotations'],
  ['L', 'hide / show labels'],
  ['?', 'this help'],
];

export default function AnnotatePage({ projectId, imageId }: { projectId: number; imageId: number }) {
  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [meta, setMeta] = useState<ImageMeta | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tool, setToolState] = useState<Tool>(logger.mode === 'assisted' ? 'smart' : 'bbox');
  const [activeClassId, setActiveClassId] = useState<number | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hidden, setHidden] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [saveState, setSaveState] = useState<SaveState>('saved');
  const [showHelp, setShowHelp] = useState(false);

  // ---------- AI assistance state ----------
  const assisted = logger.mode === 'assisted';
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [modelVersion, setModelVersion] = useState<number | null>(null);
  const [zeroShotClasses, setZeroShotClasses] = useState<string[]>([]);
  const [threshold, setThreshold] = useState(0.5);
  const [smartPoints, setSmartPoints] = useState<SmartPoint[]>([]);
  const [smartBox, setSmartBox] = useState<BBox | null>(null);
  const [smartPreview, setSmartPreview] = useState<Pt[] | null>(null);
  const [aiBusy, setAiBusy] = useState<string | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [epochs, setEpochs] = useState(10);
  const samReq = useRef(0);
  const [samMode, setSamModeState] = useState<SamMode>(() => {
    const m = localStorage.getItem('labelloop.samMode');
    return m === 'focus' || m === 'hybrid' ? m : 'plain';
  });
  const setSamMode = (m: SamMode) => {
    setSamModeState(m);
    localStorage.setItem('labelloop.samMode', m);
    logger.log('sam_mode', { mode: m });
  };
  const thresholdRef = useRef(threshold);
  thresholdRef.current = threshold;

  const history = useHistory<Annotation[]>([]);
  const anns = history.present;
  const canvasRef = useRef<CanvasHandle>(null);

  // refs mirror state for async code (saving, cleanup)
  const savedRef = useRef<Annotation[]>([]);
  const presentRef = useRef(anns);
  presentRef.current = anns;
  const currentIdRef = useRef(imageId);
  currentIdRef.current = imageId;
  const loadedRef = useRef(false);
  const openedAt = useRef(Date.now());
  const saveChain = useRef<Promise<void>>(Promise.resolve());

  const classMap = useMemo(() => new Map((project?.classes ?? []).map((c) => [c.id, c])), [project]);
  const activeClass = activeClassId != null ? classMap.get(activeClassId) : undefined;
  const index = images.findIndex((i) => i.id === imageId);

  // ---------- loading ----------
  useEffect(() => {
    Promise.all([api.getProject(projectId), api.listImages(projectId)])
      .then(([p, ims]) => {
        setProject(p);
        setImages(ims);
        setActiveClassId((cur) => cur ?? p.classes[0]?.id ?? null);
      })
      .catch((e) => setError((e as Error).message));
  }, [projectId]);

  const persist = useCallback((id: number, list: Annotation[], status?: ImageStatus) => {
    saveChain.current = saveChain.current.then(async () => {
      setSaveState('saving');
      try {
        const res = await api.saveAnnotations(id, list.map(toDTO), status);
        if (id === currentIdRef.current) {
          savedRef.current = list;
          setMeta(res.image);
        }
        setImages((ims) => ims.map((im) => (im.id === id ? res.image : im)));
        setSaveState(presentRef.current === list || id !== currentIdRef.current ? 'saved' : 'dirty');
      } catch (e) {
        setSaveState('error');
        setError(`Save failed: ${(e as Error).message}`);
      }
    });
    return saveChain.current;
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadedRef.current = false;
    setLoaded(false);
    setSelected(null);
    setProposals([]);
    samReq.current++;
    setSmartPoints([]);
    setSmartBox(null);
    setSmartPreview(null);
    logger.setContext(projectId, imageId);
    api
      .getAnnotations(imageId)
      .then(({ image, annotations }) => {
        if (cancelled) return;
        const list = annotations.map(fromDTO);
        savedRef.current = list;
        history.reset(list);
        setMeta(image);
        setSaveState('saved');
        loadedRef.current = true;
        setLoaded(true);
        openedAt.current = Date.now();
        logger.log('image_open', { n_existing: list.length, status: image.status });
        if (logger.mode === 'assisted') {
          api.samPrepare(imageId).catch(() => {}); // warm up the encoder so the first click is fast
          void loadProposals(imageId, list, () => cancelled);
        }
      })
      .catch((e) => setError((e as Error).message));

    return () => {
      cancelled = true;
      // leaving this image (navigation, back button): log and save pending edits
      if (loadedRef.current) {
        logger.log('image_leave', { dwell_ms: Date.now() - openedAt.current, n_annotations: presentRef.current.length }, );
        if (presentRef.current !== savedRef.current) void persist(imageId, presentRef.current);
      }
    };
    // history.reset is stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageId, projectId, persist]);

  // ---------- autosave ----------
  useEffect(() => {
    if (!loadedRef.current || anns === savedRef.current) return;
    setSaveState('dirty');
    const t = window.setTimeout(() => void persist(imageId, anns), 800);
    return () => window.clearTimeout(t);
  }, [anns, imageId, persist]);

  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (presentRef.current !== savedRef.current) e.preventDefault();
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, []);

  // ---------- editing ----------
  const setTool = (t: Tool) => {
    if (t !== tool) logger.log('tool_change', { tool: t });
    setToolState(t);
    if (t !== 'select') setSelected(null);
  };

  const create = (shape: NewShape, source: Source = 'manual', score: number | null = null, classId?: number) => {
    if (!activeClass && classId == null) {
      setError('Create a class first (project page).');
      return;
    }
    const a: Annotation = { uid: newUid(), class_id: classId ?? activeClass!.id, source, score, ...shape };
    history.set((prev) => [...prev, a]);
    logger.log('ann_created', { kind: a.kind, source: a.source, n_points: a.points?.length ?? 4 });
  };

  const change = (uid: string, patch: Partial<Annotation>, record = true) =>
    history.set((prev) => prev.map((a) => (a.uid === uid ? { ...a, ...patch } : a)), record);

  const remove = (uid: string) => {
    const a = anns.find((x) => x.uid === uid);
    history.set((prev) => prev.filter((x) => x.uid !== uid));
    if (selected === uid) setSelected(null);
    logger.log('ann_deleted', { kind: a?.kind, source: a?.source });
  };

  const relabel = (uid: string, classId: number) => {
    const a = anns.find((x) => x.uid === uid);
    if (!a || a.class_id === classId) return;
    change(uid, { class_id: classId });
    logger.log('class_changed', { source: a.source });
  };

  const chooseClass = (classId: number) => {
    setActiveClassId(classId);
    if (selected) relabel(selected, classId);
  };

  // ---------- AI: detector proposals ----------
  async function loadProposals(id: number, existing: Annotation[], isCancelled: () => boolean = () => false) {
    try {
      const r = await api.proposals(id, thresholdRef.current);
      if (isCancelled() || id !== currentIdRef.current) return;
      // skip suggestions for objects that are already labeled (revisiting an image)
      const fresh = r.proposals.filter((q) => !existing.some((a) => bboxIou(a.bbox, q.bbox) > 0.5));
      setProposals(fresh.map((q) => ({ uid: newUid(), bbox: q.bbox, score: q.score, class_id: q.class_id, source: q.source })));
      setModelVersion(r.version);
      setZeroShotClasses(r.zero_shot_classes ?? []);
      logger.log('proposals_shown', {
        n: fresh.length, version: r.version, threshold: thresholdRef.current, ms: r.ms,
        n_zero_shot: fresh.filter((q) => q.source === 'zero-shot').length,
      });
    } catch (e) {
      setError(`Detector: ${(e as Error).message}`);
    }
  }

  const onProposal = (uid: string, accept: boolean) => {
    const q = proposals.find((x) => x.uid === uid);
    if (!q) return;
    setProposals((ps) => ps.filter((x) => x.uid !== uid));
    if (accept) {
      create({ kind: 'bbox', bbox: q.bbox, points: null }, 'detector', q.score, q.class_id);
      logger.log('proposal_accepted', { score: q.score, version: modelVersion, class_id: q.class_id, source: q.source });
    } else {
      logger.log('proposal_rejected', { score: q.score, version: modelVersion, class_id: q.class_id, source: q.source });
    }
  };

  const acceptAll = () => {
    if (!proposals.length) return;
    const added: Annotation[] = proposals.map((q) => ({
      uid: newUid(), class_id: q.class_id, kind: 'bbox', bbox: q.bbox, points: null, source: 'detector', score: q.score,
    }));
    history.set((prev) => [...prev, ...added]);
    logger.log('proposals_accept_all', { n: added.length, version: modelVersion });
    setProposals([]);
  };

  // ---------- AI: SAM ----------
  const clearSmart = () => {
    samReq.current++;
    setSmartPoints([]);
    setSmartBox(null);
    setSmartPreview(null);
  };

  const runSam = async (points: SmartPoint[], box: BBox | null) => {
    const req = ++samReq.current;
    if (!points.length && !box) {
      setSmartPreview(null);
      return;
    }
    const t0 = performance.now();
    setAiBusy('SAM…');
    try {
      const r = await api.samPredict(imageId, points, box, samMode);
      if (req !== samReq.current) return; // a newer click already went out
      setSmartPreview(r.points);
      logger.log('sam_predict', {
        n_points: points.length, box: !!box, score: r.score, sam_mode: samMode, focus_used: !!r.focus_window,
        latency_ms: Math.round(performance.now() - t0), encode_ms: r.encode_ms, decode_ms: r.decode_ms,
      });
    } catch (e) {
      setError(`SAM: ${(e as Error).message}`);
    } finally {
      if (req === samReq.current) setAiBusy(null);
    }
  };

  const onSmartClick = (pt: Pt, positive: boolean) => {
    const next: SmartPoint[] = [...smartPoints, [pt[0], pt[1], positive ? 1 : 0]];
    setSmartPoints(next);
    logger.log('sam_click', { positive });
    void runSam(next, smartBox);
  };

  const onSmartBox = (b: BBox) => {
    setSmartBox(b);
    setSmartPoints([]);
    logger.log('sam_box');
    void runSam([], b);
  };

  const acceptSmart = () => {
    if (!smartPreview || smartPreview.length < 3) return false;
    create({ kind: 'polygon', points: smartPreview, bbox: bboxFromPoints(smartPreview) }, 'sam');
    logger.log('sam_accept', { clicks: smartPoints.length, box: !!smartBox, sam_mode: samMode });
    clearSmart();
    return true;
  };

  /** Boxes -> polygons with SAM's box prompt: the selected box, or every box on the image. */
  const refineBoxes = async () => {
    const sel = anns.find((a) => a.uid === selected && a.kind === 'bbox');
    const targets = sel ? [sel] : anns.filter((a) => a.kind === 'bbox');
    if (!targets.length) return;
    const t0 = performance.now();
    setAiBusy(`Refining ${targets.length} box(es)…`);
    try {
      const r = await api.samBoxes(imageId, targets.map((a) => a.bbox));
      const byUid = new Map<string, Pt[]>();
      targets.forEach((a, i) => {
        const poly = r.polygons[i];
        if (poly && poly.length >= 3) byUid.set(a.uid, poly);
      });
      history.set((prev) =>
        prev.map((a) => {
          const poly = byUid.get(a.uid);
          if (!poly) return a;
          return { ...a, kind: 'polygon', points: poly, bbox: bboxFromPoints(poly), source: a.source === 'manual' ? 'sam' : a.source };
        }),
      );
      logger.log('sam_refine', { n: targets.length, converted: byUid.size, ms: Math.round(performance.now() - t0) });
    } catch (e) {
      setError(`SAM: ${(e as Error).message}`);
    } finally {
      setAiBusy(null);
    }
  };

  // ---------- AI: retraining ----------
  useEffect(() => {
    if (!assisted) return;
    api.modelStatus(projectId).then(setModel).catch(() => {});
  }, [projectId, assisted]);

  // Poll the model: fast while it trains or ranks, slowly otherwise (rounds also start automatically
  // on the server every N finished images, so the UI has to notice them by itself).
  const busy = model?.state === 'training' || model?.state === 'scoring';
  const prevModel = useRef<ModelStatus | null>(null);
  useEffect(() => {
    if (!assisted) return;
    const t = window.setInterval(async () => {
      try {
        const st = await api.modelStatus(projectId);
        const prev = prevModel.current;
        prevModel.current = st;
        setModel(st);
        const wasBusy = prev && (prev.state === 'training' || prev.state === 'scoring');
        if (prev && !wasBusy && st.state === 'training') logger.log('train_started', { trigger: 'auto' });
        if (wasBusy && st.state !== 'training' && st.state !== 'scoring') {
          const last = st.rounds.at(-1);
          logger.log('train_finished', { version: st.version, state: st.state, message: st.message, round: last });
          if (st.state === 'idle' && st.version !== prev.version) void loadProposals(imageId, presentRef.current);
        }
      } catch {
        /* keep polling */
      }
    }, busy ? 2000 : 10000);
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, assisted, projectId, imageId]);

  const [order, setOrderState] = useState<'upload' | 'uncertainty'>(
    () => (localStorage.getItem(`labelloop.order.${projectId}`) as 'upload' | 'uncertainty') || 'upload',
  );
  const setOrder = (o: 'upload' | 'uncertainty') => {
    localStorage.setItem(`labelloop.order.${projectId}`, o);
    setOrderState(o);
    logger.log('order_change', { order: o });
  };

  const rank = async () => {
    try {
      const r = await api.rank(projectId);
      logger.log('rank_started', { n_images: r.n_images });
      setModel((m) => (m ? { ...m, state: 'scoring', progress: 0, message: 'ranking' } : m));
    } catch (e) {
      setError(`Ranking: ${(e as Error).message}`);
    }
  };

  const retrain = async () => {
    try {
      await persist(imageId, presentRef.current);
      const r = await api.train(projectId, epochs);
      logger.log('train_started', { n_images: r.n_images, epochs });
      setModel((m) => (m ? { ...m, state: 'training', progress: 0, message: 'starting' } : m));
    } catch (e) {
      setError(`Training: ${(e as Error).message}`);
    }
  };

  // ---------- navigation ----------
  const goTo = async (id: number | undefined) => {
    if (id == null || id === imageId) return;
    navigate(`/projects/${projectId}/images/${id}`); // cleanup of the load effect saves pending edits
  };

  const markDone = async () => {
    if (proposals.length) logger.log('proposals_discarded', { n: proposals.length, version: modelVersion });
    await persist(imageId, presentRef.current, 'done');
    const bySource: Record<string, number> = {};
    presentRef.current.forEach((a) => (bySource[a.source] = (bySource[a.source] ?? 0) + 1));
    logger.log('image_done', { n_annotations: presentRef.current.length, by_source: bySource, dwell_ms: Date.now() - openedAt.current });
    if (assisted && order === 'uncertainty') {
      try {
        const r = await api.next(projectId, imageId, 'uncertainty');
        logger.log('next_image', { order: r.order, score: r.score });
        if (r.image_id != null) return void goTo(r.image_id);
      } catch {
        /* fall back to upload order */
      }
    }
    const after = images.slice(index + 1).find((i) => i.status !== 'done') ?? images.find((i) => i.status !== 'done' && i.id !== imageId);
    if (after) void goTo(after.id);
  };

  // ---------- keyboard ----------
  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandler.current = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement)?.closest?.('input, select, textarea')) return;
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key;
    let handled = true;

    if (mod && k.toLowerCase() === 'z') {
      if (e.shiftKey) {
        history.redo();
        logger.log('redo');
      } else {
        history.undo();
        logger.log('undo');
      }
    } else if (mod && k.toLowerCase() === 'y') {
      history.redo();
      logger.log('redo');
    } else if (mod && k.toLowerCase() === 's') {
      void persist(imageId, presentRef.current);
    } else if (mod && k === 'Enter') {
      void markDone();
    } else if (mod) {
      handled = false;
    } else if (assisted && (k === 's' || k === 'S')) setTool('smart');
    else if (assisted && (k === 'a' || k === 'A')) acceptAll();
    else if (assisted && (k === 'r' || k === 'R')) void refineBoxes();
    else if (k === 'v' || k === 'V') setTool('select');
    else if (k === 'b' || k === 'B') setTool('bbox');
    else if (k === 'p' || k === 'P') setTool('polygon');
    else if (k >= '1' && k <= '9') {
      const c = project?.classes[Number(k) - 1];
      if (c) chooseClass(c.id);
    } else if (k === 'Enter') {
      if (tool === 'smart') acceptSmart();
      else canvasRef.current?.closePolygon();
    } else if (k === 'Escape') {
      if (smartPoints.length || smartBox) {
        clearSmart();
        logger.log('sam_cancel');
      } else if (!canvasRef.current?.cancelDraft()) setSelected(null);
      setShowHelp(false);
    } else if (k === 'Backspace') {
      if (tool === 'smart' && smartPoints.length) {
        const next = smartPoints.slice(0, -1);
        setSmartPoints(next);
        void runSam(next, smartBox);
      } else if (!canvasRef.current?.popPoint() && selected) remove(selected);
    } else if (k === 'Delete') {
      if (selected) remove(selected);
    } else if (k === 'Tab') {
      if (anns.length) {
        const i = anns.findIndex((a) => a.uid === selected);
        const next = (i + (e.shiftKey ? -1 : 1) + anns.length) % anns.length;
        setToolState('select');
        setSelected(anns[i < 0 ? 0 : next].uid);
      }
    } else if (k === 'ArrowRight') void goTo(images[index + 1]?.id);
    else if (k === 'ArrowLeft') void goTo(images[index - 1]?.id);
    else if (k === 'f' || k === 'F') canvasRef.current?.fit();
    else if (k === '+' || k === '=') canvasRef.current?.zoomBy(1.25);
    else if (k === '-') canvasRef.current?.zoomBy(0.8);
    else if (k === 'h' || k === 'H') setHidden((h) => !h);
    else if (k === 'l' || k === 'L') setShowLabels((v) => !v);
    else if (k === '?') setShowHelp((s) => !s);
    else handled = false;

    if (handled) {
      e.preventDefault();
      logger.log('shortcut', { key: mod ? `Mod+${k}` : k });
    }
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);

  // ---------- render ----------
  if (error && !project) return <div className="page"><p className="error">{error}</p></div>;

  const saveLabel = { saved: 'Saved', dirty: 'Unsaved', saving: 'Saving…', error: 'Save failed' }[saveState];
  const doneCount = images.filter((i) => i.status === 'done').length;
  const hints: [string, string][] = (
    {
      select: [['Click', 'select'], ['Drag', 'move'], ['Handles', 'resize'], ['Del', 'delete'], ['Tab', 'next object']],
      bbox: [['Drag', 'draw a box'], ['Esc', 'cancel'], ['1-9', 'class']],
      polygon: [['Click', 'add point'], ['Enter', 'close'], ['Backspace', 'remove point'], ['Esc', 'cancel']],
      smart: [['Click', 'object'], ['Right-click', 'background'], ['Drag', 'box prompt'], ['Enter', 'accept'], ['Esc', 'clear']],
    } as Record<Tool, [string, string][]>
  )[tool];
  const proposalHints: [string, string][] = assisted && proposals.length ? [['Click dashed', 'accept'], ['A', 'accept all'], ['R', 'to polygons']] : [];

  return (
    <div className="annotate">
      <header className="topbar">
        <a href={`#/projects/${projectId}`} className="back-btn" title="Back to project">
          <Icon name="arrowLeft" size={16} />
          <span>{project?.name ?? 'Project'}</span>
        </a>
        <div className="nav-group">
          <button className="icon-btn" disabled={index <= 0} onClick={() => void goTo(images[index - 1]?.id)} title="Previous image (←)">
            <Icon name="chevronLeft" />
          </button>
          <div className="img-pos">
            <span className="filename" title={meta?.filename}>{meta?.filename ?? '…'}</span>
            <div className="pos-row">
              <span>{index + 1} / {images.length}</span>
              <div className="mini-progress"><div style={{ width: `${images.length ? (100 * doneCount) / images.length : 0}%` }} /></div>
              <span>{doneCount} done</span>
            </div>
          </div>
          <button className="icon-btn" disabled={index < 0 || index >= images.length - 1} onClick={() => void goTo(images[index + 1]?.id)} title="Next image (→)">
            <Icon name="chevronRight" />
          </button>
        </div>
        <span className={`status-pill ${meta?.status ?? ''}`}>{meta?.status?.replace('_', ' ')}</span>
        <span className="grow" />
        <span className={`mode-pill ${logger.mode}`} title="Session mode (change on the project page)">
          <Icon name={assisted ? 'sparkle' : 'polygon'} size={14} /> {assisted ? 'AI assisted' : 'Manual'}
        </span>
        <span className={`save-state ${saveState}`}><span className="dot" />{saveLabel}</span>
        <button className="primary" onClick={() => void markDone()} title="Mark done and go to the next image">
          <Icon name="check" size={16} /> Done <span className="kbd on-primary">Ctrl+Enter</span>
        </button>
      </header>

      {error && project && (
        <div className="toast error" onClick={() => setError(null)}>
          <Icon name="x" size={14} /> {error}
        </div>
      )}

      <div className="annotate-body">
        <nav className="toolbar">
          {TOOLS.filter((t) => assisted || t.id !== 'smart').map((t) => (
            <button
              key={t.id}
              className={`tool-btn ${tool === t.id ? 'active' : ''} ${t.id === 'smart' ? 'ai' : ''}`}
              onClick={() => setTool(t.id)}
              title={`${t.label} (${t.key})`}
            >
              <Icon name={TOOL_ICON[t.id]} size={20} />
              <span className="tool-label">{TOOL_SHORT[t.id]}</span>
              <span className="tool-key">{t.key}</span>
            </button>
          ))}
          <div className="tool-sep" />
          <button className="tool-btn small" onClick={() => { history.undo(); logger.log('undo'); }} disabled={!history.canUndo} title="Undo (Ctrl+Z)">
            <Icon name="undo" />
          </button>
          <button className="tool-btn small" onClick={() => { history.redo(); logger.log('redo'); }} disabled={!history.canRedo} title="Redo (Ctrl+Shift+Z)">
            <Icon name="redo" />
          </button>
          <div className="tool-sep" />
          <button className="tool-btn small" onClick={() => canvasRef.current?.fit()} title="Fit to screen (F)"><Icon name="fit" /></button>
          <button className={`tool-btn small ${hidden ? 'active' : ''}`} onClick={() => setHidden((h) => !h)} title="Hide annotations (H)">
            <Icon name={hidden ? 'eyeOff' : 'eye'} />
          </button>
          <button className={`tool-btn small ${showLabels ? '' : 'active'}`} onClick={() => setShowLabels((v) => !v)} title="Show or hide labels (L)">
            <Icon name="tag" />
          </button>
          <span className="grow" />
          <button className="tool-btn small" onClick={() => setShowHelp(true)} title="Keyboard shortcuts (?)"><Icon name="keyboard" /></button>
        </nav>

        <main className="stage-area">
          {meta && loaded ? (
            <AnnotationCanvas
              ref={canvasRef}
              imageUrl={api.imageUrl(meta.id)}
              width={meta.width}
              height={meta.height}
              annotations={anns}
              classes={classMap}
              selected={selected}
              tool={tool}
              hidden={hidden}
              drawColor={activeClass?.color ?? '#ffffff'}
              onSelect={setSelected}
              onCreate={create}
              onChange={change}
              onCheckpoint={history.checkpoint}
              proposals={assisted ? proposals : []}
              onProposal={onProposal}
              smartPoints={smartPoints}
              smartPreview={smartPreview}
              onSmartClick={onSmartClick}
              onSmartBox={onSmartBox}
              showLabels={showLabels}
            />
          ) : (
            <div className="canvas-loading">Loading…</div>
          )}
          {aiBusy && <div className="ai-busy"><span className="spinner" />{aiBusy}</div>}
          <div className="hint-bar">
            {[...hints, ...proposalHints].map(([k, d]) => (
              <span key={k + d}><span className="kbd">{k}</span> {d}</span>
            ))}
          </div>
        </main>

        <aside className="panel">
          <section>
            <h4>Classes</h4>
            <ul className="class-pick">
              {project?.classes.map((c, i) => (
                <li key={c.id} className={c.id === activeClassId ? 'active' : ''} onClick={() => chooseClass(c.id)}>
                  <span className="swatch" style={{ background: c.color }} />
                  <span className="grow">{c.name}</span>
                  {i < 9 && <span className="kbd">{i + 1}</span>}
                </li>
              ))}
            </ul>
            {project && project.classes.length === 0 && (
              <p className="muted small">No classes yet. Add them on the <a href={`#/projects/${projectId}`}>project page</a>.</p>
            )}
          </section>

          {assisted && (
            <>
              <section className="ai-card">
                <div className="ai-card-head">
                  <span className="ai-card-title"><Icon name="zap" size={15} /> Suggestions</span>
                  <span className={`count-badge ${proposals.length ? 'live' : ''}`}>{proposals.length}</span>
                </div>
                <div className="muted small">
                  {modelVersion == null
                    ? 'Detector …'
                    : modelVersion === 0
                      ? 'Zero-shot from your class names (OWLv2)'
                      : `Detector v${modelVersion} · trained on your labels`}
                  {modelVersion != null && modelVersion > 0 && zeroShotClasses.length > 0 && (
                    <> · zero-shot for {zeroShotClasses.join(', ')}</>
                  )}
                </div>
                <div className="row">
                  <button onClick={acceptAll} disabled={!proposals.length} title="Accept all suggestions (A)">
                    <Icon name="check" size={15} /> Accept all <span className="kbd">A</span>
                  </button>
                  <button onClick={() => void refineBoxes()} disabled={!anns.some((a) => a.kind === 'bbox')} title="Turn boxes into polygons with SAM (R)">
                    <Icon name="polygon" size={15} /> To polygons <span className="kbd">R</span>
                  </button>
                </div>
                <label className="slider-label small">
                  <span className="row-between"><span>Confidence threshold</span><b>{threshold.toFixed(2)}</b></span>
                  <input
                    type="range" min={0.1} max={0.9} step={0.05} value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                    onMouseUp={() => {
                      logger.log('threshold_change', { threshold: thresholdRef.current });
                      void loadProposals(imageId, presentRef.current);
                    }}
                  />
                </label>
              </section>

              <section className="ai-card">
                <div className="ai-card-head">
                  <span className="ai-card-title"><Icon name="sparkle" size={15} /> Smart tool</span>
                  <span className="muted small">SAM 2.1</span>
                </div>
                <div className="row-between small">
                  <span title="Plain: SAM sees the whole image. Focus: SAM re-runs on a zoomed window around small objects. Hybrid: first click plain, corrections zoomed.">
                    Mode
                  </span>
                  <div className="segmented mini">
                    {(['plain', 'focus', 'hybrid'] as SamMode[]).map((m) => (
                      <button key={m} className={samMode === m ? 'active' : ''} onClick={() => setSamMode(m)}>{m}</button>
                    ))}
                  </div>
                </div>
                <div className="muted small">Press <span className="kbd">S</span>, click an object, right-click background, Enter to accept.</div>
              </section>

              <section className="ai-card">
                <div className="ai-card-head">
                  <span className="ai-card-title"><Icon name="cpu" size={15} /> Model</span>
                  <span className={`version-badge ${busy ? 'busy' : ''}`}>{busy ? (model?.state === 'scoring' ? 'ranking' : 'training') : `v${model?.version ?? 0}`}</span>
                </div>
                {busy && model && (
                  <div className="progress">
                    <div className="progress-bar" style={{ width: `${Math.round(model.progress * 100)}%` }} />
                    <span>{model.message}</span>
                  </div>
                )}
                {model?.state === 'error' && <p className="error small">{model.message}</p>}
                {model?.state === 'idle' && model.message && <p className="muted small">{model.message}</p>}
                {model && model.rounds.length > 0 && <ModelChart rounds={model.rounds} />}
                <div className="row">
                  <button className="primary" onClick={() => void retrain()} disabled={busy}>
                    <Icon name="loop" size={15} /> {model?.state === 'training' ? 'Training…' : 'Retrain now'}
                  </button>
                  <label className="epochs" title="Training epochs for this round">
                    <input type="number" min={1} max={50} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
                    <span className="muted small">epochs</span>
                  </label>
                </div>
                <p className="muted small">Retrains every 20 finished images. A new model replaces the old one only if it scores better on held-out images.</p>
                <div className="row-between small">
                  <span><Icon name="shuffle" size={14} /> Next image</span>
                  <div className="segmented mini">
                    <button className={order === 'upload' ? 'active' : ''} onClick={() => setOrder('upload')}>in order</button>
                    <button className={order === 'uncertainty' ? 'active' : ''} onClick={() => setOrder('uncertainty')}>most uncertain</button>
                  </div>
                </div>
                {order === 'uncertainty' && (
                  <div className="row-between small">
                    <span className="muted">
                      {model?.ranked_version != null ? `${model.n_ranked} images ranked by v${model.ranked_version}` : 'Not ranked yet'}
                    </span>
                    <button onClick={() => void rank()} disabled={busy}>Rank now</button>
                  </div>
                )}
              </section>
            </>
          )}

          <section className="grow-section">
            <h4>Objects <span className="count">{anns.length}</span></h4>
            {anns.length === 0 && <p className="muted small">No objects yet.</p>}
            <ul className="ann-list">
              {anns.map((a, i) => (
                <li
                  key={a.uid}
                  className={a.uid === selected ? 'active' : ''}
                  onClick={() => {
                    setToolState('select');
                    setSelected(a.uid);
                  }}
                >
                  <span className="swatch" style={{ background: classMap.get(a.class_id)?.color }} />
                  <span className="muted small idx">{i + 1}</span>
                  <select value={a.class_id} onClick={(e) => e.stopPropagation()} onChange={(e) => relabel(a.uid, Number(e.target.value))}>
                    {project?.classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                  <span className="muted" title={a.kind}><Icon name={a.kind === 'bbox' ? 'box' : 'polygon'} size={14} /></span>
                  {a.source !== 'manual' && <span className={`ai-badge ${a.source}`} title={`made with ${a.source === 'sam' ? 'SAM' : 'the detector'}`}>{a.source === 'sam' ? 'SAM' : 'DET'}</span>}
                  <button
                    className="icon"
                    title="Delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      remove(a.uid);
                    }}
                  >
                    <Icon name="x" size={14} />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </div>

      {showHelp && (
        <div className="modal-backdrop" onClick={() => setShowHelp(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="card-head">
              <h3><Icon name="keyboard" /> Keyboard shortcuts</h3>
              <button className="icon" onClick={() => setShowHelp(false)}><Icon name="x" /></button>
            </div>
            <table>
              <tbody>
                {SHORTCUTS.map(([k, d]) => (
                  <tr key={k}><td className="kbd-cell">{k}</td><td>{d}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
