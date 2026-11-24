import { useCallback, useState } from 'react';

const LIMIT = 200;

interface State<T> {
  past: T[];
  present: T;
  future: T[];
}

/**
 * Undo/redo over immutable snapshots.
 *
 * - set(next)          records a new undo step
 * - set(next, false)   replaces the present without a step (live dragging)
 * - checkpoint()       records the present as a step (call at drag start)
 */
export function useHistory<T>(initial: T) {
  const [s, setS] = useState<State<T>>({ past: [], present: initial, future: [] });

  const set = useCallback((updater: T | ((prev: T) => T), record = true) => {
    setS((st) => {
      const next = typeof updater === 'function' ? (updater as (p: T) => T)(st.present) : updater;
      if (next === st.present) return st;
      return record
        ? { past: [...st.past, st.present].slice(-LIMIT), present: next, future: [] }
        : { ...st, present: next };
    });
  }, []);

  const checkpoint = useCallback(() => {
    setS((st) => ({ past: [...st.past, st.present].slice(-LIMIT), present: st.present, future: [] }));
  }, []);

  const undo = useCallback(() => {
    setS((st) =>
      st.past.length
        ? { past: st.past.slice(0, -1), present: st.past[st.past.length - 1], future: [st.present, ...st.future] }
        : st,
    );
  }, []);

  const redo = useCallback(() => {
    setS((st) =>
      st.future.length ? { past: [...st.past, st.present], present: st.future[0], future: st.future.slice(1) } : st,
    );
  }, []);

  const reset = useCallback((value: T) => setS({ past: [], present: value, future: [] }), []);

  return {
    present: s.present,
    set,
    checkpoint,
    undo,
    redo,
    reset,
    canUndo: s.past.length > 0,
    canRedo: s.future.length > 0,
  };
}
