import { useEffect, useState } from 'react';

/** Tiny hash router: #/, #/projects/3, #/projects/3/images/17 */
export type Route =
  | { page: 'home' }
  | { page: 'project'; projectId: number }
  | { page: 'annotate'; projectId: number; imageId: number };

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  if (parts[0] === 'projects' && parts[1]) {
    const projectId = Number(parts[1]);
    if (parts[2] === 'images' && parts[3]) return { page: 'annotate', projectId, imageId: Number(parts[3]) };
    return { page: 'project', projectId };
  }
  return { page: 'home' };
}

export function navigate(path: string) {
  window.location.hash = path;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}
