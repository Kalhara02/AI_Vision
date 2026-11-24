import { useRoute } from './lib/router';
import ProjectList from './components/ProjectList';
import ProjectPage from './components/ProjectPage';
import AnnotatePage from './components/AnnotatePage';

export default function App() {
  const route = useRoute();
  switch (route.page) {
    case 'project':
      return <ProjectPage key={route.projectId} projectId={route.projectId} />;
    case 'annotate':
      return <AnnotatePage key={route.projectId} projectId={route.projectId} imageId={route.imageId} />;
    default:
      return <ProjectList />;
  }
}
