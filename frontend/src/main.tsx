import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import Overview from './pages/Overview';
import LiveInference from './pages/LiveInference';
import Reliability from './pages/Reliability';
import DriftMonitoring from './pages/DriftMonitoring';
import RoutingIntelligence from './pages/RoutingIntelligence';
import FailureDiagnosis from './pages/FailureDiagnosis';
import HumanReview from './pages/HumanReview';
import ExperimentCentre from './pages/ExperimentCentre';
import './index.css';

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Overview /> },
      { path: 'inference', element: <LiveInference /> },
      { path: 'reliability', element: <Reliability /> },
      { path: 'drift', element: <DriftMonitoring /> },
      { path: 'routing', element: <RoutingIntelligence /> },
      { path: 'diagnosis', element: <FailureDiagnosis /> },
      { path: 'reviews', element: <HumanReview /> },
      { path: 'experiments', element: <ExperimentCentre /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
