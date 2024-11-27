import React from 'react';
import { createRoot } from 'react-dom/client';
import FocusChart from './components/FocusChart';

const container = document.getElementById('focus-chart');
if (container) {
    const root = createRoot(container);
    root.render(<FocusChart />);
} else {
    console.error('Could not find focus-chart element');
}

// Debug log to check if data is available
console.log('Window focus data:', window.focusData);