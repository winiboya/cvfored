import React from 'react';
import { createRoot } from 'react-dom/client';
import FocusChart from './components/FocusChart';
import TopicChart from './components/TopicChart';

const focusContainer = document.getElementById('focus-chart');
if (focusContainer) {
    const root = createRoot(focusContainer);
    root.render(<FocusChart />);
    
} else {
    console.error('Could not find focus-chart element');
}

const topicContainer = document.getElementById('topic-chart');
if (topicContainer) {
    const topicRoot = createRoot(topicContainer);
    topicRoot.render(<TopicChart />);
} else {
    console.error('Could not find topic-chart element');
}

// Debug logs to check if data is available
console.log('Window focus data:', window.focusData);
console.log('Window topic data:', window.topicData);