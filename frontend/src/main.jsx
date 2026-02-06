import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import '@shopify/polaris/build/esm/styles.css';
import './styles/app.css';

import { AppProvider } from '@shopify/polaris';
import translations from '@shopify/polaris/locales/en.json';

// Root render function
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppProvider i18n={translations}>
      <App />
    </AppProvider>
  </React.StrictMode>
);
