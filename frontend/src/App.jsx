import React, { useState } from 'react';
import { Page, Tabs } from '@shopify/polaris';
import CollaboratorDashboard from './components/CollaboratorDashboard';
import CreationCorner from './components/CreationCorner';
import ResonanceAnalysis from './components/ResonanceAnalysis';

export default function App() {
  const [selectedTab, setSelectedTab] = useState(0);
  const tabs = [
    { id: 'dashboard', content: 'Dashboard' },
    { id: 'creation', content: 'Creation Corner' },
    { id: 'analysis', content: 'Resonance Analysis' },
  ];

  return (
    <Page
      title="GestaltView Revenue Collaborator"
      subtitle="Weave your story into your store and boost revenue"
    >
      <Tabs
        tabs={tabs}
        selected={selectedTab}
        onSelect={(selected) => setSelectedTab(selected)}
      >
        <div>
          {selectedTab === 0 && <CollaboratorDashboard />}
          {selectedTab === 1 && <CreationCorner />}
          {selectedTab === 2 && <ResonanceAnalysis />}
        </div>
      </Tabs>
    </Page>
  );
}
