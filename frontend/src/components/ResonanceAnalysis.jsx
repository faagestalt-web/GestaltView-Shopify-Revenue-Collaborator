import React, { useState } from 'react';
import {
  Card,
  LegacyStack,
  TextField,
  Button,
  Text,
  Spinner,
} from '@shopify/polaris';

export default function ResonanceAnalysis() {
  const [text, setText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [improved, setImproved] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/collaborator/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text }),
      });
      const data = await res.json();
      setAnalysis(data);
      setImproved(null);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleImprove = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/collaborator/improve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text }),
      });
      const data = await res.json();
      setImproved(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sectioned>
      <LegacyStack vertical spacing="loose">
        <TextField
          label="Analyze Text"
          placeholder="Enter text to analyze resonance..."
          value={text}
          onChange={setText}
          multiline
          autoComplete="off"
        />
        <LegacyStack distribution="fillEvenly">
          <Button onClick={handleAnalyze} disabled={loading || !text.trim()}>
            Analyze
          </Button>
          <Button onClick={handleImprove} disabled={loading || !text.trim()}>
            Improve
          </Button>
        </LegacyStack>
        {loading && <Spinner accessibilityLabel="Loading" size="small" />}
        {analysis && (
          <div>
            <Text variant="headingMd">Analysis</Text>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(analysis, null, 2)}</pre>
          </div>
        )}
        {improved && (
          <div>
            <Text variant="headingMd">Improved Text</Text>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{improved.improved}</pre>
          </div>
        )}
      </LegacyStack>
    </Card>
  );
}
