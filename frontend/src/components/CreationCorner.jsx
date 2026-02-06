import React, { useState } from 'react';
import {
  Card,
  Stack,
  Select,
  TextField,
  Button,
  Text,
  Spinner,
} from '@shopify/polaris';

// Available artifact types
const artifactOptions = [
  { label: 'Story', value: 'story' },
  { label: 'Pitch Deck', value: 'pitchDeck' },
  { label: 'Poem', value: 'poem' },
  { label: 'Mind Map', value: 'mindMap' },
  { label: 'Image', value: 'image' },
  { label: 'Video', value: 'video' },
];

export default function CreationCorner() {
  const [type, setType] = useState('story');
  const [topic, setTopic] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/collaborator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, topic }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sectioned>
      <Stack vertical spacing="loose">
        <Select label="Artifact Type" options={artifactOptions} onChange={setType} value={type} />
        <TextField
          label="Topic"
          placeholder="e.g. Origin story of our brand"
          value={topic}
          onChange={setTopic}
          autoComplete="off"
        />
        <Button onClick={handleGenerate} primary disabled={loading || !topic.trim()}>
          Generate
        </Button>
        {loading && <Spinner accessibilityLabel="Generating" size="small" />}
        {result && (
          <div>
            <Text variant="headingMd">Result</Text>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{result.content || JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </Stack>
    </Card>
  );
}
