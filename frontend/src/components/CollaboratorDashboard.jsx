import React, { useState } from 'react';
import {
  Card,
  BlockStack,
  TextField,
  Button,
  ResourceList,
  ResourceItem,
  Text,
} from '@shopify/polaris';

// Dashboard component shows captured bucket drops and allows capturing new ones
export default function CollaboratorDashboard() {
  const [dropText, setDropText] = useState('');
  const [drops, setDrops] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleCapture = async () => {
    if (!dropText.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/collaborator/bucket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: dropText }),
      });
      const data = await res.json();
      setDrops([data.drop, ...drops]);
      setDropText('');
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sectioned>
      <BlockStack gap="400">
        <TextField
          label="Quick Bucket Drop"
          placeholder="Capture an idea or thought..."
          value={dropText}
          onChange={setDropText}
          multiline
          autoComplete="off"
        />
        <Button onClick={handleCapture} loading={loading} primary>
          Capture
        </Button>
        <Text variant="headingMd">Recent Bucket Drops</Text>
        <ResourceList
          resourceName={{ singular: 'drop', plural: 'drops' }}
          items={drops}
          renderItem={(drop) => (
            <ResourceItem id={drop.id}>
              <Text variant="bodyMd">{drop.text}</Text>
              <Text variant="subdued" fontWeight="regular">
                {new Date(drop.timestamp).toLocaleString()}
              </Text>
            </ResourceItem>
          )}
        />
      </BlockStack>
    </Card>
  );
}
