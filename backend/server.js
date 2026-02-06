/*
 * GestaltView Revenue Collaborator – Express Server
 *
 * This server bootstraps a Shopify embedded app.  It exposes a handful
 * of API routes that demonstrate how you might capture bucket drops,
 * perform PLK analysis, generate creative artifacts, and register
 * webhooks.  It deliberately omits full OAuth and webhook verification
 * logic to keep the example concise.  You can extend this file by
 * integrating the @shopify/shopify-api package and your preferred AI
 * services.
 */

const express = require('express');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');

// Load environment variables from .env
dotenv.config();

const { verifyAuth } = require('./auth');
const {
  captureBucketDrop,
  generateArtifact,
  analyzeText,
  improveText
} = require('./collaborator');

const app = express();

// Parse JSON and urlencoded bodies
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Home route for health check
app.get('/', (req, res) => {
  res.json({ message: 'GestaltView Revenue Collaborator API' });
});

// Middleware: verify Shopify session (placeholder)
// In production, you should verify the HMAC signature and session token
app.use('/api', verifyAuth);

// Capture a bucket drop
app.post('/api/collaborator/bucket', async (req, res) => {
  try {
    const { text, tags } = req.body;
    const response = await captureBucketDrop(text, tags || []);
    res.json(response);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to capture bucket drop' });
  }
});

// Generate a creative artifact
app.post('/api/collaborator/generate', async (req, res) => {
  try {
    const { type, topic } = req.body;
    const result = await generateArtifact(type, topic);
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to generate artifact' });
  }
});

// Analyze a piece of text against the PLK profile
app.post('/api/collaborator/analyze', async (req, res) => {
  try {
    const { content } = req.body;
    const analysis = await analyzeText(content);
    res.json(analysis);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to analyze text' });
  }
});

// Improve a piece of text to better match the PLK profile
app.post('/api/collaborator/improve', async (req, res) => {
  try {
    const { content } = req.body;
    const improved = await improveText(content);
    res.json(improved);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to improve text' });
  }
});

// Register webhook endpoints (placeholders)
const { registerCheckoutWebhook } = require('./webhook_handlers');

app.post('/webhooks/checkouts_create', async (req, res) => {
  try {
    // Verify HMAC signature of webhook here
    await registerCheckoutWebhook(req.body);
    res.status(200).send('Webhook received');
  } catch (err) {
    console.error(err);
    res.status(500).send('Webhook handling failed');
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 GestaltView Revenue Collaborator running on port ${PORT}`);
});
