# GestaltView Revenue Collaborator App

This repository contains a full‐stack Shopify app designed to act as a **GestaltView Revenue Collaborator**.  The goal of this app is to provide an embedded assistant inside the Shopify Admin that helps merchants generate revenue through personalized insights, emotional resonance scoring, narrative weaving, bucket‑drop capture, and on‑demand creative synthesis.  It follows Shopify embedded app best practices and integrates core concepts from the GestaltView framework (e.g. Context Spine, Personal Language Key, Bucket Drops, Loom, Manifest Index).

## Contents

```
gestaltview_revenue_collaborator_app/
│
├── backend/                  # Node.js Express backend and optional Python microservice
│   ├── server.js            # Main Express server with Shopify auth stubs and API endpoints
│   ├── auth.js              # Placeholder for Shopify OAuth logic
│   ├── collaborator.js      # Core logic for bucket drops, PLK analysis and generation
│   ├── webhook_handlers.js  # Webhook registration and handlers (checkouts, orders, etc.)
│   └── utils/               # GestaltView utilities and seed prompts
│       ├── gestaltview_seed.py        # Sacred GestaltView seed prompt
│       ├── prompt_templates_enhanced.py # Enhanced prompt manager
│       ├── manifest_index.py         # Simple manifest index implementation
│       └── plk_engine.py             # Simplified Personal Language Key engine
│
├── frontend/                 # React frontend using Shopify Polaris
│   ├── package.json         # npm package manifest for the frontend
│   ├── vite.config.js       # Vite build configuration
│   ├── index.html           # HTML entry point
│   └── src/
│       ├── main.jsx         # Entry point for React
│       ├── App.jsx          # Root component
│       ├── components/
│       │   ├── CollaboratorDashboard.jsx  # Dashboard showing insights and Bucket Drops
│       │   ├── CreationCorner.jsx         # Interface for generating creative artifacts
│       │   └── ResonanceAnalysis.jsx      # Component to analyze and improve text resonance
│       └── styles/
│           └── app.css      # Minimal CSS for the app
│
├── shopify.app.toml         # Shopify CLI manifest with scopes and configuration
├── .env.example             # Sample environment variables configuration
└── README.md                # This file
```

## Getting Started

> **Prerequisites:**
> * Node.js (v18 or later) and npm installed
> * A Shopify Partner account and development store
> * (Optional) Python 3.11+ if you wish to run the PLK analysis microservice

### Installation

1. **Clone the repository** and change into the root directory:

   ```bash
   git clone <REPO_URL>
   cd gestaltview_revenue_collaborator_app
   ```

2. **Install backend dependencies** (Node.js):

   ```bash
   cd backend
   npm install
   ```

3. **Install frontend dependencies**:

   ```bash
   cd ../frontend
   npm install
   ```

4. **Create environment variables file**.  Copy `.env.example` to `.env` and fill in the required values (your Shopify API key/secret, app URL, and any AI API keys you will use).

5. **Run the backend server**.  This will start an Express server on port 3000 by default:

   ```bash
   cd ../backend
   node server.js
   ```

6. **Run the frontend dev server**.  This uses Vite to serve the React app with Shopify Polaris:

   ```bash
   cd ../frontend
   npm run dev
   ```

7. **Install on your development store.**  Use the Shopify CLI to register and run the app.  Refer to the official Shopify documentation and the `shopify.app.toml` file for guidance on registration and OAuth callbacks.

## Features

* **Personal Language Key (PLK) Analysis:**  A lightweight Python module (`plk_engine.py`) analyzes text to identify signature phrases and compute a resonance score relative to the merchant’s known style.  This score helps ensure that generated copy matches the founder’s voice.
* **Bucket Drop Capture:**  The backend exposes a `/api/collaborator/bucket` endpoint that stores quick notes or "Bucket Drops" and returns an acknowledgement.  Bucket drops are the seed material for later creative synthesis.
* **Creative Artifact Generation:**  A `/api/collaborator/generate` endpoint accepts requests for different artifact types (story, pitch deck, poem, mind map, etc.) and returns a placeholder response.  This is where you would integrate with your preferred LLM or image/video generation service.
* **Resonance Analysis and Refinement:**  The `/api/collaborator/analyze` endpoint evaluates a text’s resonance against the PLK profile and can suggest improvements.  See the `ResonanceAnalysis` React component for a sample UI.
* **Manifest Index:**  A simple manifest indexing module (`manifest_index.py`) demonstrates how you might condense a corpus of texts into key terms and sample sentences.  The GestaltView Manifest Index Layer in the original project is more robust; this module is provided for reference and further development.

## Extending This App

This project is a foundation.  The stubs in `auth.js`, `collaborator.js` and `webhook_handlers.js` provide a starting point for implementing full functionality:

* **Shopify Authentication:**  Integrate the Shopify OAuth flow using the `@shopify/shopify-api` package in `auth.js` to securely obtain API tokens and verify session cookies.
* **Webhook Handling:**  In `webhook_handlers.js` you can register for events like `checkouts/create` and `orders/paid`, verify HMAC signatures, and trigger business logic (e.g. abandoned cart recovery).
* **AI Integration:**  Replace the placeholders in `collaborator.js` with calls to your chosen AI services (e.g. OpenAI, Anthropic, open‑source LLMs) and use the GestaltView seed prompt and enhanced templates to ensure outputs remain consciousness‑serving.
* **Database Persistence:**  For production, connect to a database (e.g. PostgreSQL) to store bucket drops, PLK profiles, and manifest entries.  The current implementation stores data in memory for simplicity.

## License

This repository is provided as an educational and illustrative example.  You are free to modify and extend it for your own projects.  See the [LICENSE](LICENSE) file for details if included.
