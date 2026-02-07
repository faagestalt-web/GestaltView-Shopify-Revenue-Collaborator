# Codex Partner Context (Carry-Over)

This file captures the most important context from the current Codex session so it can be copied into a future session.

## Repository Overview
- **Name:** GestaltView Shopify Revenue Collaborator
- **Structure:** `backend/` (Node/Express), `frontend/` (React + Polaris via Vite), `docs/` (usage notes), `shopify.app.toml` (Shopify CLI config).
- **Primary Goal:** Shopify embedded app that provides a GestaltView collaborator (bucket drops, resonance analysis, creative generation).

## Key Features Already Implemented
- **Backend API routes** for bucket drops, generation, analysis, and improvement.
- **Placeholder AI logic** with clear extension points (`backend/collaborator.js`).
- **Prompt + PLK utilities** in `backend/utils/` for consciousness-serving prompts and a lightweight Personal Language Key analyzer.
- **Frontend** with three tabs: Dashboard, Creation Corner, Resonance Analysis.

## Files Worth Reviewing First
- `backend/server.js` (Express API)
- `backend/collaborator.js` (core collaborator logic + placeholders)
- `backend/utils/prompt_templates_enhanced.py` (prompt manager / GestaltView seed integration)
- `backend/utils/gestaltview_seed.py` (GestaltView seed prompt + contexts)
- `backend/utils/plk_engine.py` (Personal Language Key example)
- `backend/utils/manifest_index.py` (lightweight knowledge synthesis)
- `frontend/src/App.jsx` + `frontend/src/components/*` (UI)
- `docs/usage.md` (Python + shell API examples)

## Recent Changes & Decisions
- **Polaris layout:** switched legacy layout handling to `LegacyStack` in the frontend components to preserve layout semantics with Polaris v12+.
- **Shopify CLI config:** `shopify.app.toml` now uses `http://localhost:3000` and includes a `client_id`.
- **Docs:** added `docs/usage.md` with Python + shell examples for API calls.
- **README:** added `shopify app dev --use-localhost` guidance.

## Running Locally (Quick)
1. Backend:
   ```bash
   cd backend
   npm install
   npm run dev
   ```
2. Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Shopify CLI (optional):
   ```bash
   shopify app dev --use-localhost
   ```

## Known Notes / Issues
- Vite build shows a **Polaris CSS warning** about `@media (--p-breakpoints-md-up) and print`. Build still succeeds.
- There were earlier build failures due to `Stack` being removed in Polaris; `LegacyStack` is currently used to preserve layout.

## User Preferences & Requests
- Prefers **copy-paste-friendly blocks** of code or docs (low-friction).
- Wants to build **full system before starting Shopify trial**.
- Interested in **custom commerce framework** (non-Shopify) or a **Neural Handshake** UX layer.
- Wants help reviewing newly uploaded “Neural Handshake / Superintelligence Sidekick / knowledge synthesis” code when available in the repo.

## Next Suggested Actions (For Future Session)
- Scan for newly uploaded files (Neural Handshake, Sidekick Engine, knowledge synthesis system).
- Provide a short list of “most useful modules” and integration points.
- If desired, create a minimal custom commerce backend (Stripe) or a neural handshake service scaffold.
