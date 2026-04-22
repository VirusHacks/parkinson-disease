# embed-myosuite-ui Implementation Plan

> **For agentic workers:** REQUIRED SUB‑SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task‑by‑task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed the MyoSuite demo UI inside the Parkinsons Motor OpenEnv environment via a Gradio `HTML` component, serving all static assets through FastAPI.

**Architecture:** Static MyoSuite assets are copied into a `static/myosuite_demo/` folder inside the RL environment. FastAPI mounts this folder at `/static`. The Gradio UI loads the `index.html` file content and displays it within the Gradio app, with all relative URLs rewritten to point to the `/static` route.

**Tech Stack:** Python 3.14, FastAPI, Gradio, OpenEnv, three.js (bundled in the demo).

---

### Task 1: Copy MyoSuite demo assets

**Files:**
- Create: `parkinsons_Motor/static/myosuite_demo/` (directory).
- Copy: all files and subfolders from `models/myosuite_demo/` into that directory (including `index.html`, `style.css`, `examples/`, `node_modules/`, images).

- [ ] **Step 1: Copy assets**
```bash
cp -R /Users/vinay/vscode/hackathon/meta-hackathon/models/myosuite_demo/* \
    /Users/vinay/vscode/hackathon/meta-hackathon/environment/parkinsons_Motor/static/myosuite_demo/
```
- [ ] **Step 2: Verify copy**
```bash
ls -R /Users/vinay/vscode/hackathon/meta-hackathon/environment/parkinsons_Motor/static/myosuite_demo
```
- [ ] **Step 3: Commit copy**
```bash
git add parkinsons_Motor/static/myosuite_demo
git commit -m "feat: add MyoSuite demo static assets"
```

---

### Task 2: Mount static folder in FastAPI

**Files:**
- Modify: `parkinsons_Motor/server/app.py` (add import and mount line).

- [ ] **Step 1: Insert import**
Add near top of file:
```python
from fastapi.staticfiles import StaticFiles
```
- [ ] **Step 2: Mount static directory**
Insert after FastAPI app creation:
```python
app.mount("/static", StaticFiles(directory="static"), name="static")
```
- [ ] **Step 3: Run lint/check**
```bash
python -m py_compile parkinsons_Motor/server/app.py
```
- [ ] **Step 4: Commit FastAPI mount**
```bash
git add parkinsons_Motor/server/app.py
git commit -m "chore: mount /static for MyoSuite assets"
```

---

### Task 3: Adjust `index.html` URLs to use `/static` route

**Files:**
- Modify: `parkinsons_Motor/static/myosuite_demo/index.html`.

- [ ] **Step 1: Replace stylesheet link**
Change
```html
<link type="text/css" rel="stylesheet" href="style.css">
```
to
```html
<link type="text/css" rel="stylesheet" href="/static/myosuite_demo/style.css">
```
- [ ] **Step 2: Update import‑map paths**
In the `<script type="importmap">` block, replace each relative path (`"./node_modules/..."`) with an absolute `/static/myosuite_demo/node_modules/...` path. Example:
```json
"three": "/static/myosuite_demo/node_modules/three/build/three.module.js",
"three/addons/": "/static/myosuite_demo/node_modules/three/examples/jsm/"
```
- [ ] **Step 3: Update main script source**
Change
```html
<script type="module" src="./examples/main.js"></script>
```
to
```html
<script type="module" src="/static/myosuite_demo/examples/main.js"></script>
```
- [ ] **Step 4: Verify HTML loads in browser** (manual test).
- [ ] **Step 5: Commit HTML changes**
```bash
git add parkinsons_Motor/static/myosuite_demo/index.html
git commit -m "fix: rewrite asset URLs for FastAPI static serving"
```

---

### Task 4: Add Gradio `HTML` component to load the demo

**Files:**
- Modify: the project's `gradio_ui.py` (located in the virtual‑env site‑packages or a local copy under the repo). Ensure the path matches the import used by `build_gradio_app`.

- [ ] **Step 1: Import `Path`**
Add near other imports:
```python
from pathlib import Path
```
- [ ] **Step 2: Read `index.html` content**
Inside `build_gradio_app`, after creating `demo = gr.Blocks(...)`, add:
```python
html_path = Path(__file__).parents[2] / "static" / "myosuite_demo" / "index.html"
with open(html_path, "r", encoding="utf-8") as f:
    myosuite_html = f.read()
```
- [ ] **Step 3: Insert `gr.HTML` component**
Place the component in the right column, e.g.:
```python
gr.HTML(value=myosuite_html)
```
- [ ] **Step 4: Run quick test**
```bash
uv run --project . server
```
Open `http://localhost:8000/web` and confirm the MyoSuite UI appears inside the Gradio page.
- [ ] **Step 5: Commit Gradio changes**
```bash
git add path/to/gradio_ui.py
git commit -m "feat: embed MyoSuite UI via Gradio HTML component"
```

---

### Task 5: End‑to‑end testing

**Files:** None (manual verification).

- [ ] **Step 1: Start server**
```bash
uv run --project . server
```
- [ ] **Step 2: Open browser to `http://localhost:8000/web`.
- [ ] **Step 3: Verify UI** – no 404s, interactive controls work, console shows no missing asset errors.
- [ ] **Step 4: Record test result** – write a short note in `README.md`.
- [ ] **Step 5: Commit test verification**
```bash
git add README.md
git commit -m "test: verify embedded MyoSuite UI works"
```

---

### Task 6: Cleanup

- [ ] **Step 1: Remove any duplicate files** (if any were unintentionally copied).
- [ ] **Step 2: Update `.gitignore`** to exclude large `node_modules` inside `static/myosuite_demo` if not needed for source control (optional).
- [ ] **Step 3: Final commit**
```bash
git add .
git commit -m "chore: cleanup after embedding MyoSuite UI"
```

---

**Plan complete.**