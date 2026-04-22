# Overview
Embed MyoSuite demo UI into Parkinsons Motor OpenEnv environment via Gradio HTML component.

# Asset placement
Create `parkinsons_Motor/static/myosuite_demo/`.
Copy all files from `models/myosuite_demo/` (index.html, style.css, examples/, node_modules/, images) into that folder.

# FastAPI static mount
In `parkinsons_Motor/server/app.py` add:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
```

# Gradio UI changes
In `build_gradio_app` after creating demo Blocks, add:
```python
from pathlib import Path
html_path = Path(__file__).parents[2] / "static" / "myosuite_demo" / "index.html"
with open(html_path) as f:
    html_content = f.read()
gr.HTML(html_content)
```
Adjust relative URLs in `index.html` to `/static/myosuite_demo/...`.

# Path adjustments
Replace `<link rel="stylesheet" href="style.css">` with `/static/myosuite_demo/style.css`.
Replace import map paths `"three": "./node_modules/three/build/three.module.js"` with `/static/myosuite_demo/node_modules/three/build/three.module.js`.
Update script src `./examples/main.js` to `/static/myosuite_demo/examples/main.js`.

# Testing
Run `uv run --project . server`.
Visit `http://localhost:8000/web`.
Verify UI loads, interactive controls work.
Check console for missing asset errors.

# Acceptance criteria
UI identical to original MyoSuite demo.
No extra Node/Gradle steps required.
FastAPI serves static assets correctly.
Gradio page displays embedded UI without iframe.
