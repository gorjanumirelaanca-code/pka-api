# pKa Predictor API — deployment guide

This wraps `pka_predictor.py` in a FastAPI service so a frontend (your Vercel
dashboard) can call it over HTTP. It runs the real Python + RDKit + scikit-learn
stack — the part that cannot run on Vercel.

## Files needed in the deploy folder
```
app.py             # FastAPI service (this)
pka_predictor.py   # the pipeline (from earlier)
requirements.txt   # dependencies
Dockerfile         # only needed for Hugging Face Spaces / container hosts
```

## Run locally first (sanity check)
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# open http://localhost:8000/docs  for an interactive API console
```

Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles":["CC(=O)Oc1ccccc1C(=O)O"],"ids":["aspirin"]}'
```

---

## Option A — Render (simplest for Python)

1. Push this folder to a GitHub repo.
2. On render.com: **New → Web Service**, connect the repo.
3. Settings:
   - **Environment:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. (Optional) add environment variables:
   - `ALLOWED_ORIGINS` = your Vercel URL, e.g. `https://your-app.vercel.app`
   - `ACIDIC_MODEL_PATH` / `BASIC_MODEL_PATH` if you commit trained `.joblib` models.
5. Deploy. Your API is at `https://your-service.onrender.com`.

Note: Render's free tier sleeps when idle, so the first request after a pause
takes ~30–60 s to wake.

---

## Option B — Hugging Face Spaces (Docker)

1. Create a new Space → **SDK: Docker** → blank template.
2. Upload all four files (`app.py`, `pka_predictor.py`, `requirements.txt`,
   `Dockerfile`) via the web UI or `git push` to the Space repo.
3. The Space builds automatically and serves on port 7860.
4. Your API base URL is `https://<user>-<space-name>.hf.space`.
5. (Optional) set `ALLOWED_ORIGINS` under the Space's **Settings → Variables**.

RDKit installs cleanly here because it's a full container, not a size-capped
serverless function.

---

## Connect your Vercel dashboard to the API

In the dashboard's JavaScript, replace the in-browser estimate with a fetch to
your API. Minimal example:

```javascript
const API = "https://your-service.onrender.com"; // or the HF Space URL

async function predictViaApi(smilesLines) {
  const res = await fetch(API + "/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: smilesLines })  // "id,SMILES" lines
  });
  if (!res.ok) throw new Error("API error " + res.status);
  const data = await res.json();
  return data.results; // [{mol_id, functional_group, pka_type, predicted_pka, confidence, ...}]
}
```

Set `ALLOWED_ORIGINS` on the API to your Vercel domain so the browser is
permitted to call it (CORS).

## Serving a trained model
By default the API uses the rule-based fallback (`model_mode:
"rule_based_fallback"`). To serve a real model, train regressors, save them with
`joblib.dump(model, "acidic.joblib")`, commit them, and point
`ACIDIC_MODEL_PATH` / `BASIC_MODEL_PATH` at the files. The response's
`model_mode` will switch to `"trained"`.
