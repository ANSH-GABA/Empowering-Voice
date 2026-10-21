# Empowering Voice: Build Guide

This guide walks through your README phase by phase. Phases 0–7 (the MVP) are already
scaffolded in this folder, so for those you **run, record, train and test**. For Phases 8–13
you get exact steps and the key code.

## Three decisions that save you days

1. **Two hands, 129 features.** Much of ISL fingerspelling uses both hands, so the README's
   "63 features" is not enough. The code tracks 2 hands: `63 + 63 + 3` (one hand's shape,
   the other hand's shape, and the offset between the wrists). A missing hand is zeros.

2. **Train with scikit-learn, run in plain TypeScript.** `train_static.py` trains the
   README's Dense128 → Dense64 → softmax network and writes `static_mlp.json`. The browser
   runs it with a 30-line forward pass. No TensorFlow install or converter problems for the MVP.
   Add TensorFlow.js in Phase 8, where you actually need LSTM/GRU.

3. **Split by participant.** Frames from one person are near-duplicates. A random split gives
   fake 99% scores. `train_static.py` splits by participant automatically.

## Repo map

```
Model/ml/...               hand tracking, features, classifier, stabilizer   (Phases 2, 5, 6)
frontend/src/pages/CollectPage.tsx    dataset collector                       (Phase 3)
frontend/src/pages/TranslatePage.tsx  ISL → text → speech                     (Phases 5–7)
Model/scripts/check_dataset.py        run after every recording session
Model/scripts/train_static.py         compare KNN/SVM/RF/MLP, export for browser (Phases 4–5)
backend/app/services/hf_classifier.py  pretrained Hugging Face model behind FastAPI (Path B)
frontend/src/services/hfClassifier.ts  hand crop → /api/predict
backend/app/scripts/evaluate_hf.py    score the HF model on your own photos
docs/experiments/                     tables + confusion matrix for your report
```
---

## Phase 0: Setup (30 minutes)

Install Node 20+, Python 3.10+ and Git. Then:

```bash
git init
cd frontend && npm install && cd ..
python -m venv .venv
Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Done when:** `cd frontend && npm run dev` opens the app at http://localhost:5173 with
Translate and Collect data tabs. Commit: `git add . && git commit -m "Phase 0: setup"`.

## Phase 1: Camera (already built)

`components/camera/CameraView.tsx` opens the webcam and shows a mirrored preview.

**Done when:** you click Allow, see yourself, and a small FPS counter appears.
If it fails: the page must run on `localhost` or HTTPS, and no other app can be using the camera.

## Phase 2: Hand detection (already built)

The same component loads MediaPipe's Hand Landmarker (`numHands: 2`) and draws 21 points per
hand. The first load downloads about 8 MB, so wait for "Loading hand tracking model…".

**Done when:** a skeleton follows each hand at roughly 20–30 FPS. Check both hands at once,
and try a dim room and a plain wall. If it is slow, close other tabs and use Chrome.

## Phase 3: Dataset collector (about 1 week of recording)

Open **Collect data**. Workflow per person:

1. Enter a **Participant ID** (`P01`, `P02`, …). Never reuse IDs for different people.
2. Pick a sign and press **Space** (or the button). After a 2-second countdown it records about
   10 samples per second until the take is full (default 200), then jumps to the next sign.
3. Do **3–5 takes per sign** and vary each: distance from camera, slight hand angle, lighting,
   position in frame. This variety is what makes the model work on other people.
4. Click **Download CSV** every few letters. The tab keeps data in memory only, and it warns
   you if you try to close it with unsaved samples.
5. Move the CSVs into `Model/ml/data/processed/static/`, then check them:

```bash
python Model/scripts/check_dataset.py
```

**Before you record:**
- Verify every sign against a reliable ISL reference (for example ISLRTC materials), and ideally
  have someone fluent in ISL review your reference poses. Do not rely on ASL charts: the alphabets differ.
- Get verbal or written consent from every participant and keep names out of the files.
- Aim for **6+ participants** (README example: 4 train / 1 validation / 1 test). Roughly 100–170 samples
  per sign per person gives the README's 500–1,000 per class.

**Done when:** `check_dataset.py` shows every sign ≥ 300 samples and ≥ 3 participants with no warnings.

## Phase 4: Static ML (1–2 weeks → realistically 2 days once data exists)

```bash
python Model/scripts/train_static.py
```

It prints accuracy, precision, recall, F1 and latency for KNN, SVM, Random Forest and MLP,
and saves to `docs/experiments/`:
`static_baselines.csv` (a table for your report) and `static_confusion_matrix.png`.

Force specific participants if you like: `--val P05 --test P06`.

**If scores are low:**

| Symptom | Fix |
|---|---|
| Test F1 far below validation F1 | Too few participants; record 2 more people |
| A couple of letters confused (see "Most confused pairs") | Look at the two ISL signs: are they truly distinct from the camera view? Record more varied takes of those two |
| Everything poor | Run `check_dataset.py`; look for all-zero rows, mislabeled takes, or a participant signing differently |
| One-handed signs confused with two-handed ones | Make sure the second hand was fully out of frame while recording one-handed signs |

**Done when:** you have a table and confusion matrix you can paste into the report, and you know
which pairs are confused and why.

## Phase 5: Browser inference (already built)

`train_static.py` writes `frontend/public/models/static_mlp.json`. Reload the Translate page:
it shows "Model loaded", and the current sign plus a confidence bar update live.

**Done when:** you sign A–Z and the right letters appear. **This is your MVP: ISL → Text.**

## Phase 6: Sentence builder (already built)

`Model/ml/prediction/smoother.ts` types a letter only after it wins **10 of 12 frames at ≥ 75% confidence**,
then waits for you to change sign or lower your hand (8 frames) before repeating a letter.
Buttons: Space, Delete, Clear. Labels such as `SPACE`, `DELETE`, `CLEAR` or whole words in your
model act on the sentence directly (`applyToken`).

Tuning (`DEFAULT_STABILIZER`):

| Problem | Change |
|---|---|
| Letters appear too slowly | `windowSize` 12 → 8, `minVotes` 10 → 6 |
| Wrong letters flicker in | raise `minConfidence` to 0.85 |
| Can't type double letters | lower `releaseFrames` to 5 |

## Phase 7: Speech (already built)

The **Speak** button uses the browser's Web Speech API (`services/speechService.ts`). It defaults
to `en-IN`. Use `speak(text, "hi-IN")` for Hindi, if your OS has a Hindi voice installed. Chrome and Edge work best.

**Core milestone reached: ISL → Text → Speech.** Tag it: `git tag mvp`.

---

## Path B: Use a pretrained Hugging Face model (skips Phases 3–4 for letters and digits)

If you don't want to record data and train, run an existing model instead. The default is
[`Hemgg/sign-language-classification`](https://huggingface.co/Hemgg/sign-language-classification):
a Vision Transformer (ViT-base) fine-tuned for **35 classes: digits 1–9 and letters A–Z**.

**How it works**

```
Webcam → MediaPipe finds the hands → crop a square around them → JPEG
      → POST /api/predict (FastAPI, on your own computer)
      → Hugging Face model → label + confidence
      → same stabilizer → sentence → speech
```

MediaPipe is still used, but only to find and crop the hands. This model looks at **photos**, not landmarks,
so the 129-feature pipeline is not used in this mode. The model runs in Python because it is a 343 MB PyTorch model;
that is too heavy to send to a browser as-is.

**Run it**

```bash
# Terminal 1: backend (first start downloads about 343 MB, then it is cached)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Terminal 2: frontend
cd frontend && npm run dev
```

Open Translate and choose **Hugging Face model**. A green "Backend connected" line shows the model name.
Camera pixels only go to `localhost`, so nothing leaves your computer.

**Try a different Hugging Face model:** `SIGNBRIDGE_HF_MODEL=owner/model uvicorn app.main:app --port 8000`.
It must be an *image-classification* model. If its labels aren't single letters or words, tell the app how to
type them in `applyToken` (`smoother.ts`).

**What the card's "99% accuracy" does and does not mean.** The author reports it on their own evaluation split of a
dataset the card doesn't name. Photos from one dataset share backgrounds, lighting and often the same few people,
so scores like this usually drop on new people and cameras. Treat it as "works on its own data" until you test it yourself:

1. Take 10–20 photos per sign from 3+ people who are not in the model's data (phone photos are fine).
2. Put them in `eval_images/A/…jpg`, `eval_images/B/…jpg`, etc.
3. Run `python backend/scripts/evaluate_hf.py eval_images`. It prints overall accuracy, the weakest signs and the
   most confused pairs, and writes `docs/experiments/hf_eval.csv` for your report.

**Tuning:**

| Problem | Fix |
|---|---|
| Letters look mirrored or swapped | Tick "Flip the hand image" |
| Poor accuracy for you | Plain background, even light, hand fills the crop; try the flip option |
| Slow or laggy | CPU speed is the limit: about 3–6 predictions per second is normal. Use a GPU (`device=0` in `hf_classifier.py`) |
| Two-handed letters wrong | Keep both hands fully in frame; the crop includes both |

**What it can't do:** words and moving signs (Phase 8) still need your own sequence model, and the board,
personalization and predictive features work the same in both modes. You can also keep both engines and compare
them in your report: pretrained on its own data vs. your participant-split model.

---

## Phase 8: Dynamic signs (2–3 weeks)

Do this only after Phase 7 is solid.

1. **Add a sequence recorder.** Copy `CollectPage.tsx` to `CollectSequencesPage.tsx`. Instead of single
   frames, keep a buffer of the last 30 `extractFeatures()` outputs (zeros if no hand) and save one
   `30 × 129` sample per take (press to start, auto-stop after 30 frames, about 1 second). Save as JSON or CSV with columns
   `participant_id,label,frame,f0..f128`. Collect 40–60 takes per word per participant, for 10–20 words.
2. **Train** in `ml/scripts/train_dynamic.py` (install `tensorflow` in the venv; use Python 3.10–3.12):

```python
model = keras.Sequential([
    keras.layers.Input(shape=(30, 129)),
    keras.layers.Masking(mask_value=0.0),
    keras.layers.GRU(64),            # swap for LSTM(64) and compare for your report
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(num_words, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
```
   Use the same participant-based split. Augment with small random shifts, time-stretch and noise.
3. **Export:** `pip install tensorflowjs` then
   `tensorflowjs_converter --input_format=keras model.keras frontend/public/models/dynamic/`.
   If the converter fights you, export the weights to JSON and write the GRU forward pass by hand, as with the MLP.
4. **Combine with static:** in the browser, compute average landmark movement over the last 10 frames.
   Low movement → static model; high movement → fill the 30-frame buffer and run the dynamic model.
   Put this in `ml/models/dynamicClassifier.ts`.

**Done when:** 10+ words are recognised on a held-out participant and the static letters still work.

## Phase 9: Communication board (3–5 days)

1. Create `frontend/src/data/phrases.json`:
   `{ "Basic": ["YES","NO","HELLO","THANKS","PLEASE","HELP"], "Food": ["WATER","FOOD","HUNGRY"], "Emergency": ["I NEED HELP","PLEASE HELP ME","I AM NOT FEELING WELL"] }`
2. Build `CategoryTabs`, `PhraseButton` (min 64 px tall, large text) and `CommunicationBoard` in
   `components/communication/`. Tapping a phrase appends it to a shared sentence.
3. Move `sentence` state into a small store (`store/sentenceStore.ts`, React context is enough) so the
   Translate page and the Board both feed the same `SentencePanel`.
4. Add the `/board` route and nav link in `App.tsx`.
5. **Emergency mode:** a red full-screen page at `/emergency` with 3 giant buttons that speak immediately
   (`speak(text)` on click). Keep the screen awake and make sure it works with no model loaded.

## Phase 10: Personalization (1–2 weeks)

1. `CustomSignRecorder`: reuse the collector's capture loop. The user names a sign and records about 30 samples.
2. Store samples in **IndexedDB** (not `localStorage`, which is too small), keyed by sign name.
3. `personalClassifier.ts`: k-nearest-neighbours on the 129-feature vectors (standardise with the same
   mean/scale as the main model). If the nearest custom sample is closer than a threshold (tune it, e.g. the
   95th percentile of within-sign distances), return the custom label; otherwise fall back to the generic model.
4. `CustomSignList`: list, rename, delete, re-record.

**Done when:** teaching a new sign takes under a minute and it is recognised right away.

## Phase 11: Predictive phrases (3–5 days)

1. `frontend/src/data/predictions.json` with the README's format (`"I WANT": ["WATER","FOOD","HELP"]`).
2. `services/suggest.ts`: take the last 1–2 words of the sentence, look them up, return up to 3 suggestions.
3. Show them as large tappable buttons under the sentence box.
4. Upgrade to frequency counts: when a sentence is spoken, increment `counts[prevWords][nextWord]`,
   saved in IndexedDB (or the backend in Phase 12), and sort suggestions by count.

## Phase 12: Backend (about 1 week)

```bash
cd backend && python -m venv ../.venv && pip install fastapi "uvicorn[standard]" sqlalchemy pydantic
```

Minimal `backend/app/main.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SignBridge API")
phrases: list[dict] = []  # replace with SQLAlchemy + SQLite

class Phrase(BaseModel):
    text: str
    category: str = "Basic"

@app.get("/api/phrases")
def list_phrases(): return phrases

@app.post("/api/phrases")
def add_phrase(p: Phrase):
    phrases.append(p.model_dump()); return p
```

Run with `uvicorn app.main:app --reload --port 8000`. The Vite dev server already proxies
`/api` to port 8000. Then, in order:
1. Replace the in-memory list with SQLAlchemy models (`Phrase`, `CustomSign`, `Feedback`, `UserProfile`) on SQLite.
2. Add the remaining README endpoints: `/api/user/profile`, `/api/custom-signs`, `/api/feedback`, `/api/models/current`.
3. Write pytest tests in `backend/tests/` using FastAPI's `TestClient`.
4. Add `frontend/src/services/api.ts` and point the Board, Custom Signs and Feedback to it.

Camera frames never go to the server, only phrases, custom-sign vectors, preferences and feedback.

## Phase 13: Testing and evaluation (1 week)

**ML:** the table and confusion matrix from `train_static.py`, repeated for the dynamic model.

**Real-world matrix.** Test with at least 3 people who were not in training, and record accuracy for each row:

| Condition | Variants |
|---|---|
| Lighting | bright, dim, backlit |
| Background | plain, cluttered |
| Distance | 0.5 m, 1 m, 1.5 m |
| Hand | right, left, both |
| Camera | laptop webcam, phone |

**Performance:** FPS (shown on screen), time from sign to prediction, model load time, and memory
(Chrome DevTools → Performance / Memory). Add `vitest` for `smoother.ts`, `featureExtractor.ts`
and `predictStatic` (the logic is already pure functions).

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| Black camera box with an error | Permission blocked, camera busy, or not on `localhost`/HTTPS |
| Stuck on "Loading hand tracking model…" | No internet to jsdelivr or googleapis. Download `hand_landmarker.task` into `public/models/` and change `MODEL_URL` in `handLandmarker.ts` |
| Predictions jump between two letters | Not enough variety in training takes; record more varied takes of both |
| Works for you, fails for friends | Too few participants. This is the most common issue |
| Left-handed signers fail | Record left-handed participants too, or add mirror augmentation (`x → -x`) in training |
| `train_static.py` says "random split" | Fewer than 3 participants. Fine for a smoke test, not for the report |
| Model changed but the browser shows old results | Hard refresh (Ctrl+Shift+R) |

## For the final report

Use the README's structure, plus these artefacts: `static_baselines.csv`, the confusion matrix,
the real-world test matrix, a short dataset datasheet (participants, consent, conditions,
who verified the ISL signs), and a limitations section (vocabulary size, lighting dependence, signers tested).
