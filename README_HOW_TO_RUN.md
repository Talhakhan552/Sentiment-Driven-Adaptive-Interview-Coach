# 🎙️ Voice-Driven Adaptive Interview Coach — Setup & Run Guide

## What This Project Does

This is a **sentiment-driven AI interview coach**.  
It asks you technical questions (from the SQuAD dataset), listens to your spoken answer, analyses your **voice emotion** (nervous vs confident) and **answer complexity**, and then switches its coaching persona accordingly:

- 😤 **Strict Recruiter** — asks harder questions when you sound confident
- 🤝 **Supportive Mentor** — switches to easier questions when it detects nervousness

The **RAVDESS audio dataset** (`audio_speech_actors_01-24.zip`) is used to *calibrate* the emotion detector thresholds to your mic environment (optional but recommended).

---

## Project Files

```
interview_coach.py    ← Main entry point (run this)
content_manager.py    ← Loads SQuAD questions
logic_engine.py       ← Microphone capture + emotion analysis
requirements.txt      ← Python dependencies
dev-v2.0.json         ← SQuAD dataset (you need to download this)
audio_speech_actors_01-24/  ← RAVDESS dataset (extracted from zip)
```

---

## Step 1 — Install Python

Make sure you have **Python 3.9+**:
```bash
python --version
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows users**: PyAudio needs a special install:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
>
> **macOS users**: Install portaudio first:
> ```bash
> brew install portaudio
> pip install pyaudio
> ```
>
> **Linux users**:
> ```bash
> sudo apt-get install python3-pyaudio portaudio19-dev
> ```

---

## Step 3 — Download the SQuAD Dataset

The project reads from `dev-v2.0.json` (SQuAD 2.0 dev set).

Download it:
```bash
# Option A: direct download (Linux/macOS)
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# Option B: via Python
python -c "import urllib.request; urllib.request.urlretrieve('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json', 'dev-v2.0.json'); print('Done')"
```

Put `dev-v2.0.json` in the **same folder** as your Python files.

---

## Step 4 — Extract the RAVDESS Dataset

Unzip the uploaded file:
```bash
unzip audio_speech_actors_01-24.zip
```

You should now have a folder: `audio_speech_actors_01-24/` with subfolders `Actor_01/` through `Actor_24/`.

---

## Step 5 — Run the Coach

```bash
python interview_coach.py
```

### What happens:
1. It asks if you want to **calibrate** the emotion model on RAVDESS (type `y` and press Enter)
2. It shows available technical fields — pick one by number
3. Enter how many questions you want
4. For each question:
   - Read the question on screen
   - Speak your answer clearly into your mic (**you have 15 seconds**)
   - The system transcribes and analyses your voice
   - The persona adapts based on your emotional state
5. At the end you get a full evaluation report

---

## Key Fixes Made (vs Original)

| Issue | Original | Fixed |
|---|---|---|
| Wait time before listening starts | 15 seconds (too long) | **5 seconds** |
| Max answer duration | 30 seconds (too slow) | **15 seconds** |
| Ambient noise calibration | 0.5 seconds | **0.3 seconds** |
| Emotion features | Only MFCC std deviation | **4 features** (MFCC, RMS energy, spectral centroid, ZCR) — majority vote |
| RAVDESS dataset | Not used at all | **train_on_ravdess()** computes calibrated thresholds |
| Difficulty levels | easy / hard only | **easy / intermediate / hard** |
| Complexity scoring | Only unique word count | **Unique ratio + length combined** |
| Persona switch condition | Confidence + Advanced only | **Confidence + Intermediate or Advanced** |
| Final report | Questions and answers only | **Stats summary + tips added** |

---

## Troubleshooting

**"No module named 'pyaudio'"**  
→ See Step 2 platform-specific install above.

**"Speech API Connection Error"**  
→ You need internet. The Google Speech API is used for transcription.

**Microphone not detected**  
→ Check your OS mic permissions. On macOS: System Settings → Privacy → Microphone → allow Terminal.

**Questions move too fast / too slow**  
→ Adjust `phrase_time_limit` in `logic_engine.py` line ~24. Default is now 15 seconds.

**Emotion always shows Nervous**  
→ Run the RAVDESS calibration step (type `y` at the first prompt) to get thresholds tuned to your voice.
