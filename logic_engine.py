import speech_recognition as sr
import librosa
import numpy as np
import warnings
import os
import json
import time
import threading

warnings.filterwarnings("ignore", category=UserWarning)

CACHE_FILE = "ravdess_calibration_cache.json"


class VoiceAffectiveEngine:
    def __init__(self):
        self.mfcc_threshold = 25.0
        self._load_cache()

    # ─────────────────────────────────────────────
    #  Cache helpers
    # ─────────────────────────────────────────────
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                self.mfcc_threshold = data.get("mfcc_threshold", 25.0)
                print(
                    f"✅ Loaded saved calibration (threshold={self.mfcc_threshold:.2f})")
            except Exception:
                pass

    def _save_cache(self, threshold):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"mfcc_threshold": threshold}, f)
            print(f"💾 Calibration saved — won't re-run next time.")
        except Exception as e:
            print(f"[Warning] Could not save cache: {e}")

    # ─────────────────────────────────────────────
    #  Microphone capture
    # ─────────────────────────────────────────────
    def listen_and_transcribe(self):
        recognizer = sr.Recognizer()

        # Re-enable dynamic threshold so it adapts to YOUR mic volume
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 100
        recognizer.pause_threshold = 5.0
        recognizer.non_speaking_duration = 5.0
        recognizer.phrase_threshold = 0.1

        # Calibrate against room noise first
        print("\n🎤 Calibrating mic to room noise...")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            live_threshold = recognizer.energy_threshold

        print(f"🎤 Listening... speak now. Pause 5 seconds to finish.")
        print(f"   (noise floor set to {live_threshold:.0f})")

        # Spinner while listening
        self._listening = True

        def _spin():
            symbols = ["   .  ", "   .. ", "   ..."]
            i = 0
            while self._listening:
                print(f"\r{symbols[i % len(symbols)]}", end="", flush=True)
                i += 1
                time.sleep(0.6)
        spinner = threading.Thread(target=_spin, daemon=True)
        spinner.start()

        try:
            with sr.Microphone() as source:
                recognizer.energy_threshold = live_threshold
                audio = recognizer.listen(
                    source, timeout=10, phrase_time_limit=180)
        except sr.WaitTimeoutError:
            self._listening = False
            print("\r⚠️  No speech heard within 10 seconds.          ")
            return "[No speech detected]", None
        finally:
            self._listening = False

        print("\r⏳ Transcribing...                              ")

        try:
            text_transcript = recognizer.recognize_google(audio)
            audio_path = "live_user_response.wav"
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            return text_transcript, audio_path

        except sr.UnknownValueError:
            return "[Could not understand audio]", None
        except sr.RequestError:
            return "[Speech API error — check internet]", None

    # ─────────────────────────────────────────────
    #  Emotion analysis
    # ─────────────────────────────────────────────
    def analyze_voice_emotion(self, audio_path, transcript=""):
        # No audio captured at all
        if not audio_path or not os.path.exists(audio_path):
            return "No Audio"

        try:
            print("⏳ Analysing vocal tone...")
            y, sr_rate = librosa.load(audio_path, duration=8.0)

            if len(y) < sr_rate * 0.3:
                return "No Audio"

            # ── Feature extraction ────────────────────────────────────
            mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
            mfcc_std = np.mean(np.std(mfccs, axis=1))

            rms = np.mean(librosa.feature.rms(y=y))

            centroid = np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr_rate))

            zcr = np.mean(
                librosa.feature.zero_crossing_rate(y))

            spec = np.abs(librosa.stft(y))
            flux = np.mean(np.diff(spec, axis=1) ** 2)

            # ── Stutter detection from audio ──────────────────────────
            zcr_frames = librosa.feature.zero_crossing_rate(y)[0]
            zcr_spikes = np.sum(zcr_frames > 0.15) / max(len(zcr_frames), 1)
            audio_stutter = (zcr_spikes > 0.20 and rms < 0.05)

            # ── Transcript stutter & Advanced Tentative Language ───────────────────────────
            transcript_stutter = False
            tentative_language = False

            if transcript and not transcript.startswith("["):
                trans_lower = transcript.lower()
                # Pads for accurate exact-phrase matching
                padded_trans = f" {trans_lower} "
                words = trans_lower.split()

                # 1. Expanded Filler & Stalling Check
                fillers = [
                    " um ", " uh ", " er ", " ah ", " like ", " you know ",
                    " so ", " basically ", " i mean ", " well ", " kind of ",
                    " sort of ", " actually ", " literally ", " just "
                ]

                # Count how many of these stalling phrases appear
                filler_count = sum(padded_trans.count(f) for f in fillers)

                # Check for direct repetition (e.g., "the the", "I I")
                repeated = sum(1 for j in range(1, len(words))
                               if words[j] == words[j-1])

                transcript_stutter = (filler_count >= 2 or repeated >= 1)

                # 2. Expanded Tentative, Hedging, & Apologetic Language
                tentative_phrases = [
                    # Hedges / Doubt
                    "guess", "maybe", "not sure", "think", "probably", "might",
                    "suppose", "perhaps", "could be", "possibly", "pretty sure",
                    "something like", "believe", "if i recall", "if i remember",
                    # Defeatist / Apologetic
                    "don't know", "dont know", "can't remember", "cant remember",
                    "forget", "sorry", "apologies", "my bad"
                ]

                has_tentative_words = any(
                    p in trans_lower for p in tentative_phrases)
                has_question_inflection = "?" in transcript

                tentative_language = has_tentative_words or has_question_inflection

            # ── Vote tally ────────────────────────────────────────────
            nervous_votes = 0
            nervous_votes += 1 if mfcc_std > self.mfcc_threshold else 0
            nervous_votes += 1 if rms < 0.03 else 0
            nervous_votes += 1 if centroid > 3000 else 0
            nervous_votes += 1 if zcr > 0.08 else 0
            nervous_votes += 1 if flux > 500 else 0
            nervous_votes += 2 if audio_stutter else 0
            nervous_votes += 1 if transcript_stutter else 0
            # Strong signal for nervous vocabulary (guess, maybe, sorry)
            nervous_votes += 2 if tentative_language else 0

            return "Nervous" if nervous_votes >= 3 else "Confident"

        except Exception as e:
            print(f"[Debug] Audio analysis error: {e}")
            return "Nervous"

    # ─────────────────────────────────────────────
    #  Complexity scoring
    # ─────────────────────────────────────────────
    def analyze_complexity(self, text):
        if not text.strip() or text.startswith("["):
            return "Basic"
        words = text.lower().split()
        if not words:
            return "Basic"
        unique_ratio = len(set(words)) / len(words)
        if len(words) >= 10 and unique_ratio >= 0.6:
            return "Advanced"
        elif len(words) >= 6:
            return "Intermediate"
        return "Basic"

    # ─────────────────────────────────────────────
    #  RAVDESS calibration
    # ─────────────────────────────────────────────
    def train_on_ravdess(self, ravdess_dir):
        if os.path.exists(CACHE_FILE):
            self._load_cache()
            return self.mfcc_threshold

        if not os.path.isdir(ravdess_dir):
            print(f"[Warning] RAVDESS folder not found: {ravdess_dir}")
            return None

        all_actors = sorted([d for d in os.listdir(ravdess_dir)
                             if os.path.isdir(os.path.join(ravdess_dir, d))])
        actors_to_scan = all_actors[:10]
        total_actors = len(actors_to_scan)
        nervous_stds, confident_stds = [], []

        print(
            f"\n📂 Scanning {total_actors} actors from RAVDESS (runs once, cached after)...\n")

        for actor_idx, actor_folder in enumerate(actors_to_scan, 1):
            actor_path = os.path.join(ravdess_dir, actor_folder)
            done = int((actor_idx / total_actors) * 20)
            bar = "█" * done + "░" * (20 - done)
            pct = int((actor_idx / total_actors) * 100)
            print(f"\r   [{bar}] {pct:3d}%  Actor {actor_idx:02d}/{total_actors}",
                  end="", flush=True)

            for fname in os.listdir(actor_path):
                if not fname.endswith(".wav"):
                    continue
                parts = fname.replace(".wav", "").split("-")
                if len(parts) < 3:
                    continue
                emotion_code = int(parts[2])
                fpath = os.path.join(actor_path, fname)
                try:
                    y, sr_rate = librosa.load(fpath, duration=2.0)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
                    std_val = np.mean(np.std(mfccs, axis=1))
                    if emotion_code in [4, 6]:
                        nervous_stds.append(std_val)
                    elif emotion_code in [2, 3]:
                        confident_stds.append(std_val)
                except Exception:
                    pass
        print()

        if nervous_stds and confident_stds:
            threshold = (np.mean(nervous_stds) + np.mean(confident_stds)) / 2
            self.mfcc_threshold = threshold
            self._save_cache(threshold)
            print(
                f"\n📊 Calibration complete — threshold set to {threshold:.2f}")
            return threshold
        else:
            print("[Warning] Not enough RAVDESS samples found.")
            return None
