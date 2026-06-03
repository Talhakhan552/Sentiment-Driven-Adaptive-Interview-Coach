import eel
import time
import random
import json
import requests
from content_manager import SQuADManager
from logic_engine import VoiceAffectiveEngine

eel.init('web')
squad = SQuADManager('dev-v2.0.json')
analyzer = VoiceAffectiveEngine()

all_fields = squad.get_available_fields(limit=50)
current_persona = "Strict Recruiter"
stop_requested = False

# ─────────────────────────────────────────────
#  AI-powered answer scoring
#  Uses Claude to understand synonyms, paraphrases,
#  and partial credit — not just exact keyword match.
#  Falls back to keyword overlap if API call fails.
# ─────────────────────────────────────────────


def ai_answer_check(question, user_ans, correct_ans):
    if not user_ans or user_ans.startswith("["):
        return 0, "No answer captured.", ""

    prompt = f"""You are an interview answer evaluator.

Question: {question}
Expected answer: {correct_ans}
Student's spoken answer: {user_ans}

Evaluate how correct the student's answer is. Rules:
- Synonyms and paraphrases count as correct (e.g. "gravitational pull" = "force of gravity")
- Partial answers deserve partial credit
- Extra correct details should NOT reduce the score
- Being approximately right is better than completely wrong

Respond ONLY with a JSON object, no markdown, no extra text:
{{
  "score": <integer 0-100>,
  "verdict": "<Correct / Partially Correct / Incorrect>",
  "feedback": "<one sentence explaining the score>",
  "what_was_right": "<what the student got right, or empty string>",
  "what_was_missing": "<key concept missing, or empty string>"
}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15
        )
        data = response.json()
        raw = data["content"][0]["text"].strip()

        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        score = max(0, min(int(result.get("score", 0)), 100))
        verdict = result.get("verdict", "")
        fb = result.get("feedback", "")
        right = result.get("what_was_right", "")
        missing = result.get("what_was_missing", "")

        icon = "✅" if score >= 60 else "🟡" if score >= 30 else "❌"
        detail = ""
        if right:
            detail += f" ✔ {right}."
        if missing:
            detail += f" Missing: {missing}."

        return score, f"{icon} {verdict} — {fb}{detail}", ""

    except Exception as e:
        print(f"[AI scorer fallback triggered: {e}]")
        return _keyword_fallback(user_ans, correct_ans)


def _keyword_fallback(user_ans, correct_ans):
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
                 "to", "and", "or", "it", "its", "this", "that", "for", "with", "by"}
    user_kw = set(user_ans.lower().split()) - stopwords
    correct_kw = set(correct_ans.lower().split()) - stopwords
    if not correct_kw:
        return 50, "Could not evaluate (reference too short).", ""
    overlap = user_kw & correct_kw
    score = min(int(len(overlap) / len(correct_kw) * 100), 100)
    if score >= 60:
        fb = f"✅ Good — matched {len(overlap)}/{len(correct_kw)} key terms."
    elif score >= 30:
        fb = f"🟡 Partial — matched {len(overlap)}/{len(correct_kw)} key terms."
    else:
        fb = f"❌ Missed most key terms ({len(overlap)}/{len(correct_kw)} matched)."
    return score, fb, ""

# ─────────────────────────────────────────────
#  Coaching comments
# ─────────────────────────────────────────────


def get_coaching_comment(persona, switched, voice_state, score, complexity):
    if switched and persona == "Supportive Mentor":
        return "I noticed some hesitation — that's okay. Let's slow down and try a simpler concept together."
    if switched and persona == "Strict Recruiter":
        return "Excellent delivery — confident and clear! Time to step it up. Next question is harder."

    if persona == "Supportive Mentor":
        if voice_state == "Nervous" and score < 30:
            msgs = [
                "It's okay to feel nervous — everyone does at first. Focus on one keyword at a time.",
                "Don't worry about perfection. Even a partial answer shows you're thinking!",
                "Take a deep breath before answering next time. You've got this. 💪",
            ]
        elif voice_state == "Nervous" and score >= 30:
            msgs = [
                "Your answer had good points! Your voice will steady as you build confidence.",
                "See? You knew more than you thought. Keep going — you're improving.",
                "Good effort! Your content is solid. Just speak a little slower next time.",
            ]
        elif voice_state == "Confident" and score < 30:
            msgs = [
                "Great confident tone! Now let's sharpen the content — try to name specific terms.",
                "You sound assured — now make sure your answer matches that energy in detail.",
            ]
        else:
            msgs = [
                "That was a strong answer! You're clearly finding your rhythm. 🔥",
                "Excellent! Confident delivery AND solid content. Keep that up!",
            ]
    else:
        if voice_state == "Nervous" and score < 30:
            msgs = [
                "In a real interview, hesitation signals unpreparedness. Review this topic.",
                "That answer needs more substance. Focus — what are the core concepts here?",
            ]
        elif voice_state == "Nervous" and score >= 30:
            msgs = [
                "Content was acceptable, but your delivery needs more confidence. Work on that.",
                "You have the knowledge — now project it. Speak with conviction.",
            ]
        elif voice_state == "Confident" and score < 30:
            msgs = [
                "You sound confident but missed the key points. Substance matters more than style.",
                "Good delivery, weak content. Study this topic more carefully.",
            ]
        else:
            msgs = [
                "Strong answer. That's interview-ready. Next question will test you harder.",
                "Good — concise, accurate, confident. Exactly what a recruiter wants.",
            ]

    return random.choice(msgs)

# ─────────────────────────────────────────────
#  Eel exposed functions
# ─────────────────────────────────────────────


@eel.expose
def get_initial_data():
    return all_fields


@eel.expose
def start_interview_from_ui(selected_field, num_questions):
    eel.spawn(run_interview_logic, selected_field, int(num_questions))


@eel.expose
def stop_interview_from_ui():
    global stop_requested
    stop_requested = True

# ─────────────────────────────────────────────
#  Main interview loop
# ─────────────────────────────────────────────


def run_interview_logic(selected_field, num_questions):
    global current_persona, stop_requested
    current_persona = "Strict Recruiter"
    stop_requested = False
    session_scores = []

    eel.update_chat(
        f"🚀 Started <b>{selected_field}</b> module. {num_questions} questions.")

    for i in range(num_questions):
        if stop_requested:
            eel.update_chat("🛑 Session aborted by user.")
            eel.update_status("ABORTED", "#ef4444", False)
            break

        eel.update_progress(i + 1, num_questions)
        level = "hard" if current_persona == "Strict Recruiter" else "easy"
        question, correct_answer = squad.get_qa_pair(selected_field, level)
        eel.update_question(question)
        eel.update_status("LISTENING...", "#ef4444", True)
        eel.update_chat(
            "<span style='color:#94a3b8'>────────────────────────────</span>")
        eel.update_chat(f"<b>Q{i+1}:</b> {question}")

        user_transcript, saved_audio = analyzer.listen_and_transcribe()

        if stop_requested:
            eel.update_chat("🛑 Session aborted by user.")
            eel.update_status("ABORTED", "#ef4444", False)
            break

        eel.update_status("SCORING...", "#0ea5e9", False)
        eel.update_transcript(user_transcript)

        # ── AI scoring ───────────────────────────────────────────────
        eel.update_chat(
            "⏳ <span style='color:#94a3b8'>Evaluating your answer with AI...</span>")
        score, feedback, _ = ai_answer_check(
            question, user_transcript, correct_answer)
        session_scores.append(score)

        score_color = "#10b981" if score >= 60 else "#f59e0b" if score >= 30 else "#ef4444"
        eel.update_chat(
            f"📖 <span style='color:#94a3b8'>Expected:</span> "
            f"<span style='color:#e2e8f0'>{correct_answer}</span>"
        )
        eel.update_chat(
            f"📊 <span style='color:{score_color}'><b>{score}/100</b></span> — {feedback}"
        )
        eel.update_score(score, score_color, feedback)

        # ── Voice + complexity ────────────────────────────────────────
        eel.update_status("ANALYZING...", "#0ea5e9", False)
        voice_state = analyzer.analyze_voice_emotion(
            saved_audio, transcript=user_transcript)
        complexity = analyzer.analyze_complexity(user_transcript)

        if voice_state == "No Audio":
            eel.update_metrics("No Audio", "#475569", complexity)
            eel.update_chat(
                "⚠️ Mic didn't capture audio — check your microphone.")
        else:
            emo_color = "#ef4444" if voice_state == "Nervous" else "#10b981"
            eel.update_metrics(voice_state, emo_color, complexity)

        # ── Persona switching ─────────────────────────────────────────
        switched = False
        if voice_state != "No Audio":
            if voice_state == "Nervous" and current_persona == "Strict Recruiter":
                current_persona = "Supportive Mentor"
                switched = True
            elif (voice_state == "Confident"
                  and complexity in ("Advanced", "Intermediate")
                  and current_persona == "Supportive Mentor"):
                current_persona = "Strict Recruiter"
                switched = True

        persona_color = "#10b981" if current_persona == "Supportive Mentor" else "#fbbf24"
        eel.update_persona(current_persona, persona_color)

        # ── Coaching comment ──────────────────────────────────────────
        if voice_state != "No Audio":
            comment = get_coaching_comment(
                current_persona, switched, voice_state, score, complexity)
            icon = "🤝" if current_persona == "Supportive Mentor" else "💼"
            eel.update_chat(
                f"<span style='color:{persona_color}'>"
                f"[{current_persona} {icon}]: {comment}</span>"
            )

        time.sleep(1)

    # ── Final summary ─────────────────────────────────────────────────
    if not stop_requested and session_scores:
        avg = int(sum(session_scores) / len(session_scores))
        avg_color = "#10b981" if avg >= 60 else "#f59e0b" if avg >= 30 else "#ef4444"
        eel.update_chat(
            "<span style='color:#94a3b8'>════════════════════════════</span>")
        eel.update_chat(
            f"🎯 <b>SESSION COMPLETE</b> — "
            f"Average score: <span style='color:{avg_color}'><b>{avg}/100</b></span>")
        if avg >= 60:
            eel.update_chat(
                "✅ Strong session! Your answers covered key concepts well.")
        elif avg >= 30:
            eel.update_chat(
                "🟡 Decent effort. Review the expected answers above.")
        else:
            eel.update_chat(
                "💡 Study the model answers above and focus on key terms.")
        eel.update_status("COMPLETE", "#10b981", False)

    eel.session_finished()


eel.start('index.html', size=(1200, 800))
