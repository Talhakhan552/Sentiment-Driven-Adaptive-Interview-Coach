import sys
import time
from content_manager import SQuADManager
from logic_engine import VoiceAffectiveEngine
import random

# ─────────────────────────────────────────────
#  AI Streaming Effect Helper
# ─────────────────────────────────────────────


def stream_print(text, delay=0.015):
    """Simulates AI streaming text sequentially to the terminal."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# ─────────────────────────────────────────────
#  Coaching Comments
# ─────────────────────────────────────────────


def get_coaching_comment(persona, switched, voice_state, score, complexity):
    """Context-aware mid-session coaching comment shown after every answer."""

    # ── Persona switch announcements ─────────────────────────────
    if switched and persona == "Supportive Mentor":
        return (
            "\n╔══ [Supportive Mentor 🤝] ══════════════════════════════╗\n"
            "  I noticed some hesitation — that's completely okay.\n"
            "  Let's slow down. Take a breath before the next question.\n"
            "╚════════════════════════════════════════════════════════╝"
        )
    if switched and persona == "Strict Recruiter":
        return (
            "\n╔══ [Strict Recruiter 💼] ════════════════════════════════╗\n"
            "  Excellent delivery — confident and clear!\n"
            "  Time to step it up. Next question is harder.\n"
            "╚════════════════════════════════════════════════════════╝"
        )

    # ── Supportive Mentor reactions ──────────────────────────────
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
                "Confidence is half the battle. Work on adding more key points to your answers.",
            ]
        else:  # confident + good score
            msgs = [
                "That was a strong answer! You're clearly finding your rhythm. 🔥",
                "Excellent! Confident delivery AND solid content. Keep that up!",
                "Well done — you're doing great. One more like that and we level up the difficulty.",
            ]

    # ── Strict Recruiter reactions ────────────────────────────────
    else:
        if voice_state == "Nervous" and score < 30:
            msgs = [
                "In a real interview, hesitation signals unpreparedness. Review this topic.",
                "That answer needs more substance. Focus — what are the core concepts here?",
                "Interviewers notice nerves. Take a moment to collect your thoughts before speaking.",
            ]
        elif voice_state == "Nervous" and score >= 30:
            msgs = [
                "Content was acceptable, but your delivery needs more confidence. Work on that.",
                "You have the knowledge — now project it. Speak with conviction.",
                "Partial credit on content. Your vocal delivery still needs work under pressure.",
            ]
        elif voice_state == "Confident" and score < 30:
            msgs = [
                "You sound confident but missed the key points. Substance matters more than style.",
                "Good delivery, weak content. In an interview that won't cut it — study this topic.",
                "Presentation was fine. But interviewers want correct answers, not just confidence.",
            ]
        else:  # confident + good score
            msgs = [
                "Strong answer. That's interview-ready. Next question will test you harder.",
                "Good — concise, accurate, confident. Exactly what a recruiter wants to hear.",
                "Solid response. Complexity: {c}. Keep this standard for the rest of the session.".format(
                    c=complexity),
            ]

    return f"\n[{persona}]: {random.choice(msgs)}"


def simple_answer_check(user_ans, correct_ans):
    """Keyword overlap check. Returns (score 0-100, feedback string)."""
    if user_ans.startswith("["):
        return 0, "❌ No answer captured."

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
                 "to", "and", "or", "it", "its", "this", "that", "for", "with", "by"}
    user_kw = set(user_ans.lower().split()) - stopwords
    correct_kw = set(correct_ans.lower().split()) - stopwords

    if not correct_kw:
        return 50, "⚠️  Could not evaluate (reference answer too short)."

    overlap = user_kw & correct_kw
    score = min(int(len(overlap) / len(correct_kw) * 100), 100)

    if score >= 60:
        feedback = f"✅ Good answer! You covered {len(overlap)}/{len(correct_kw)} key points."
    elif score >= 30:
        feedback = f"🟡 Partial answer. You hit {len(overlap)}/{len(correct_kw)} key points."
    else:
        feedback = f"❌ Missed most key points ({len(overlap)}/{len(correct_kw)} matched)."

    return score, feedback


# ─────────────────────────────────────────────
#  Initialise
# ─────────────────────────────────────────────
squad = SQuADManager('dev-v2.0.json')
analyzer = VoiceAffectiveEngine()
session_history = []

print("=" * 65)
print("🎙️  VOICE-DRIVEN MULTIMODAL ADAPTIVE INTERVIEW COACH")
print("=" * 65)

# ─────────────────────────────────────────────
#  Optional RAVDESS calibration
# ─────────────────────────────────────────────
calibrate = input(
    "\nCalibrate emotion model on RAVDESS dataset first? (y/n): ").strip().lower()
if calibrate == 'y':
    ravdess_path = input(
        "Enter path to RAVDESS folder [audio_speech_actors_01-24]: ").strip()
    if not ravdess_path:
        ravdess_path = "audio_speech_actors_01-24"
    analyzer.train_on_ravdess(ravdess_path)

# ─────────────────────────────────────────────
#  Field and session setup
# ─────────────────────────────────────────────
all_fields = squad.get_available_fields(limit=50)
print("\nAVAILABLE TECHNICAL FIELDS:")
for i, f in enumerate(all_fields):
    end_char = "\t\t" if (i + 1) % 2 != 0 else "\n"
    print(f"{i+1:2}. {f[:25]:25}", end=end_char)

try:
    choice = int(input("\n\nEnter the NUMBER of the technical domain: ")) - 1
    selected_field = all_fields[choice]
    count_input = int(input(f"How many questions for '{selected_field}'? "))
except (ValueError, IndexError):
    print("Invalid input. Defaulting to field 1 with 3 questions.")
    selected_field = all_fields[0]
    count_input = 3

current_persona = "Strict Recruiter"
print(f"\n--- Starting Live Session: {selected_field} ---")
print("💡 Speak your answer. A 5-second pause moves to the next question.\n")

# ─────────────────────────────────────────────
#  Main session loop
# ─────────────────────────────────────────────
for i in range(count_input):

    level = "hard" if current_persona == "Strict Recruiter" else \
            "easy" if current_persona == "Supportive Mentor" else \
            "intermediate"

    question, correct_answer = squad.get_qa_pair(selected_field, level)

    print(f"\n{'═'*60}")
    print(f"  Q{i+1} [{current_persona}]")
    print(f"  {question}")
    print(f"{'─'*60}")

    # ── Live Voice Capture ───────────────────────
    user_transcript, saved_audio = analyzer.listen_and_transcribe()

    # ── Show transcript + model answer immediately (Streaming!) ──
    print(f"\n🗣️  Your answer  : \"{user_transcript}\"")
    stream_print(f"📖  Model answer : \"{correct_answer}\"")

    # ── Correctness score ────────────────────────
    score, feedback = simple_answer_check(user_transcript, correct_answer)
    stream_print(f"📊  Score        : {score}/100  —  {feedback}")

    # ── Emotion + complexity ──────────────────────
    voice_state = analyzer.analyze_voice_emotion(
        saved_audio, transcript=user_transcript)
    complexity = analyzer.analyze_complexity(user_transcript)

    if voice_state == "No Audio":
        print(f"🎙️  Voice state  : ⚠️  No audio captured")
        print(f"   Complexity: {complexity}")
    else:
        print(f"🎙️  Voice state  : {voice_state}  |  Complexity: {complexity}")

    session_history.append({
        "question":    question,
        "user_ans":    user_transcript,
        "correct_ans": correct_answer,
        "score":       score,
        "feedback":    feedback,
        "voice_state": voice_state,
        "complexity":  complexity,
        "persona":     current_persona,
    })

    # ── Persona switching ────────────
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

    # ── Context-aware coaching comment (Streaming!) ──────────
    if voice_state == "No Audio":
        stream_print(
            "\n⚠️  [Coach]: Couldn't hear you clearly. Make sure your mic is working.")
    else:
        stream_print(get_coaching_comment(current_persona,
                     switched, voice_state, score, complexity))

# ─────────────────────────────────────────────
#  Final Evaluation Report
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("🎯  SESSION COMPLETE — EVALUATION FEEDBACK REPORT")
print("=" * 65)

nervous_count = sum(
    1 for h in session_history if h['voice_state'] == "Nervous")
confident_count = sum(
    1 for h in session_history if h['voice_state'] == "Confident")
no_audio_count = sum(
    1 for h in session_history if h['voice_state'] == "No Audio")
advanced_count = sum(
    1 for h in session_history if h['complexity'] == "Advanced")
avg_score = sum(h['score'] for h in session_history) / \
    max(len(session_history), 1)

for idx, item in enumerate(session_history):
    print(f"\n  Q{idx+1}: {item['question']}")
    print(f"  Persona         : {item['persona']}")
    print(f"  Your Answer     : {item['user_ans']}")
    print(f"  Expected Answer : {item['correct_ans']}")
    print(f"  Score           : {item['score']}/100  —  {item['feedback']}")
    print(
        f"  Voice State     : {item['voice_state']}  |  Complexity: {item['complexity']}")
    print("  " + "─" * 50)

print(f"\n📈 Overall Stats")
print(f"  Average score     : {avg_score:.0f}/100")
print(f"  Confident answers : {confident_count}/{len(session_history)}")
print(f"  Nervous answers   : {nervous_count}/{len(session_history)}")
print(f"  No audio captured : {no_audio_count}/{len(session_history)}")
print(f"  Advanced responses: {advanced_count}/{len(session_history)}\n")

# ── Dynamic Streaming Final Feedback ──

if avg_score >= 60:
    stream_print(
        "✅ Strong session! Your answers covered the key concepts well.")
elif avg_score >= 30:
    stream_print(
        "🟡 Decent effort. Review the expected answers above and practice the specifics.")
else:
    stream_print(
        "💡 Tip: Study the model answers above. Focus heavily on remembering specific technical keywords.")

# The "Confident but Wrong" cross-check
confidence_ratio = confident_count / max(len(session_history), 1)

if confidence_ratio >= 0.5:
    if avg_score < 40:
        stream_print("⚠️ CRITICAL FEEDBACK: You sounded highly confident, but your technical accuracy was very low. In a real interview, providing confident but incorrect answers is a major red flag that indicates a lack of self-awareness. Prioritise studying the actual concepts before focusing on your presentation style.")
    else:
        stream_print(
            "✅ Excellent job maintaining confidence throughout the session, and your knowledge backed it up perfectly.")
else:
    if avg_score >= 60:
        stream_print("💡 Tip: Your technical knowledge is solid, but your vocal delivery showed nerves. Trust what you know! Practice speaking at a steady, slower pace to project the confidence you deserve.")
    else:
        stream_print("💡 Tip: Practice slow, deliberate speech to reduce vocal nervousness. Re-reading the material will also naturally boost your baseline confidence.")
