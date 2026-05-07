from content_manager import SQuADManager
from logic_engine import DualMetricAnalyzer
import random

# During development/training:
squad = SQuADManager('train-v2.0.json') 

# For your Midway Check/Validation (The current setup):
# squad = SQuADManager('dev-v2.0.json')
# 'Ear' trained on Sentiment140 
analyzer = DualMetricAnalyzer('training.1600000.processed.noemoticon.csv') 
session_history = []

def get_persona_comment(persona, switched):
    """Provides affective scaffolding mid-conversation[cite: 16]."""
    if switched and persona == "Supportive Mentor":
        return "\n[Supportive Mentor]: I noticed that was a bit tricky. Let's slow down and try a simpler concept to get your confidence back."
    if switched and persona == "Strict Recruiter":
        return "\n[Strict Recruiter]: Your technical depth is impressive. I'm increasing the difficulty to test your limits."
    
    mentor_msgs = ["Good point!", "Keep that momentum going.", "I like your perspective."]
    recruiter_msgs = ["Acknowledged. Next.", "Be more precise.", "Moving forward."]
    
    return f"\n[{persona}]: {random.choice(mentor_msgs if persona == 'Supportive Mentor' else recruiter_msgs)}"

print("="*60)
print("SENTIMENT-DRIVEN ADAPTIVE INTERVIEW COACH")
print("="*60)

# Display 50 fields for the user to decide from [cite: 46]
all_fields = squad.get_available_fields(limit=50)
print("\nAVAILABLE TECHNICAL FIELDS:")
for i, f in enumerate(all_fields):
    # Print in two columns for easier reading
    end_char = "\t\t" if (i+1) % 2 != 0 else "\n"
    print(f"{i+1:2}. {f[:25]:25}", end=end_char)

# Selection by number
choice = int(input("\n\nEnter the NUMBER of the field you want: ")) - 1
selected_field = all_fields[choice]
count_input = int(input(f"How many questions for the '{selected_field}' session? "))

current_persona = "Strict Recruiter" # Default starting state [cite: 39]
print(f"\n--- Initializing Session: {selected_field} ---")

for i in range(count_input):
    level = "hard" if current_persona == "Strict Recruiter" else "easy"
    question, correct_answer = squad.get_qa_pair(selected_field, level)
    
    print(f"\nQ{i+1} [{current_persona}]: {question}")
    user_response = input("Your Answer: ")
    
    # Dual-Metric Analysis: Sentiment (EQ) and Complexity (IQ) [cite: 16, 18]
    sentiment = analyzer.analyze_sentiment(user_response)
    complexity = analyzer.analyze_complexity(user_response)
    
    session_history.append({"question": question, "user_ans": user_response, "correct_ans": correct_answer})

    # Persona Switching Logic [cite: 37, 38]
    switched = False
    if sentiment == "Nervous" and current_persona == "Strict Recruiter":
        current_persona = "Supportive Mentor"
        switched = True
    elif sentiment == "Confident" and complexity == "Advanced" and current_persona == "Supportive Mentor":
        current_persona = "Strict Recruiter"
        switched = True

    print(get_persona_comment(current_persona, switched))

# Final Report Summary [cite: 46]
print("\n" + "="*60)
print("INTERVIEW COMPLETE - FINAL FEEDBACK REPORT")
print("="*60)
for idx, item in enumerate(session_history):
    print(f"\nQ{idx+1}: {item['question']}\nYour Answer: {item['user_ans']}\nCorrect Answer: {item['correct_ans']}\n" + "-"*30)