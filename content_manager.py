import json
import random

class SQuADManager:
    def __init__(self, file_path):
        # Using UTF-8 to handle the diverse technical vocabulary in SQuAD [cite: 23]
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.fields = [article['title'].replace('_', ' ') for article in self.data['data']]

    def get_available_fields(self, limit=50):
        """Returns a defined number of technical domains from the dataset[cite: 24]."""
        return self.fields[:limit]

    def get_qa_pair(self, field_name, level="easy"):
        # Locates the specific technical domain chosen by the user [cite: 10]
        field_data = next((item for item in self.data['data'] if item['title'].replace('_', ' ') == field_name), None)
        
        if not field_data:
            return "No question found.", "N/A"

        all_qas = []
        for paragraph in field_data['paragraphs']:
            for qas in paragraph['qas']:
                if qas['answers']: 
                    all_qas.append({
                        "question": qas['question'],
                        "answer": qas['answers'][0]['text']
                    })

        # Affective Scaffolding: Shifts technical load based on persona [cite: 16, 18]
        filtered = [qa for qa in all_qas if (len(qa['question'].split()) < 10 if level == "easy" else len(qa['question'].split()) >= 10)]
        
        selected = random.choice(filtered if filtered else all_qas)
        return selected['question'], selected['answer']