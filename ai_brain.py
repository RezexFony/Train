import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TrainingAI:
    def __init__(self):
        self.training_data = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.load_data()
        print(f"✅ AI loaded with {len(self.training_data)} learned conversations")
    
    def load_data(self):
        """Load training data from JSON file"""
        if os.path.exists('training_data.json'):
            try:
                with open('training_data.json', 'r') as f:
                    self.training_data = json.load(f)
                if len(self.training_data) > 0:
                    self.update_vectors()
            except Exception as e:
                print(f"Error loading data: {e}")
                self.training_data = []
    
    def save_data(self):
        """Save training data to JSON file"""
        with open('training_data.json', 'w') as f:
            json.dump(self.training_data, f, indent=2)
    
    def update_vectors(self):
        """Update TF-IDF vectors for similarity matching"""
        if len(self.training_data) > 0:
            questions = [item['question'] for item in self.training_data]
            self.vectors = self.vectorizer.fit_transform(questions)
    
    def add_conversation(self, question, answer):
        """Add new Q&A pair and retrain"""
        q = question.lower().strip()
        a = answer.strip()
        
        # Add to training data
        self.training_data.append({
            'question': q,
            'answer': a
        })
        
        # Save and update
        self.save_data()
        self.update_vectors()
        
        print(f"🧠 Learned: '{q}' → '{a}'")
        return True
    
    def get_response(self, question):
        """Get AI response to a question"""
        q = question.lower().strip()
        
        # No data yet
        if len(self.training_data) == 0:
            return None
        
        # Check exact match
        for item in self.training_data:
            if item['question'] == q:
                return item['answer']
        
        # Find similar question using cosine similarity
        if self.vectors is not None and len(self.training_data) > 0:
            try:
                q_vector = self.vectorizer.transform([q])
                similarities = cosine_similarity(q_vector, self.vectors)[0]
                
                # Get best match if similarity > threshold
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score > 0.5:  # 50% similarity threshold
                    return self.training_data[best_idx]['answer']
            except Exception as e:
                print(f"Error in similarity matching: {e}")
        
        # No match found
        return None
    
    def get_stats(self):
        """Get AI statistics"""
        return {
            'knowledge_count': len(self.training_data),
            'ready': len(self.training_data) > 0
        }

# Create global AI instance
ai = TrainingAI()
