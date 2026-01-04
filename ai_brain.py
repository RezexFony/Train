import json
import os
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TrainingAI:
    def __init__(self):
        self.training_data = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        
        # Initialize Groq client
        self.groq_client = Groq(
            api_key="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"
        )
        
        self.load_data()
        print(f"✅ AI loaded with {len(self.training_data)} learned conversations")
        print("🤖 Groq AI autopilot enabled - I'll teach myself!")
    
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
        
        print(f"🧠 Auto-learned: '{q}'")
        return True
    
    def get_groq_response(self, question):
        """Get intelligent response from Groq AI and auto-save it"""
        try:
            system_prompt = """You are an expert programming tutor specializing in:
- Roblox Lua scripting and game development
- Roblox executor scripting and GUI development  
- Lua programming for executors and scripts
- General programming languages (Python, JavaScript, C++, Java, etc.)
- Professional communication and technical explanations

Your teaching style:
- Clear, beginner-friendly explanations
- Provide practical code examples when relevant
- Explain concepts step-by-step
- Use proper code formatting
- Be encouraging and supportive
- Answer in a conversational, friendly tone
- Teach how to communicate professionally

When discussing Roblox/Lua:
- Focus on educational and development aspects
- Provide technical knowledge about Lua and Roblox API
- Explain how GUI systems work in executors
- Help with debugging and code optimization
- Teach scripting best practices

Keep responses helpful and informative."""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1500,
            )
            
            response = chat_completion.choices[0].message.content
            
            # AUTO-SAVE: Groq teaches the AI automatically!
            self.add_conversation(question.lower().strip(), response)
            print(f"✅ Groq AI taught me about: '{question[:50]}...'")
            
            return response
            
        except Exception as e:
            print(f"❌ Groq API Error: {e}")
            return f"❌ Sorry, I couldn't connect to my teacher (Groq AI). Error: {str(e)}"
    
    def get_response(self, question):
        """Get AI response - checks learned data first, then asks Groq"""
        q = question.lower().strip()
        
        # 1. Check exact match in learned data
        for item in self.training_data:
            if item['question'] == q:
                print(f"📚 Found in memory: '{q}'")
                return item['answer']
        
        # 2. Check for similar questions (fuzzy match)
        if self.vectors is not None and len(self.training_data) > 0:
            try:
                q_vector = self.vectorizer.transform([q])
                similarities = cosine_similarity(q_vector, self.vectors)[0]
                
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                # If very similar (>75% match), use learned answer
                if best_score > 0.75:
                    print(f"🔍 Found similar in memory (score: {best_score:.2f})")
                    return self.training_data[best_idx]['answer']
            except Exception as e:
                print(f"Error in similarity matching: {e}")
        
        # 3. No match found - Ask Groq AI and auto-learn!
        print(f"🤖 Learning from Groq AI: '{q}'")
        return self.get_groq_response(question)
    
    def get_stats(self):
        """Get AI statistics"""
        return {
            'knowledge_count': len(self.training_data),
            'ready': True,
            'groq_enabled': True
        }

# Create global AI instance
ai = TrainingAI()
