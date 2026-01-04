import json
import os
import threading
import time
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TrainingAI:
    def __init__(self):
        self.training_data = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.groq_mode = True  # True = Learning Mode, False = Test Mode
        self.afk_mode = False  # AFK Auto-Training Mode
        self.afk_thread = None
        self.training_active = False
        
        # Initialize Groq client
        self.groq_client = Groq(
            api_key="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"
        )
        
        # Topics for AFK training
        self.training_topics = [
            # Roblox Lua Basics
            "how to create a part in roblox lua",
            "how to make a gui in roblox",
            "what is a localscript vs script in roblox",
            "how to detect player click in roblox",
            "how to make a teleport script in roblox",
            "how to use tweenservice in roblox",
            "how to make a speed boost script",
            "how to create a button in roblox gui",
            "what is remotevent in roblox",
            "how to detect player join in roblox",
            
            # Roblox Executor/GUI
            "how to create an executor gui",
            "how to make a script executor interface",
            "what are the main components of a roblox executor",
            "how to create draggable gui in roblox",
            "how to make a code editor in roblox gui",
            "how to create tabs in roblox executor",
            "how to save scripts in roblox executor",
            "how to make a script hub gui",
            "how to create a minimize button for gui",
            
            # Lua Programming
            "what is a table in lua",
            "how to use loops in lua",
            "what is metatables in lua",
            "how to create functions in lua",
            "what is coroutine in lua",
            "how to handle errors in lua",
            "what is pairs vs ipairs in lua",
            "how to concatenate strings in lua",
            
            # General Programming
            "what is object oriented programming",
            "how to debug code effectively",
            "what are best practices for naming variables",
            "how to optimize code performance",
            "what is the difference between local and global variables",
            "how to comment code properly",
            "what are design patterns",
            
            # Communication Skills
            "how to ask for help with coding problems",
            "how to explain technical concepts clearly",
            "how to write good documentation",
            "how to give constructive code feedback",
            "how to collaborate on coding projects"
        ]
        
        self.training_index = 0
        
        self.load_data()
        print(f"AI loaded with {len(self.training_data)} learned conversations")
        print("Groq AI autopilot ready - AFK mode available")
    
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
    
    def set_mode(self, groq_enabled):
        """Switch between Test Mode (False) and Learning Mode (True)"""
        self.groq_mode = groq_enabled
        mode_name = "Learning Mode" if groq_enabled else "Test Mode"
        print(f"Switched to: {mode_name}")
        return mode_name
    
    def set_afk_mode(self, enabled):
        """Enable/Disable AFK Auto-Training Mode"""
        self.afk_mode = enabled
        
        if enabled and not self.training_active:
            print("AFK Mode ENABLED - Starting auto-training...")
            self.training_active = True
            self.afk_thread = threading.Thread(target=self._afk_training_loop, daemon=True)
            self.afk_thread.start()
        elif not enabled:
            print("AFK Mode DISABLED - Stopping auto-training...")
            self.training_active = False
        
        return enabled
    
    def _afk_training_loop(self):
        """Background thread for AFK auto-training"""
        print("AFK Training Loop Started")
        
        while self.training_active and self.afk_mode:
            try:
                # Get next training topic
                if self.training_index >= len(self.training_topics):
                    self.training_index = 0  # Loop back
                
                topic = self.training_topics[self.training_index]
                
                # Check if already learned
                already_learned = False
                for item in self.training_data:
                    if item['question'] == topic.lower():
                        already_learned = True
                        break
                
                if not already_learned:
                    print(f"AFK Training: Learning about '{topic}'...")
                    response = self.get_groq_response(topic)
                    print(f"AFK: Successfully learned topic {self.training_index + 1}/{len(self.training_topics)}")
                else:
                    print(f"AFK: Skipping '{topic}' (already learned)")
                
                self.training_index += 1
                
                # Wait 3 seconds before next training - FAST MODE
                time.sleep(3)
                
            except Exception as e:
                print(f"AFK Training Error: {e}")
                time.sleep(10)  # Wait 10 seconds on error, then retry
        
        print("AFK Training Loop Ended")
    
    def add_conversation(self, question, answer):
        """Add new Q&A pair and retrain"""
        q = question.lower().strip()
        a = answer.strip()
        
        # Check for duplicates
        for item in self.training_data:
            if item['question'] == q:
                return True  # Already exists
        
        # Add to training data
        self.training_data.append({
            'question': q,
            'answer': a
        })
        
        # Save and update
        self.save_data()
        self.update_vectors()
        
        print(f"Auto-learned: '{q}'")
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
- Teach All Programmers Language if she wants 


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
            
            # AUTO-SAVE: Groq teaches the AI automatically
            self.add_conversation(question.lower().strip(), response)
            
            return response
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return f"Sorry, I couldn't connect to my teacher (Groq AI). Error: {str(e)}"
    
    def get_response(self, question):
        """Get AI response - respects current mode (Test/Learning)"""
        q = question.lower().strip()
        
        # 1. Check exact match in learned data
        for item in self.training_data:
            if item['question'] == q:
                print(f"Found exact match in memory: '{q}'")
                return {
                    'answer': item['answer'],
                    'source': 'memory',
                    'found': True
                }
        
        # 2. Check for similar questions (fuzzy match)
        if self.vectors is not None and len(self.training_data) > 0:
            try:
                q_vector = self.vectorizer.transform([q])
                similarities = cosine_similarity(q_vector, self.vectors)[0]
                
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                # If very similar (>75% match), use learned answer
                if best_score > 0.75:
                    print(f"Found similar in memory (score: {best_score:.2f})")
                    return {
                        'answer': self.training_data[best_idx]['answer'],
                        'source': 'memory',
                        'found': True,
                        'similarity': best_score
                    }
            except Exception as e:
                print(f"Error in similarity matching: {e}")
        
        # 3. Not found in memory
        if self.groq_mode:
            # LEARNING MODE: Ask Groq and learn
            print(f"Learning Mode: Asking Groq AI about '{q}'")
            answer = self.get_groq_response(question)
            return {
                'answer': answer,
                'source': 'groq',
                'found': False
            }
        else:
            # TEST MODE: Don't ask Groq, just say I don't know
            print(f"Test Mode: No match found for '{q}'")
            return {
                'answer': "I don't know this yet. (Test Mode - Groq is OFF)",
                'source': 'none',
                'found': False
            }
    
    def get_stats(self):
        """Get AI statistics"""
        return {
            'knowledge_count': len(self.training_data),
            'ready': True,
            'groq_enabled': self.groq_mode,
            'afk_enabled': self.afk_mode,
            'afk_progress': f"{self.training_index}/{len(self.training_topics)}",
            'mode': 'Learning Mode' if self.groq_mode else 'Test Mode'
        }

# Create global AI instance
ai = TrainingAI()
