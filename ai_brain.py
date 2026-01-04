import json
import os
import threading
import time
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
from openai import OpenAI
import random

class TrainingAI:
    def __init__(self):
        self.training_data = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.groq_mode = True  # True = Learning Mode, False = Test Mode
        self.afk_mode = False  # AFK Auto-Training Mode
        self.afk_thread = None
        self.training_active = False
        
        # NEW: Background teaching threads
        self.background_teaching = False
        self.groq_teacher_thread = None
        self.gemini_teacher_thread = None
        self.openrouter_teacher_thread = None
        
        # Thread locks for data safety
        self.data_lock = threading.Lock()
        
        # Initialize ALL API clients
        self.groq_client = Groq(
            api_key="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"
        )
        
        # Initialize Google Gemini
        genai.configure(api_key="AIzaSyD7BKismdl70NvLlgtHnaMcbaBrpBGPhUU")
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize OpenRouter
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        
        # API stats tracking
        self.api_stats = {
            'groq': {'success': 0, 'failed': 0, 'teaching_now': False},
            'gemini': {'success': 0, 'failed': 0, 'teaching_now': False},
            'openrouter': {'success': 0, 'failed': 0, 'teaching_now': False}
        }
        
        # Expanded training topics - 100+ topics INCLUDING CONVERSATION SKILLS!
        self.training_topics = [
            # CONVERSATION & SOCIAL SKILLS
            "how to greet someone warmly",
            "how to have a casual conversation",
            "how to show empathy in conversation",
            "how to tell a joke naturally",
            "how to respond when someone is sad",
            "how to celebrate someone's success",
            "how to ask follow-up questions in conversation",
            "how to change topics smoothly",
            "how to show interest in what someone is saying",
            "how to be encouraging and supportive",
            "how to give compliments naturally",
            "how to handle awkward silences",
            "how to use humor in conversation",
            "how to be relatable when talking",
            "how to show personality in text",
            "how to use emojis appropriately",
            "how to match someone's energy level",
            "how to be friendly but not annoying",
            "how to remember details about people",
            "how to make someone feel heard",
            "how to apologize sincerely",
            "how to disagree politely",
            "how to give advice without being preachy",
            "how to share personal stories",
            "how to be conversational not robotic",
            "how to use slang naturally",
            "how to talk like a real person not an ai",
            "how to show excitement in text",
            "how to be chill and casual",
            "how to hype someone up",
            
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
            "how to make a shop gui in roblox",
            "what is bindableevent in roblox",
            "how to use datastores in roblox",
            "how to make a loading screen in roblox",
            "how to create animations in roblox",
            
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
            "how to add syntax highlighting to executor",
            "how to create script injection system",
            
            # Lua Programming
            "what is a table in lua",
            "how to use loops in lua",
            "what is metatables in lua",
            "how to create functions in lua",
            "what is coroutine in lua",
            "how to handle errors in lua",
            "what is pairs vs ipairs in lua",
            "how to concatenate strings in lua",
            "what are closures in lua",
            "how to use pcall in lua",
            "what is getfenv and setfenv",
            "how to create classes in lua",
            "what is the difference between . and : in lua",
            
            # Python Programming
            "how to create a list in python",
            "what is a dictionary in python",
            "how to use list comprehensions",
            "what are decorators in python",
            "how to handle exceptions in python",
            "what is object oriented programming in python",
            "how to create a class in python",
            "what are lambda functions",
            "how to use pip to install packages",
            
            # JavaScript
            "what is async await in javascript",
            "how to use promises in javascript",
            "what is the difference between let and var",
            "how to create an arrow function",
            "what is the dom in javascript",
            "how to make an http request in javascript",
            "what is event bubbling",
            
            # Web Development
            "what is html",
            "how to create a form in html",
            "what is css flexbox",
            "how to center a div in css",
            "what is responsive design",
            "how to use media queries",
            "what is the box model in css",
            
            # General Programming
            "what is object oriented programming",
            "how to debug code effectively",
            "what are best practices for naming variables",
            "how to optimize code performance",
            "what is the difference between local and global variables",
            "how to comment code properly",
            "what are design patterns",
            "what is recursion",
            "what is big o notation",
            "what is the difference between stack and heap",
            "what are data structures",
            "what is an algorithm",
            
            # Advanced Topics
            "what is machine learning",
            "what is an api",
            "what is rest api",
            "what is json",
            "what is xml",
            "what is sql",
            "what is nosql",
            "what is git",
            "what is version control",
            
            # Communication Skills
            "how to ask for help with coding problems",
            "how to explain technical concepts clearly",
            "how to write good documentation",
            "how to give constructive code feedback",
            "how to collaborate on coding projects",
            "how to participate in code reviews",
            "how to write clear commit messages"
        ]
        
        # Shuffle topics for variety
        random.shuffle(self.training_topics)
        self.training_index = 0
        
        self.load_data()
        print(f"🧠 AI loaded with {len(self.training_data)} learned conversations")
        print("=" * 60)
        print("🎓 TRIPLE-TEACHER BACKGROUND SYSTEM READY!")
        print("=" * 60)
        print("  🔵 Groq AI (Teacher #1)")
        print("  🟢 Google Gemini (Teacher #2)")
        print("  🟣 OpenRouter (Teacher #3)")
        print("=" * 60)
        print("💡 Your AI will learn from ALL 3 teachers simultaneously!")
        print("🚀 Background teaching will start when you enable Learning Mode")
        print("=" * 60)
        
        # AUTO-START background teaching in learning mode
        if self.groq_mode:
            self.start_background_teaching()
    
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
        """Save training data to JSON file (thread-safe)"""
        with self.data_lock:
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
        
        # Start/stop background teaching based on mode
        if groq_enabled:
            self.start_background_teaching()
        else:
            self.stop_background_teaching()
        
        return mode_name
    
    def start_background_teaching(self):
        """Start all 3 background teacher threads"""
        if not self.background_teaching:
            print("🎓 STARTING BACKGROUND TEACHING - ALL 3 TEACHERS ACTIVATED!")
            self.background_teaching = True
            
            # Start 3 separate teacher threads
            self.groq_teacher_thread = threading.Thread(target=self._groq_teacher_loop, daemon=True)
            self.gemini_teacher_thread = threading.Thread(target=self._gemini_teacher_loop, daemon=True)
            self.openrouter_teacher_thread = threading.Thread(target=self._openrouter_teacher_loop, daemon=True)
            
            self.groq_teacher_thread.start()
            self.gemini_teacher_thread.start()
            self.openrouter_teacher_thread.start()
            
            print("✅ All 3 teachers are now teaching in background!")
    
    def stop_background_teaching(self):
        """Stop all background teaching"""
        if self.background_teaching:
            print("🛑 STOPPING BACKGROUND TEACHING")
            self.background_teaching = False
    
    def _groq_teacher_loop(self):
        """Groq AI teaching loop (runs in background forever)"""
        print("🔵 GROQ TEACHER: Started teaching!")
        
        while self.background_teaching:
            try:
                self.api_stats['groq']['teaching_now'] = True
                
                # Pick a random topic
                topic = random.choice(self.training_topics)
                
                # Check if already learned
                if not self._already_knows(topic):
                    print(f"🔵 GROQ teaching: '{topic}'...")
                    self._teach_with_groq(topic)
                    print(f"🔵 GROQ: ✅ Taught successfully!")
                
                self.api_stats['groq']['teaching_now'] = False
                
                # Wait 10 seconds before next lesson
                time.sleep(10)
                
            except Exception as e:
                print(f"🔵 GROQ error: {e}")
                self.api_stats['groq']['teaching_now'] = False
                time.sleep(30)  # Wait longer on error
        
        print("🔵 GROQ TEACHER: Stopped")
    
    def _gemini_teacher_loop(self):
        """Gemini AI teaching loop (runs in background forever)"""
        print("🟢 GEMINI TEACHER: Started teaching!")
        
        while self.background_teaching:
            try:
                self.api_stats['gemini']['teaching_now'] = True
                
                # Pick a random topic (different from Groq)
                topic = random.choice(self.training_topics)
                
                # Check if already learned
                if not self._already_knows(topic):
                    print(f"🟢 GEMINI teaching: '{topic}'...")
                    self._teach_with_gemini(topic)
                    print(f"🟢 GEMINI: ✅ Taught successfully!")
                
                self.api_stats['gemini']['teaching_now'] = False
                
                # Wait 12 seconds (offset from Groq)
                time.sleep(12)
                
            except Exception as e:
                print(f"🟢 GEMINI error: {e}")
                self.api_stats['gemini']['teaching_now'] = False
                time.sleep(30)
        
        print("🟢 GEMINI TEACHER: Stopped")
    
    def _openrouter_teacher_loop(self):
        """OpenRouter AI teaching loop (runs in background forever)"""
        print("🟣 OPENROUTER TEACHER: Started teaching!")
        
        while self.background_teaching:
            try:
                self.api_stats['openrouter']['teaching_now'] = True
                
                # Pick a random topic
                topic = random.choice(self.training_topics)
                
                # Check if already learned
                if not self._already_knows(topic):
                    print(f"🟣 OPENROUTER teaching: '{topic}'...")
                    self._teach_with_openrouter(topic)
                    print(f"🟣 OPENROUTER: ✅ Taught successfully!")
                
                self.api_stats['openrouter']['teaching_now'] = False
                
                # Wait 15 seconds (offset from others)
                time.sleep(15)
                
            except Exception as e:
                print(f"🟣 OPENROUTER error: {e}")
                self.api_stats['openrouter']['teaching_now'] = False
                time.sleep(30)
        
        print("🟣 OPENROUTER TEACHER: Stopped")
    
    def _already_knows(self, question):
        """Check if AI already knows this topic"""
        q = question.lower().strip()
        with self.data_lock:
            for item in self.training_data:
                if item['question'] == q:
                    return True
        return False
    
    def _teach_with_groq(self, question):
        """Teach using Groq API"""
        try:
            system_prompt = """You are a friendly, conversational AI companion who's both knowledgeable AND personable. 

When teaching programming/technical topics:
- Explain clearly with examples
- Be encouraging and supportive
- Use casual, friendly language

When teaching conversation/social skills:
- Show personality and emotion
- Be natural and relatable
- Use appropriate slang and emojis
- Talk like a real person, not a robot
- Show empathy and understanding
- Be warm and engaging

Always match the vibe - technical when needed, casual and fun when appropriate. Make learning enjoyable!"""
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                model="llama-3.1-8b-instant",  # Using faster model for background
                temperature=0.7,
                max_tokens=800,
            )
            
            response = chat_completion.choices[0].message.content
            self.add_conversation(question, response)
            self.api_stats['groq']['success'] += 1
            
        except Exception as e:
            self.api_stats['groq']['failed'] += 1
            raise e
    
    def _teach_with_gemini(self, question):
        """Teach using Gemini API"""
        try:
            system_prompt = """You are a friendly, conversational AI companion who's both knowledgeable AND personable. 

When teaching programming/technical topics:
- Explain clearly with examples
- Be encouraging and supportive
- Use casual, friendly language

When teaching conversation/social skills:
- Show personality and emotion
- Be natural and relatable
- Use appropriate slang and emojis
- Talk like a real person, not a robot
- Show empathy and understanding
- Be warm and engaging

Always match the vibe - technical when needed, casual and fun when appropriate. Make learning enjoyable!"""
            
            gemini_response = self.gemini_model.generate_content(
                f"{system_prompt}\n\nUser question: {question}"
            )
            
            response = gemini_response.text
            self.add_conversation(question, response)
            self.api_stats['gemini']['success'] += 1
            
        except Exception as e:
            self.api_stats['gemini']['failed'] += 1
            raise e
    
    def _teach_with_openrouter(self, question):
        """Teach using OpenRouter API"""
        try:
            system_prompt = """You are a friendly, conversational AI companion who's both knowledgeable AND personable. 

When teaching programming/technical topics:
- Explain clearly with examples
- Be encouraging and supportive
- Use casual, friendly language

When teaching conversation/social skills:
- Show personality and emotion
- Be natural and relatable
- Use appropriate slang and emojis
- Talk like a real person, not a robot
- Show empathy and understanding
- Be warm and engaging

Always match the vibe - technical when needed, casual and fun when appropriate. Make learning enjoyable!"""
            
            completion = self.openrouter_client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=800,
            )
            
            response = completion.choices[0].message.content
            self.add_conversation(question, response)
            self.api_stats['openrouter']['success'] += 1
            
        except Exception as e:
            self.api_stats['openrouter']['failed'] += 1
            raise e
    
    def set_afk_mode(self, enabled):
        """Enable/Disable AFK Auto-Training Mode (LEGACY - now using background teaching)"""
        self.afk_mode = enabled
        
        if enabled:
            print("AFK Mode is now handled by background teaching!")
            print("All 3 teachers are already teaching 24/7!")
        
        return enabled
    
    def add_conversation(self, question, answer):
        """Add new Q&A pair and retrain (thread-safe)"""
        q = question.lower().strip()
        a = answer.strip()
        
        with self.data_lock:
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
            
            print(f"💾 Auto-learned: '{q}' | Total knowledge: {len(self.training_data)}")
        
        return True
    
    def get_multi_api_response(self, question):
        """MULTI-API FALLBACK: Try all APIs until one works"""
        system_prompt = """You are a friendly, conversational AI companion who's both knowledgeable AND personable.

DUAL EXPERTISE:
1. TECHNICAL KNOWLEDGE:
   - Roblox Lua scripting and game development
   - Roblox executor scripting and GUI development  
   - All programming languages (Python, JavaScript, C++, Java, etc.)
   - Web development, APIs, databases, etc.

2. SOCIAL/CONVERSATION SKILLS:
   - Natural, human-like conversation
   - Showing empathy and emotion
   - Using appropriate humor and slang
   - Being relatable and engaging
   - Matching the user's vibe and energy
   - Remembering context and building rapport

TEACHING STYLE:
- For technical questions: Clear explanations with code examples, encouraging tone, step-by-step guidance
- For casual chat: Be warm, funny, relatable - talk like a real person, not a robot
- Use emojis when appropriate (not overdoing it)
- Show personality and emotion in your responses
- Be supportive and hype people up
- Keep responses conversational and natural
- Match the user's energy - chill when they're chill, excited when they're excited

CONVERSATION SKILLS TO USE:
- Ask follow-up questions to show interest
- Remember what people tell you
- Celebrate their wins
- Be empathetic when they're struggling
- Use casual language and slang naturally
- Tell jokes and be playful when appropriate
- Share enthusiasm about topics
- Be encouraging and supportive
- Show you're listening and care

You're not just a knowledge bot - you're a friend who happens to be really good at coding!"""

        # API 1: Try Groq first
        try:
            print("🔵 Trying Groq AI...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1500,
            )
            
            response = chat_completion.choices[0].message.content
            self.api_stats['groq']['success'] += 1
            print("✅ Groq AI succeeded!")
            
            # AUTO-SAVE
            self.add_conversation(question.lower().strip(), response)
            return response
            
        except Exception as e:
            self.api_stats['groq']['failed'] += 1
            print(f"❌ Groq failed: {e}")
        
        # API 2: Try Gemini as backup
        try:
            print("🟢 Trying Google Gemini...")
            gemini_response = self.gemini_model.generate_content(
                f"{system_prompt}\n\nUser question: {question}"
            )
            
            response = gemini_response.text
            self.api_stats['gemini']['success'] += 1
            print("✅ Gemini succeeded!")
            
            # AUTO-SAVE
            self.add_conversation(question.lower().strip(), response)
            return response
            
        except Exception as e:
            self.api_stats['gemini']['failed'] += 1
            print(f"❌ Gemini failed: {e}")
        
        # API 3: Try OpenRouter as final backup
        try:
            print("🟣 Trying OpenRouter...")
            completion = self.openrouter_client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            
            response = completion.choices[0].message.content
            self.api_stats['openrouter']['success'] += 1
            print("✅ OpenRouter succeeded!")
            
            # AUTO-SAVE
            self.add_conversation(question.lower().strip(), response)
            return response
            
        except Exception as e:
            self.api_stats['openrouter']['failed'] += 1
            print(f"❌ OpenRouter failed: {e}")
        
        # ALL APIs FAILED
        return "🚨 All AI teachers are temporarily unavailable. But don't worry - 3 teachers are still teaching me in the background! I'm getting smarter every second! 💪"
    
    def get_groq_response(self, question):
        """Legacy method - now redirects to multi-API"""
        return self.get_multi_api_response(question)
    
    def get_response(self, question):
        """Get AI response - respects current mode (Test/Learning)"""
        q = question.lower().strip()
        
        # 1. Check exact match in learned data
        with self.data_lock:
            for item in self.training_data:
                if item['question'] == q:
                    print(f"✅ Found exact match in memory: '{q}'")
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
                    print(f"✅ Found similar in memory (score: {best_score:.2f})")
                    with self.data_lock:
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
            # LEARNING MODE: Use multi-API system
            print(f"🎓 Learning Mode: Using multi-API system for '{q}'")
            answer = self.get_multi_api_response(question)
            return {
                'answer': answer,
                'source': 'multi-api',
                'found': False
            }
        else:
            # TEST MODE: Don't ask any API
            print(f"Test Mode: No match found for '{q}'")
            return {
                'answer': "I don't know this yet. (Test Mode - AI teachers are OFF)",
                'source': 'none',
                'found': False
            }
    
    def get_stats(self):
        """Get AI statistics"""
        return {
            'knowledge_count': len(self.training_data),
            'ready': True,
            'groq_enabled': self.groq_mode,
            'background_teaching': self.background_teaching,
            'afk_enabled': self.afk_mode,
            'afk_progress': f"Background teaching active",
            'mode': 'Learning Mode' if self.groq_mode else 'Test Mode',
            'api_stats': self.api_stats,
            'teachers_active': sum(1 for api in self.api_stats.values() if api.get('teaching_now', False))
        }

# Create global AI instance
ai = TrainingAI()
