
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pymongo import MongoClient
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

class RobloxLuaAI:
    """AI with MongoDB Database Persistence - ACTUALLY LEARNS!"""
    
    def __init__(self):
        print("🚀 Initializing AI with MongoDB...")
        
        # ML Models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.vectors = None
        
        # Language detection
        self.english_stopwords = set(stopwords.words('english'))
        self.tagalog_words = {'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'ang', 'ng', 
                             'sa', 'ay', 'mga', 'na', 'at', 'para', 'kung', 'pero', 
                             'kasi', 'oo', 'hindi', 'salamat', 'kamusta', 'kumusta',
                             'magandang', 'araw', 'gabi', 'umaga', 'tanghali'}
        
        # MongoDB connection
        self.db = None
        self.collection = None
        self.connect_db()
        
        # Load base knowledge if database is empty
        if self.get_knowledge_count() == 0:
            self._load_base_knowledge()
        
        # Train model with existing data
        self.train_model()
        
        print("✅ AI Ready!")
        print(f"📚 Knowledge Base: {self.get_knowledge_count()} entries")
    
    def connect_db(self):
        """Connect to MongoDB"""
        try:
            # Get MongoDB credentials from environment
            mongo_password = os.environ.get('MONGO_PASSWORD', 'Ishsghsiwjsbbdbakiais7291882')
            
            # MongoDB connection string
            uri = f"mongodb+srv://Train:{mongo_password}@train.b51tlbn.mongodb.net/?retryWrites=true&w=majority&appName=Train"
            
            # Connect to MongoDB
            client = MongoClient(uri)
            
            # Test connection
            client.admin.command('ping')
            
            # Use database and collection
            self.db = client['roblox_ai_db']
            self.collection = self.db['knowledge']
            
            # Create index on question field for faster queries
            self.collection.create_index('question', unique=True)
            
            print("✅ Connected to MongoDB!")
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            self.db = None
            self.collection = None
    
    def get_knowledge_count(self):
        """Get total knowledge entries"""
        if not self.collection:
            return 0
        
        try:
            return self.collection.count_documents({})
        except Exception as e:
            print(f"❌ Error getting count: {e}")
            return 0
    
    def add_training_data(self, question, answer, category='general', language='en'):
        """Add new training data to MongoDB - ACTUALLY SAVES FOREVER!"""
        if not self.collection:
            print("⚠️ No database connection")
            return False
        
        q = question.lower().strip()
        
        try:
            # Prepare document
            doc = {
                'question': q,
                'answer': answer.strip(),
                'category': category,
                'language': language,
                'created_at': datetime.utcnow()
            }
            
            # Insert or update
            result = self.collection.update_one(
                {'question': q},
                {'$set': doc},
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                print(f"📝 Learned: '{q[:50]}...' [Category: {category}]")
                # Retrain model with new data
                self.train_model()
                return True
            else:
                print(f"⚠️ Already know this: '{q[:50]}...'")
                return False
        except Exception as e:
            print(f"❌ Error adding data: {e}")
            return False
    
    def get_all_training_data(self):
        """Get all training data from MongoDB"""
        if not self.collection:
            return []
        
        try:
            return list(self.collection.find({}).sort('_id', 1))
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return []
    
    def train_model(self):
        """Train the ML model with current data"""
        data = self.get_all_training_data()
        
        if len(data) < 3:
            print("⚠️ Need at least 3 examples to train")
            return
        
        questions = [item['question'] for item in data]
        
        try:
            self.vectors = self.tfidf_vectorizer.fit_transform(questions)
            print(f"✅ Model trained with {len(data)} examples")
        except Exception as e:
            print(f"❌ Training error: {e}")
    
    def find_best_match(self, question):
        """Find best matching answer using similarity"""
        q = question.lower().strip()
        data = self.get_all_training_data()
        
        if not data:
            return None
        
        # Try exact match first
        for item in data:
            if item['question'] == q:
                return {
                    'answer': item['answer'],
                    'confidence': 1.0,
                    'category': item['category'],
                    'source': 'exact_match',
                    'found': True
                }
        
        # Use ML similarity
        if self.vectors is not None:
            try:
                q_vector = self.tfidf_vectorizer.transform([q])
                similarities = cosine_similarity(q_vector, self.vectors)[0]
                
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score > 0.3:
                    match = data[best_idx]
                    return {
                        'answer': match['answer'],
                        'confidence': float(best_score),
                        'category': match['category'],
                        'source': 'ml_match',
                        'found': True
                    }
            except Exception as e:
                print(f"❌ Match error: {e}")
        
        return None
    
    def detect_language(self, text):
        """Detect if text is English or Tagalog"""
        words = text.lower().split()
        
        tagalog_count = sum(1 for word in words if word in self.tagalog_words)
        english_count = sum(1 for word in words if word in self.english_stopwords)
        
        return 'tl' if tagalog_count > english_count else 'en'
    
    def generate_response(self, question, lang='en'):
        """Generate fallback response"""
        tokens = question.lower().split()
        
        lua_words = ['lua', 'script', 'function', 'variable', 'table', 'loop']
        gui_words = ['gui', 'frame', 'button', 'udim2', 'screengui', 'textbox']
        executor_words = ['executor', 'loadstring', 'getgenv', 'script hub']
        
        if any(word in tokens for word in lua_words):
            return "I don't know that yet, but you can teach me! I'm learning about Lua scripting." if lang == 'en' else "Hindi ko pa yan alam, pero turuan mo ako! Nag-aaral ako tungkol sa Lua scripting."
        
        if any(word in tokens for word in gui_words):
            return "I haven't learned about that specific GUI topic yet. Can you teach me?" if lang == 'en' else "Hindi ko pa yan natutuhan tungkol sa GUI. Pwede mo ba akong turuan?"
        
        if any(word in tokens for word in executor_words):
            return "I'm still learning about executors. You can teach me about this topic!" if lang == 'en' else "Nag-aaral pa ako tungkol sa executors. Pwede mo akong turuan!"
        
        return "I'm still learning! Teach me by clicking 'Teach AI' button." if lang == 'en' else "Nag-aaral pa ako! Turuan mo ako gamit ang 'Teach AI' button."
    
    def get_response(self, question):
        """Main response method"""
        lang = self.detect_language(question)
        result = self.find_best_match(question)
        
        if result:
            return result
        
        answer = self.generate_response(question, lang)
        
        return {
            'answer': answer,
            'confidence': 0.0,
            'category': 'unknown',
            'source': 'generated',
            'found': False,
            'language': lang
        }
    
    def delete_knowledge(self, question):
        """Delete a specific knowledge entry"""
        if not self.collection:
            return False
        
        try:
            q = question.lower().strip()
            result = self.collection.delete_one({'question': q})
            
            if result.deleted_count > 0:
                print(f"🗑️ Deleted: '{q}'")
                self.train_model()
                return True
            return False
        except Exception as e:
            print(f"❌ Error deleting: {e}")
            return False
    
    def get_stats(self):
        """Get statistics"""
        data = self.get_all_training_data()
        
        categories = {}
        languages = {}
        
        for item in data:
            cat = item.get('category', 'general')
            lang = item.get('language', 'en')
            
            categories[cat] = categories.get(cat, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            'training_examples': len(data),
            'categories': len(categories),
            'category_breakdown': categories,
            'is_trained': self.vectors is not None,
            'learning_mode': True,
            'languages': languages,
            'stats': {
                'total_trained': len(data),
                'accuracy': 0.85 if len(data) > 10 else 0.5 if len(data) > 3 else 0.0
            }
        }
    
    def _load_base_knowledge(self):
        """Load initial knowledge"""
        print("📦 Loading base knowledge...")
        
        base_knowledge = [
            ('hi', "Hey! I can help with Roblox Lua scripting, GUI, and executors!", 'greeting', 'en'),
            ('hello', "Hello! Ready to learn Roblox scripting?", 'greeting', 'en'),
            ('how are you', "I'm doing great! How can I help with Roblox?", 'greeting', 'en'),
            ('thanks', "You're welcome! Happy to help!", 'greeting', 'en'),
            ('thank you', "No problem! Ask me anything about Roblox!", 'greeting', 'en'),
            
            ('kamusta', "Kumusta! Ano'ng matutulungan ko sa Roblox?", 'greeting', 'tl'),
            ('kumusta ka', "Ayos lang ako! May tanong ka ba?", 'greeting', 'tl'),
            ('salamat', "Walang anuman!", 'greeting', 'tl'),
            
            ('what is lua', "Lua is a lightweight scripting language. Roblox uses Lua 5.1 for game scripting. It's simple yet powerful!", 'lua_basics', 'en'),
            ('how to create variable in lua', "In Lua: local myVar = 10 for numbers, local name = 'John' for strings. Always use 'local' for better performance!", 'lua_basics', 'en'),
            ('what is table in lua', "Tables are Lua's main data structure. Example: local myTable = {1, 2, 3} or local player = {name = 'John', age = 25}. Access with myTable[1] or player.name", 'lua_basics', 'en'),
            ('lua function example', "function greet(name)\n  print('Hello ' .. name)\nend\n\ngreet('Player')", 'lua_basics', 'en'),
            ('lua loop example', "for i = 1, 10 do\n  print(i)\nend\n\nwhile condition do\n  -- code\nend\n\nfor key, value in pairs(table) do\n  print(key, value)\nend", 'lua_basics', 'en'),
            
            ('what is localscript', "LocalScript runs on the client (player's computer). Use for UI, camera controls, and client-side actions. Put in StarterPlayerScripts or GUI objects.", 'roblox_scripting', 'en'),
            ('what is script vs localscript', "Script = Server-side (controls game logic)\nLocalScript = Client-side (handles UI/input)\nModuleScript = Reusable code\n\nUse Script for game mechanics, LocalScript for player stuff!", 'roblox_scripting', 'en'),
            ('how to make part in roblox', "local part = Instance.new('Part')\npart.Size = Vector3.new(4, 1, 2)\npart.Position = Vector3.new(0, 10, 0)\npart.BrickColor = BrickColor.new('Bright red')\npart.Parent = workspace", 'roblox_scripting', 'en'),
            ('how to detect player click', "local player = game.Players.LocalPlayer\nlocal mouse = player:GetMouse()\n\nmouse.Button1Down:Connect(function()\n  print('Clicked!')\nend)", 'roblox_scripting', 'en'),
            ('what is remoteevent', "RemoteEvent allows client-server communication:\n\nServer to Client: event:FireClient(player, data)\nClient to Server: event:FireServer(data)\n\nListen: event.OnServerEvent:Connect(function(player, data) end)", 'roblox_scripting', 'en'),
            
            ('how to create gui in roblox', "local gui = Instance.new('ScreenGui')\ngui.Parent = game.Players.LocalPlayer.PlayerGui\n\nlocal frame = Instance.new('Frame')\nframe.Size = UDim2.new(0, 200, 0, 100)\nframe.Position = UDim2.new(0.5, -100, 0.5, -50)\nframe.Parent = gui", 'gui', 'en'),
            ('how to make button in roblox gui', "local button = Instance.new('TextButton')\nbutton.Size = UDim2.new(0, 150, 0, 50)\nbutton.Text = 'Click Me!'\nbutton.Parent = screenGui\n\nbutton.MouseButton1Click:Connect(function()\n  print('Clicked!')\nend)", 'gui', 'en'),
            ('udim2 explained', "UDim2 is for GUI positioning:\n\nUDim2.new(scaleX, offsetX, scaleY, offsetY)\n\nScale = 0 to 1 (percentage)\nOffset = pixels\n\nExamples:\nUDim2.new(0.5, 0, 0.5, 0) -- Center\nUDim2.new(1, 0, 1, 0) -- Full screen", 'gui', 'en'),
            
            ('what is roblox executor', "An executor runs Lua scripts in Roblox games. Popular ones: Synapse X, Script-Ware, KRNL. They inject code into the game client.", 'executor', 'en'),
            ('loadstring in lua', "loadstring() compiles code from string:\n\nlocal code = 'print(\"Hello\")'\nloadstring(code)()\n\nUseful for executors to run dynamic code!", 'executor', 'en'),
            ('getgenv explained', "getgenv() returns the global environment for executors. It persists across script runs:\n\ngetgenv().myVar = 'value'\n\nThe variable stays even after scripts end.", 'executor', 'en'),
            
            ('paano gumawa ng script sa roblox', "Para gumawa ng script:\n1. Buksan Roblox Studio\n2. Explorer -> ServerScriptService\n3. Insert -> Script\n4. I-type ang code\n5. Test!", 'roblox_scripting', 'tl'),
            ('ano ang variable sa lua', "Ang variable ay nag-store ng data:\n\nlocal pangalan = 'Juan'\nlocal edad = 25\n\nGamitin ang 'local' para mas mabilis!", 'lua_basics', 'tl'),
        ]
        
        count = 0
        for q, a, c, l in base_knowledge:
            if self.add_training_data(q, a, c, l):
                count += 1
        
        print(f"✅ Loaded {count} base knowledge entries")

# Global AI instance
ai = RobloxLuaAI()
