import json
import os
import re
import random
import time
import threading
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class RobloxLuaAI:
    """Specialized AI for Roblox Lua, GUI, Executors with Bilingual Support"""
    
    def __init__(self):
        print("🚀 Initializing AI with Machine Learning libraries...")
        
        # ML Models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.classifier = MultinomialNB()
        self.is_trained = False
        
        # Knowledge storage
        self.training_data = []
        self.categories = set()
        self.vectors = None
        
        # Language detection
        self.english_stopwords = set(stopwords.words('english'))
        self.tagalog_words = {'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'ang', 'ng', 
                             'sa', 'ay', 'mga', 'na', 'at', 'para', 'kung', 'pero', 
                             'kasi', 'oo', 'hindi', 'salamat', 'kamusta', 'kumusta',
                             'magandang', 'araw', 'gabi', 'umaga', 'tanghali'}
        
        # Statistics - FIXED: Added all missing stats
        self.stats = {
            'total_trained': 0,
            'accuracy': 0.0,
            'lua_questions': 0,
            'gui_questions': 0,
            'executor_questions': 0,
            'conversation_count': 0,
            'tagalog_count': 0,
            'english_count': 0,
            'training_sessions': 0,
            'lua_knowledge': 0,  # ADDED
            'tagalog_knowledge': 0  # ADDED
        }
        
        # Training mode
        self.learning_mode = True
        self.auto_training = False
        self.training_thread = None
        self.lock = threading.Lock()
        
        # Initialize with pre-loaded knowledge
        self._load_base_knowledge()
        
        # Load saved model if exists
        self.load_model()
        
        print("✅ AI Ready!")
        print(f"📚 Knowledge Base: {len(self.training_data)} entries")
        print(f"🎯 Model Trained: {'Yes' if self.is_trained else 'No'}")
    
    def _load_base_knowledge(self):
        """Pre-load Roblox Lua, GUI, Executor, and Conversation knowledge"""
        
        base_knowledge = [
            # === NATURAL CONVERSATION (ENGLISH) ===
            {
                'question': 'hi',
                'answer': "Hey! What's up? 😊 I can help you with Roblox Lua scripting, GUI development, or just chat!",
                'category': 'greeting',
                'language': 'en'
            },
            {
                'question': 'hello',
                'answer': "Hello there! Ready to learn some Roblox scripting or just want to talk? I'm here for both! 👋",
                'category': 'greeting',
                'language': 'en'
            },
            {
                'question': 'how are you',
                'answer': "I'm doing great, thanks for asking! 😊 How about you? Need help with any Roblox scripts?",
                'category': 'greeting',
                'language': 'en'
            },
            {
                'question': 'thanks',
                'answer': "You're welcome! Happy to help anytime! 😊",
                'category': 'appreciation',
                'language': 'en'
            },
            {
                'question': 'thank you',
                'answer': "No problem at all! Feel free to ask more questions! 🎮",
                'category': 'appreciation',
                'language': 'en'
            },
            
            # === TAGALOG CONVERSATION ===
            {
                'question': 'kamusta',
                'answer': "Kumusta! Mabuti naman ako! 😊 Paano kita matutulungan sa Roblox scripting?",
                'category': 'greeting',
                'language': 'tl'
            },
            {
                'question': 'kumusta ka',
                'answer': "Ayos lang ako, salamat! Ikaw? May tanong ka ba tungkol sa Roblox Lua? 🎮",
                'category': 'greeting',
                'language': 'tl'
            },
            {
                'question': 'salamat',
                'answer': "Walang anuman! Tanong lang kung kailangan mo ng tulong! 😊",
                'category': 'appreciation',
                'language': 'tl'
            },
            {
                'question': 'magandang umaga',
                'answer': "Magandang umaga din! Ano'ng matutulungan ko sa'yo ngayong araw? ☀️",
                'category': 'greeting',
                'language': 'tl'
            },
            
            # === LUA 5.1 BASICS ===
            {
                'question': 'what is lua',
                'answer': "Lua is a lightweight scripting language. Roblox uses Lua 5.1, which is simple and powerful for game scripting. Variables, functions, and tables are the core concepts!",
                'category': 'lua_basics',
                'language': 'en'
            },
            {
                'question': 'how to create variable in lua',
                'answer': "In Lua, just assign a value: local myVar = 10 for numbers, local name = 'John' for strings. Use 'local' to keep variables in scope!",
                'category': 'lua_basics',
                'language': 'en'
            },
            {
                'question': 'lua function example',
                'answer': "Here's a Lua function:\n\nlocal function greet(name)\n    print('Hello ' .. name)\nend\n\ngreet('Player') -- Output: Hello Player",
                'category': 'lua_basics',
                'language': 'en'
            },
            {
                'question': 'what is table in lua',
                'answer': "Tables are Lua's main data structure. They're like arrays and dictionaries combined:\n\nlocal myTable = {1, 2, 3}\nlocal player = {name = 'John', age = 25}\n\nAccess with myTable[1] or player.name",
                'category': 'lua_basics',
                'language': 'en'
            },
            {
                'question': 'lua loop example',
                'answer': "For loops in Lua:\n\nfor i = 1, 10 do\n    print(i)\nend\n\nWhile loops:\n\nwhile condition do\n    -- code\nend\n\nFor tables:\n\nfor key, value in pairs(table) do\n    print(key, value)\nend",
                'category': 'lua_basics',
                'language': 'en'
            },
            
            # === ROBLOX SPECIFIC ===
            {
                'question': 'what is localscript',
                'answer': "LocalScript runs on the client (player's computer). Use it for UI, camera controls, and client-side actions. Put it in StarterPlayer > StarterPlayerScripts or in GUI objects.",
                'category': 'roblox_scripting',
                'language': 'en'
            },
            {
                'question': 'what is script vs localscript',
                'answer': "Script = Server-side (runs on server, controls game logic)\nLocalScript = Client-side (runs on player's PC, handles UI/input)\nModuleScript = Reusable code library\n\nUse Script for game mechanics, LocalScript for player-specific stuff!",
                'category': 'roblox_scripting',
                'language': 'en'
            },
            {
                'question': 'how to make part in roblox',
                'answer': "Create a part in Lua:\n\nlocal part = Instance.new('Part')\npart.Size = Vector3.new(4, 1, 2)\npart.Position = Vector3.new(0, 10, 0)\npart.BrickColor = BrickColor.new('Bright red')\npart.Parent = workspace",
                'category': 'roblox_scripting',
                'language': 'en'
            },
            {
                'question': 'how to detect player click',
                'answer': "Detect mouse clicks in Roblox:\n\nlocal player = game.Players.LocalPlayer\nlocal mouse = player:GetMouse()\n\nmouse.Button1Down:Connect(function()\n    print('Left click!')\nend)\n\nOr for parts:\n\npart.ClickDetector.MouseClick:Connect(function(player)\n    print(player.Name .. ' clicked!')\nend)",
                'category': 'roblox_scripting',
                'language': 'en'
            },
            {
                'question': 'what is remoteevent',
                'answer': "RemoteEvent lets client and server communicate:\n\nServer to Client:\nremoteEvent:FireClient(player, data)\n\nClient to Server:\nremoteEvent:FireServer(data)\n\nListen:\nremoteEvent.OnServerEvent:Connect(function(player, data)\n    -- handle data\nend)\n\nUse for secure communication!",
                'category': 'roblox_scripting',
                'language': 'en'
            },
            
            # === GUI DEVELOPMENT ===
            {
                'question': 'how to create gui in roblox',
                'answer': "Create a ScreenGui in StarterGui:\n\nlocal screenGui = Instance.new('ScreenGui')\nscreenGui.Parent = game.Players.LocalPlayer.PlayerGui\n\nlocal frame = Instance.new('Frame')\nframe.Size = UDim2.new(0, 200, 0, 100)\nframe.Position = UDim2.new(0.5, -100, 0.5, -50)\nframe.BackgroundColor3 = Color3.fromRGB(50, 50, 50)\nframe.Parent = screenGui",
                'category': 'gui',
                'language': 'en'
            },
            {
                'question': 'how to make button in roblox gui',
                'answer': "Create a TextButton:\n\nlocal button = Instance.new('TextButton')\nbutton.Size = UDim2.new(0, 150, 0, 50)\nbutton.Position = UDim2.new(0.5, -75, 0.5, -25)\nbutton.Text = 'Click Me!'\nbutton.Parent = screenGui\n\nbutton.MouseButton1Click:Connect(function()\n    print('Button clicked!')\nend)",
                'category': 'gui',
                'language': 'en'
            },
            {
                'question': 'how to make draggable gui',
                'answer': "Make GUI draggable:\n\nlocal frame = script.Parent\nlocal dragging, dragStart, startPos\n\nframe.InputBegan:Connect(function(input)\n    if input.UserInputType == Enum.UserInputType.MouseButton1 then\n        dragging = true\n        dragStart = input.Position\n        startPos = frame.Position\n    end\nend)\n\ngame:GetService('UserInputService').InputChanged:Connect(function(input)\n    if dragging and input.UserInputType == Enum.UserInputType.MouseMovement then\n        local delta = input.Position - dragStart\n        frame.Position = UDim2.new(\n            startPos.X.Scale, startPos.X.Offset + delta.X,\n            startPos.Y.Scale, startPos.Y.Offset + delta.Y\n        )\n    end\nend)\n\nframe.InputEnded:Connect(function(input)\n    if input.UserInputType == Enum.UserInputType.MouseButton1 then\n        dragging = false\n    end\nend)",
                'category': 'gui',
                'language': 'en'
            },
            {
                'question': 'udim2 explained',
                'answer': "UDim2 is for GUI positioning/sizing:\n\nUDim2.new(scaleX, offsetX, scaleY, offsetY)\n\nScale = 0 to 1 (percentage of screen)\nOffset = pixels\n\nExamples:\nUDim2.new(0.5, 0, 0.5, 0) -- Center (50% of screen)\nUDim2.new(0, 100, 0, 50) -- 100 pixels from left, 50 from top\nUDim2.new(1, 0, 1, 0) -- Full screen",
                'category': 'gui',
                'language': 'en'
            },
            
            # === EXECUTOR KNOWLEDGE ===
            {
                'question': 'what is roblox executor',
                'answer': "A Roblox executor is a tool that runs Lua scripts in Roblox games. Popular ones include Synapse X, Script-Ware, and KRNL. They inject code into the game client. Note: Use responsibly and follow Roblox ToS!",
                'category': 'executor',
                'language': 'en'
            },
            {
                'question': 'how to make executor gui',
                'answer': "Basic executor GUI structure:\n\nlocal ScreenGui = Instance.new('ScreenGui')\nlocal Frame = Instance.new('Frame')\nlocal TextBox = Instance.new('TextBox')  -- Code editor\nlocal ExecuteBtn = Instance.new('TextButton')\nlocal ClearBtn = Instance.new('TextButton')\n\nExecuteBtn.MouseButton1Click:Connect(function()\n    local code = TextBox.Text\n    loadstring(code)()  -- Execute the code\nend)\n\nClearBtn.MouseButton1Click:Connect(function()\n    TextBox.Text = ''\nend)",
                'category': 'executor',
                'language': 'en'
            },
            {
                'question': 'loadstring in lua',
                'answer': "loadstring() compiles Lua code from a string:\n\nlocal code = 'print(\"Hello\")'\nlocal func = loadstring(code)\nfunc()  -- Executes the code\n\nOr shorter:\nloadstring('print(\"Hello\")')()  \n\nUseful for executors to run dynamic code!",
                'category': 'executor',
                'language': 'en'
            },
            {
                'question': 'getgenv explained',
                'answer': "getgenv() returns the global environment table for executors. It persists across script executions:\n\ngetgenv().myVariable = 'persistent'\n\nNow 'myVariable' stays even after scripts end. Used in executor scripts to share data between different script runs.",
                'category': 'executor',
                'language': 'en'
            },
            {
                'question': 'what is script hub',
                'answer': "A script hub is a GUI that contains multiple pre-made scripts/hacks for different games. It usually has:\n\n- Game selector\n- Script list\n- Execute buttons\n- Settings\n- Search function\n\nPopular hubs load scripts from pastebin or GitHub when you select a game.",
                'category': 'executor',
                'language': 'en'
            },
            
            # === TAGALOG PROGRAMMING HELP ===
            {
                'question': 'paano gumawa ng script sa roblox',
                'answer': "Para gumawa ng script sa Roblox:\n\n1. Buksan ang Roblox Studio\n2. Sa Explorer, i-click ang ServerScriptService\n3. Insert -> Script\n4. I-type ang iyong Lua code\n5. I-save at i-test!\n\nPara sa GUI scripts, gumamit ng LocalScript sa StarterGui.",
                'category': 'roblox_scripting',
                'language': 'tl'
            },
            {
                'question': 'ano ang variable sa lua',
                'answer': "Ang variable ay para mag-store ng data:\n\nlocal pangalan = 'Juan'  -- text\nlocal edad = 25  -- numero\nlocal mataas = true  -- boolean\n\nGamitin ang 'local' para mas mabilis at organized ang code!",
                'category': 'lua_basics',
                'language': 'tl'
            },
        ]
        
        for item in base_knowledge:
            self.add_training_data(
                item['question'],
                item['answer'],
                item['category'],
                item['language']
            )
        
        print(f"✅ Loaded {len(base_knowledge)} pre-trained knowledge entries")
    
    def detect_language(self, text):
        """Detect if text is English or Tagalog"""
        words = text.lower().split()
        
        tagalog_count = sum(1 for word in words if word in self.tagalog_words)
        english_count = sum(1 for word in words if word in self.english_stopwords)
        
        if tagalog_count > english_count:
            return 'tl'
        return 'en'
    
    def add_training_data(self, question, answer, category='general', language='en'):
        """Add new training example"""
        with self.lock:
            q = question.lower().strip()
            
            for item in self.training_data:
                if item['question'] == q:
                    return False
            
            self.training_data.append({
                'question': q,
                'answer': answer.strip(),
                'category': category,
                'language': language
            })
            
            self.categories.add(category)
            self.stats['total_trained'] += 1
            
            # FIXED: Check if keys exist before incrementing
            if category in ['lua_basics', 'roblox_scripting']:
                if 'lua_knowledge' in self.stats:
                    self.stats['lua_knowledge'] += 1
            elif category == 'gui':
                self.stats['gui_questions'] += 1
            elif category == 'executor':
                self.stats['executor_questions'] += 1
            
            if language == 'tl':
                if 'tagalog_knowledge' in self.stats:
                    self.stats['tagalog_knowledge'] += 1
            
            print(f"📝 Added: '{q[:40]}...' [Category: {category}, Lang: {language}]")
            return True
    
    def train_model(self, test_size=0.2):
        """Train the ML model"""
        if len(self.training_data) < 10:
            return "❌ Need at least 10 training examples!"
        
        print("\n" + "="*60)
        print("🎓 TRAINING ML MODEL")
        print("="*60)
        
        questions = [item['question'] for item in self.training_data]
        categories = [item['category'] for item in self.training_data]
        
        self.vectors = self.tfidf_vectorizer.fit_transform(questions)
        
        if len(set(categories)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                self.vectors, categories, test_size=test_size, random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            accuracy = self.classifier.score(X_test, y_test)
            self.stats['accuracy'] = accuracy
            
            print(f"✅ Model trained!")
            print(f"📊 Accuracy: {accuracy*100:.2f}%")
            print(f"📚 Training examples: {len(self.training_data)}")
            print(f"🎯 Categories: {len(self.categories)}")
        else:
            print("⚠️ Need multiple categories for classification")
            self.stats['accuracy'] = 0.0
        
        self.is_trained = True
        self.stats['training_sessions'] += 1
        
        self.save_model()
        
        print("="*60 + "\n")
        return f"✅ Training complete! Accuracy: {self.stats['accuracy']*100:.2f}%"
    
    def find_best_match(self, question):
        """Find best matching answer using ML"""
        q = question.lower().strip()
        
        with self.lock:
            for item in self.training_data:
                if item['question'] == q:
                    return {
                        'answer': item['answer'],
                        'confidence': 1.0,
                        'category': item['category'],
                        'source': 'exact_match',
                        'found': True
                    }
        
        if self.vectors is not None and len(self.training_data) > 0:
            q_vector = self.tfidf_vectorizer.transform([q])
            similarities = cosine_similarity(q_vector, self.vectors)[0]
            
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > 0.3:
                with self.lock:
                    match = self.training_data[best_idx]
                    return {
                        'answer': match['answer'],
                        'confidence': float(best_score),
                        'category': match['category'],
                        'source': 'ml_match',
                        'found': True
                    }
        
        return None
    
    def generate_response(self, question, lang='en'):
        """Generate fallback response"""
        tokens = question.lower().split()
        
        lua_words = ['lua', 'script', 'function', 'variable', 'table', 'loop']
        gui_words = ['gui', 'frame', 'button', 'udim2', 'screengui', 'textbox']
        executor_words = ['executor', 'loadstring', 'getgenv', 'script hub']
        
        if any(word in tokens for word in lua_words):
            if lang == 'tl':
                return "Interested ka sa Lua scripting? Tanong mo lang kung ano specific gusto mo malaman! May alam ako tungkol sa variables, functions, loops, at tables. 🎮"
            return "Interested in Lua scripting? Ask me anything specific! I know about variables, functions, loops, and tables. 🎮"
        
        if any(word in tokens for word in gui_words):
            if lang == 'tl':
                return "GUI development ba? Marunong ako gumawa ng frames, buttons, draggable windows, at iba pa! Ano specific gusto mo malaman? 🎨"
            return "GUI development? I can help you create frames, buttons, draggable windows, and more! What specifically do you need? 🎨"
        
        if any(word in tokens for word in executor_words):
            if lang == 'tl':
                return "Executor development ba? Alam ko paano gumawa ng script executor GUI, loadstring, at iba pang executor functions. Tanong lang! ⚡"
            return "Executor development? I know how to create script executor GUIs, use loadstring, and other executor functions. Just ask! ⚡"
        
        if lang == 'tl':
            return "Hindi ko pa alam yan, pero nag-aaral ako! Pwede mo akong turuan o magtanong about Roblox Lua, GUI, or executors! 😊"
        return "I'm still learning about that! You can teach me or ask about Roblox Lua, GUI development, or executors! 😊"
    
    def get_response(self, question):
        """Main response method"""
        lang = self.detect_language(question)
        
        self.stats['conversation_count'] += 1
        if lang == 'tl':
            self.stats['tagalog_count'] += 1
        else:
            self.stats['english_count'] += 1
        
        result = self.find_best_match(question)
        
        if result:
            if result['category'] in ['lua_basics', 'roblox_scripting']:
                self.stats['lua_questions'] += 1
            elif result['category'] == 'gui':
                self.stats['gui_questions'] += 1
            elif result['category'] == 'executor':
                self.stats['executor_questions'] += 1
            
            return result
        
        answer = self.generate_response(question, lang)
        
        return {
            'answer': answer,
            'confidence': 0.3,
            'category': 'unknown',
            'source': 'generated',
            'found': False,
            'language': lang
        }
    
    def save_model(self):
        """Save trained model and data"""
        with self.lock:
            try:
                model_data = {
                    'training_data': self.training_data,
                    'categories': list(self.categories),
                    'stats': self.stats,
                    'is_trained': self.is_trained,
                    'tfidf_vectorizer': self.tfidf_vectorizer,
                    'classifier': self.classifier,
                    'vectors': self.vectors
                }
                
                with open('roblox_lua_ai_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                
                print("💾 Model saved successfully!")
            except Exception as e:
                print(f"❌ Save error: {e}")
    
    def load_model(self):
        """Load saved model"""
        try:
            if os.path.exists('roblox_lua_ai_model.pkl'):
                with open('roblox_lua_ai_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.training_data = model_data.get('training_data', [])
                self.categories = set(model_data.get('categories', []))
                self.stats = model_data.get('stats', self.stats)
                self.is_trained = model_data.get('is_trained', False)
                self.tfidf_vectorizer = model_data.get('tfidf_vectorizer', self.tfidf_vectorizer)
                self.classifier = model_data.get('classifier', self.classifier)
                self.vectors = model_data.get('vectors', None)
                
                print("📖 Loaded saved model!")
        except Exception as e:
            print(f"⚠️ Could not load saved model: {e}")
    
    def get_stats(self):
        """Get statistics"""
        return {
            'training_examples': len(self.training_data),
            'categories': len(self.categories),
            'is_trained': self.is_trained,
            'learning_mode': self.learning_mode,
            'auto_training': self.auto_training,
            'stats': self.stats
        }

# Global AI instance
ai = RobloxLuaAI()
