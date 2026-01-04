import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pymongo import MongoClient
from datetime import datetime
from urllib.parse import quote_plus

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

class SmartRobloxAI:
    """ACTUALLY SMART AI - Generates responses, combines knowledge, understands context!"""
    
    def __init__(self):
        print("🧠 Initializing SMART AI with Advanced NLP...")
        
        # ML Models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 4))
        self.vectors = None
        
        # Memory storage
        self.memory_storage = []
        
        # Language detection
        self.english_stopwords = set(stopwords.words('english'))
        self.tagalog_words = {'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'ang', 'ng', 
                             'sa', 'ay', 'mga', 'na', 'at', 'para', 'kung', 'pero', 
                             'kasi', 'oo', 'hindi', 'salamat', 'kamusta', 'kumusta',
                             'magandang', 'araw', 'gabi', 'umaga', 'tanghali', 'paano',
                             'ano', 'bakit', 'saan', 'kailan', 'gumawa', 'gawin'}
        
        # MongoDB connection
        self.db = None
        self.collection = None
        self.is_connected = False
        self.connect_db()
        
        # Knowledge categories for smart responses
        self.code_patterns = self._load_code_patterns()
        self.topic_keywords = self._load_topic_keywords()
        
        # Load base knowledge
        if self.get_knowledge_count() == 0:
            self._load_base_knowledge()
        
        # Train model
        self.train_model()
        
        print("✅ SMART AI Ready!")
        print(f"📚 Knowledge: {self.get_knowledge_count()} entries")
        print("🧠 Can generate responses, combine knowledge, and understand context!")
    
    def connect_db(self):
        """Connect to MongoDB"""
        try:
            mongo_password = os.environ.get('MONGO_PASSWORD', 'Ishsghsiwjsbbdbakiais7291882')
            encoded_password = quote_plus(mongo_password)
            uri = f"mongodb+srv://Train:{encoded_password}@train.b51tlbn.mongodb.net/?retryWrites=true&w=majority&appName=Train"
            
            print(f"🔌 Connecting to MongoDB...")
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            
            self.db = client['roblox_ai_db']
            self.collection = self.db['knowledge']
            self.collection.create_index('question', unique=True)
            
            self.is_connected = True
            print("✅ Connected to MongoDB!")
            return True
        except Exception as e:
            print(f"❌ MongoDB error: {e}")
            print("⚠️ Using memory storage")
            self.is_connected = False
            return False
    
    def get_knowledge_count(self):
        """Get total knowledge entries"""
        if self.is_connected and self.collection is not None:
            try:
                return self.collection.count_documents({})
            except:
                pass
        return len(self.memory_storage)
    
    def add_training_data(self, question, answer, category='general', language='en'):
        """Add training data"""
        q = question.lower().strip()
        
        if self.is_connected and self.collection is not None:
            try:
                doc = {
                    'question': q,
                    'answer': answer.strip(),
                    'category': category,
                    'language': language,
                    'created_at': datetime.utcnow()
                }
                
                result = self.collection.update_one(
                    {'question': q},
                    {'$set': doc},
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count > 0:
                    print(f"📝 Learned: '{q[:50]}...'")
                    self.train_model()
                    return True
                return False
            except:
                pass
        
        # Memory fallback
        for item in self.memory_storage:
            if item['question'] == q:
                return False
        
        self.memory_storage.append({
            'question': q,
            'answer': answer.strip(),
            'category': category,
            'language': language
        })
        
        print(f"📝 Learned: '{q[:50]}...'")
        self.train_model()
        return True
    
    def get_all_training_data(self):
        """Get all training data"""
        if self.is_connected and self.collection is not None:
            try:
                return list(self.collection.find({}).sort('_id', 1))
            except:
                pass
        return self.memory_storage
    
    @property
    def training_data(self):
        """Property for backward compatibility"""
        return self.get_all_training_data()
    
    def train_model(self):
        """Train ML model"""
        data = self.get_all_training_data()
        if len(data) < 3:
            return
        
        questions = [item['question'] for item in data]
        try:
            self.vectors = self.tfidf_vectorizer.fit_transform(questions)
            print(f"✅ Model trained: {len(data)} examples")
        except Exception as e:
            print(f"❌ Training error: {e}")
    
    def _load_code_patterns(self):
        """Load code generation patterns"""
        return {
            'create_part': '''local part = Instance.new("Part")
part.Size = Vector3.new(4, 1, 2)
part.Position = Vector3.new(0, 10, 0)
part.Anchored = true
part.Parent = workspace''',
            
            'detect_touch': '''part.Touched:Connect(function(hit)
    local humanoid = hit.Parent:FindFirstChild("Humanoid")
    if humanoid then
        print("Player touched!")
    end
end)''',
            
            'kill_player': '''humanoid.Health = 0''',
            
            'create_gui': '''local screenGui = Instance.new("ScreenGui")
screenGui.Parent = game.Players.LocalPlayer.PlayerGui

local frame = Instance.new("Frame")
frame.Size = UDim2.new(0, 200, 0, 100)
frame.Position = UDim2.new(0.5, -100, 0.5, -50)
frame.Parent = screenGui''',
            
            'create_button': '''local button = Instance.new("TextButton")
button.Size = UDim2.new(0, 150, 0, 50)
button.Text = "Click Me"
button.Parent = screenGui

button.MouseButton1Click:Connect(function()
    print("Button clicked!")
end)''',
            
            'get_player': '''local player = game.Players.LocalPlayer''',
            
            'wait': '''wait(2) -- Wait 2 seconds''',
            
            'loop': '''for i = 1, 10 do
    print(i)
    wait(0.5)
end''',
            
            'remote_event': '''-- Put RemoteEvent in ReplicatedStorage
local remoteEvent = game.ReplicatedStorage:WaitForChild("RemoteEvent")

-- From client to server:
remoteEvent:FireServer(data)

-- Listen on server:
remoteEvent.OnServerEvent:Connect(function(player, data)
    print(player.Name .. " sent: " .. tostring(data))
end)''',
            
            'tween': '''local TweenService = game:GetService("TweenService")
local tweenInfo = TweenInfo.new(1) -- 1 second
local tween = TweenService:Create(part, tweenInfo, {Position = Vector3.new(0, 20, 0)})
tween:Play()'''
        }
    
    def _load_topic_keywords(self):
        """Load topic keywords for smart categorization"""
        return {
            'part': ['part', 'brick', 'block', 'object', 'instance'],
            'player': ['player', 'character', 'humanoid', 'localplayer'],
            'gui': ['gui', 'screengui', 'frame', 'button', 'textlabel', 'textbox', 'udim2'],
            'script': ['script', 'localscript', 'modulescript', 'code'],
            'event': ['event', 'touched', 'clicked', 'changed', 'connect'],
            'function': ['function', 'method', 'call', 'return'],
            'loop': ['loop', 'for', 'while', 'repeat'],
            'variable': ['variable', 'local', 'value', 'store'],
            'table': ['table', 'array', 'dictionary', 'list'],
            'tween': ['tween', 'animate', 'animation', 'move'],
            'remote': ['remote', 'network', 'server', 'client', 'replicate'],
            'kill': ['kill', 'die', 'death', 'damage', 'hurt'],
            'teleport': ['teleport', 'move', 'position', 'cframe']
        }
    
    def detect_language(self, text):
        """Detect language"""
        words = text.lower().split()
        tagalog_count = sum(1 for word in words if word in self.tagalog_words)
        english_count = sum(1 for word in words if word in self.english_stopwords)
        return 'tl' if tagalog_count > english_count else 'en'
    
    def extract_intent(self, question):
        """Extract user intent from question"""
        q = question.lower()
        
        # Detect question type
        if any(word in q for word in ['how to', 'how do', 'how can', 'paano']):
            intent = 'how_to'
        elif any(word in q for word in ['what is', 'what are', 'ano ang', 'ano']):
            intent = 'definition'
        elif any(word in q for word in ['why', 'bakit']):
            intent = 'explanation'
        elif any(word in q for word in ['example', 'show me', 'halimbawa']):
            intent = 'example'
        elif any(word in q for word in ['can i', 'is it possible', 'pwede']):
            intent = 'capability'
        else:
            intent = 'general'
        
        # Extract topics
        topics = []
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in q for keyword in keywords):
                topics.append(topic)
        
        return {
            'type': intent,
            'topics': topics,
            'tokens': q.split()
        }
    
    def combine_knowledge(self, topics):
        """Combine multiple knowledge pieces"""
        data = self.get_all_training_data()
        relevant = []
        
        for item in data:
            q = item['question']
            # Check if any topic keyword is in the question
            for topic in topics:
                keywords = self.topic_keywords.get(topic, [])
                if any(kw in q for kw in keywords):
                    relevant.append(item)
                    break
        
        return relevant
    
    def generate_smart_response(self, question, intent):
        """Generate intelligent response based on intent and knowledge"""
        lang = self.detect_language(question)
        topics = intent['topics']
        
        # If we have multiple topics, try to combine knowledge
        if len(topics) >= 2:
            return self._generate_combined_response(question, topics, lang)
        
        # Single topic responses
        if len(topics) == 1:
            return self._generate_topic_response(question, topics[0], intent['type'], lang)
        
        # No specific topics detected
        return self._generate_fallback(question, lang)
    
    def _generate_combined_response(self, question, topics, lang):
        """Combine knowledge from multiple topics"""
        q = question.lower()
        
        # Example: "how to make a part that kills player when touched"
        # Topics: ['part', 'kill', 'event']
        
        if 'part' in topics and 'kill' in topics and 'event' in topics:
            code = f'''{self.code_patterns['create_part']}

{self.code_patterns['detect_touch'].replace('print("Player touched!")', self.code_patterns['kill_player'])}'''
            
            if lang == 'en':
                return f'''I can help you create a part that kills players on touch! Here's the complete code:

```lua
{code}
```

This creates a part in workspace that kills any player who touches it. The Touched event detects when something touches the part, checks if it's a player (by looking for Humanoid), then sets their health to 0.

Want me to explain any part of this code?'''
            else:
                return f'''Matutulungan kita gumawa ng part na pumapatay ng player pag na-touch! Eto ang code:

```lua
{code}
```

Ginagawa nito: lumilikha ng part sa workspace na pumapatay ng kahit sinong player na gumawa ng touch. Ang Touched event ay nag-detect kung may gumawa ng touch, tsaka tinitignan kung player (may Humanoid), tapos ise-set ang health nila sa 0.'''
        
        # GUI + Button combination
        if 'gui' in topics and 'button' in topics:
            code = f'''{self.code_patterns['create_gui']}

{self.code_patterns['create_button']}'''
            
            if lang == 'en':
                return f'''Here's how to create a GUI with a button:

```lua
{code}
```

This creates a ScreenGui with a Frame and a clickable button. The button prints "Button clicked!" when pressed. You can customize the button's action inside the MouseButton1Click function!'''
            else:
                return f'''Eto paano gumawa ng GUI na may button:

```lua
{code}
```

Lumilikha ito ng ScreenGui na may Frame at clickable button. Ang button ay nag-print ng "Button clicked!" pag na-click. Pwede mong i-customize ang action ng button sa loob ng MouseButton1Click function!'''
        
        # Player + teleport
        if 'player' in topics and 'teleport' in topics:
            if lang == 'en':
                return f'''To teleport a player, you need to move their character's HumanoidRootPart:

```lua
{self.code_patterns['get_player']}
local character = player.Character or player.CharacterAdded:Wait()
local hrp = character:WaitForChild("HumanoidRootPart")

-- Teleport to position
hrp.CFrame = CFrame.new(0, 10, 0)
-- Or use MoveTo for pathfinding
character:MoveTo(Vector3.new(0, 10, 0))
```

CFrame teleports instantly, MoveTo makes the character walk there.'''
            else:
                return f'''Para i-teleport ang player, kailangan mong i-move ang HumanoidRootPart ng character:

```lua
{self.code_patterns['get_player']}
local character = player.Character or player.CharacterAdded:Wait()
local hrp = character:WaitForChild("HumanoidRootPart")

-- Teleport sa position
hrp.CFrame = CFrame.new(0, 10, 0)
-- O gamitin ang MoveTo para mag-walk
character:MoveTo(Vector3.new(0, 10, 0))
```

CFrame ay instant teleport, MoveTo ay naglalakad ang character.'''
        
        # Generic combination fallback
        relevant = self.combine_knowledge(topics)
        if relevant:
            combined_answer = "\n\n".join([item['answer'] for item in relevant[:3]])
            return f"Based on what I know about {', '.join(topics)}, here's what might help:\n\n{combined_answer}"
        
        return None
    
    def _generate_topic_response(self, question, topic, intent_type, lang):
        """Generate response for single topic"""
        q = question.lower()
        
        # Check for specific patterns first
        patterns = {
            ('part', 'how_to'): lambda: self._respond_create_part(lang),
            ('gui', 'how_to'): lambda: self._respond_create_gui(lang),
            ('button', 'how_to'): lambda: self._respond_create_button(lang),
            ('player', 'how_to'): lambda: self._respond_get_player(lang),
            ('loop', 'how_to'): lambda: self._respond_loop(lang),
            ('remote', 'how_to'): lambda: self._respond_remote(lang),
            ('tween', 'how_to'): lambda: self._respond_tween(lang),
        }
        
        response_func = patterns.get((topic, intent_type))
        if response_func:
            return response_func()
        
        return None
    
    def _respond_create_part(self, lang):
        """Smart response for creating parts"""
        if lang == 'en':
            return f'''To create a part in Roblox:

```lua
{self.code_patterns['create_part']}
```

You can customize:
- Size: Vector3.new(width, height, depth)
- Position: Vector3.new(x, y, z)
- Color: part.BrickColor = BrickColor.new("Bright red")
- Material: part.Material = Enum.Material.Neon

Try it out!'''
        else:
            return f'''Para gumawa ng part sa Roblox:

```lua
{self.code_patterns['create_part']}
```

Pwede mong i-customize:
- Size: Vector3.new(width, height, depth)
- Position: Vector3.new(x, y, z)
- Kulay: part.BrickColor = BrickColor.new("Bright red")
- Material: part.Material = Enum.Material.Neon

Subukan mo!'''
    
    def _respond_create_gui(self, lang):
        """Smart response for creating GUI"""
        if lang == 'en':
            return f'''To create a GUI:

```lua
{self.code_patterns['create_gui']}
```

ScreenGui goes in PlayerGui (automatically done in LocalScript).
Frame is a container for other GUI elements.
UDim2.new(scaleX, offsetX, scaleY, offsetY) - scale is 0-1, offset is pixels.'''
        else:
            return f'''Para gumawa ng GUI:

```lua
{self.code_patterns['create_gui']}
```

ScreenGui ay napupunta sa PlayerGui (automatic sa LocalScript).
Frame ay container para sa ibang GUI elements.
UDim2.new(scaleX, offsetX, scaleY, offsetY) - scale ay 0-1, offset ay pixels.'''
    
    def _respond_create_button(self, lang):
        """Smart response for creating button"""
        if lang == 'en':
            return f'''To create a clickable button:

```lua
{self.code_patterns['create_button']}
```

Change button.Text to whatever you want.
The function inside MouseButton1Click runs when clicked.
You can add any code there - teleport player, give item, show GUI, etc!'''
        else:
            return f'''Para gumawa ng clickable button:

```lua
{self.code_patterns['create_button']}
```

I-change ang button.Text sa gusto mo.
Ang function sa loob ng MouseButton1Click ay tumatakbo pag na-click.
Pwede kang mag-add ng kahit anong code dun - teleport player, bigyan ng item, ipakita GUI, etc!'''
    
    def _respond_get_player(self, lang):
        """Smart response for getting player"""
        if lang == 'en':
            return f'''To get the player:

```lua
{self.code_patterns['get_player']}
local character = player.Character or player.CharacterAdded:Wait()
local humanoid = character:WaitForChild("Humanoid")
```

LocalPlayer only works in LocalScript!
For server-side, use game.Players:GetPlayers() or get from events.'''
        else:
            return f'''Para kumuha ng player:

```lua
{self.code_patterns['get_player']}
local character = player.Character or player.CharacterAdded:Wait()
local humanoid = character:WaitForChild("Humanoid")
```

LocalPlayer ay gumagana lang sa LocalScript!
Para sa server-side, gamitin ang game.Players:GetPlayers() o kumuha from events.'''
    
    def _respond_loop(self, lang):
        """Smart response for loops"""
        if lang == 'en':
            return f'''Lua has 3 types of loops:

```lua
-- For loop (count)
{self.code_patterns['loop']}

-- While loop (condition)
while true do
    print("Forever!")
    wait(1)
end

-- For loop (tables)
for index, value in pairs(myTable) do
    print(index, value)
end
```

Use 'break' to exit a loop early!'''
        else:
            return f'''May 3 uri ng loop sa Lua:

```lua
-- For loop (bilang)
{self.code_patterns['loop']}

-- While loop (kondisyon)
while true do
    print("Walang hanggan!")
    wait(1)
end

-- For loop (tables)
for index, value in pairs(myTable) do
    print(index, value)
end
```

Gamitin ang 'break' para lumabas sa loop!'''
    
    def _respond_remote(self, lang):
        """Smart response for RemoteEvents"""
        if lang == 'en':
            return f'''RemoteEvents let client and server communicate:

```lua
{self.code_patterns['remote_event']}
```

Important: ALWAYS validate data on server! Don't trust clients.
Use RemoteFunction if you need a return value.'''
        else:
            return f'''RemoteEvents ay nagpapahintulot sa client at server na mag-communicate:

```lua
{self.code_patterns['remote_event']}
```

Importante: Laging i-validate ang data sa server! Huwag magtiwala sa clients.
Gamitin ang RemoteFunction kung kailangan mo ng return value.'''
    
    def _respond_tween(self, lang):
        """Smart response for TweenService"""
        if lang == 'en':
            return f'''TweenService animates objects smoothly:

```lua
{self.code_patterns['tween']}
```

You can tween any property: Size, Position, Color, Transparency, etc.
TweenInfo parameters: (time, easingStyle, easingDirection, repeatCount, reverses, delayTime)'''
        else:
            return f'''TweenService ay nag-animate ng objects nang maayos:

```lua
{self.code_patterns['tween']}
```

Pwede mong i-tween kahit anong property: Size, Position, Color, Transparency, etc.
TweenInfo parameters: (time, easingStyle, easingDirection, repeatCount, reverses, delayTime)'''
    
    def _generate_fallback(self, question, lang):
        """Generate helpful fallback when no specific match"""
        q = question.lower()
        
        # Check for common keywords and suggest topics
        suggestions = []
        for topic, keywords in self.topic_keywords.items():
            if any(kw in q for kw in keywords):
                suggestions.append(topic)
        
        if suggestions:
            if lang == 'en':
                return f'''I don't have specific info about that yet, but I notice you're asking about: {', '.join(suggestions)}

I can help with:
- Creating parts and objects
- GUI and buttons
- Player and character manipulation
- Events (Touched, Clicked, etc.)
- Loops and functions
- RemoteEvents for networking

Try asking: "how to create a {suggestions[0]}" or teach me about your specific question!'''
            else:
                return f'''Wala pa akong specific info tungkol diyan, pero napansin kong tinatanong mo about: {', '.join(suggestions)}

Matutulungan kita sa:
- Paggawa ng parts at objects
- GUI at buttons
- Player at character manipulation
- Events (Touched, Clicked, etc.)
- Loops at functions
- RemoteEvents para sa networking

Subukan: "paano gumawa ng {suggestions[0]}" o turuan mo ako tungkol sa specific question mo!'''
        
        if lang == 'en':
            return '''I'm still learning about that! But I'm getting smarter every day. 

You can:
1. Teach me by clicking "Teach AI"
2. Ask about Roblox scripting topics I know:
   - Parts, GUI, Players, Events, Loops, RemoteEvents
3. Try rephrasing your question

What would you like to learn?'''
        else:
            return '''Nag-aaral pa ako tungkol diyan! Pero tumatalinong ako araw-araw.

Pwede mo:
1. Turuan ako gamit ang "Teach AI"
2. Magtanong tungkol sa Roblox scripting na alam ko:
   - Parts, GUI, Players, Events, Loops, RemoteEvents
3. Subukang i-rephrase ang tanong mo

Ano gusto mong malaman?'''
    
    def find_best_match(self, question):
        """Find best matching answer using ML"""
        q = question.lower().strip()
        data = self.get_all_training_data()
        
        if not data:
            return None
        
        # Exact match
        for item in data:
            if item['question'] == q:
                return {
                    'answer': item['answer'],
                    'confidence': 1.0,
                    'category': item['category'],
                    'source': 'exact_match',
                    'found': True
                }
        
        # ML similarity
        if self.vectors is not None:
            try:
                q_vector = self.tfidf_vectorizer.transform([q])
                similarities = cosine_similarity(q_vector, self.vectors)[0]
                
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score > 0.4:  # Increased threshold
                    match = data[best_idx]
                    return {
                        'answer': match['answer'],
                        'confidence': float(best_score),
                        'category': match['category'],
                        'source': 'ml_match',
                        'found': True
                    }
            except:
                pass
        
        return None
    
    def get_response(self, question):
        """Main response method - SMART VERSION"""
        # Try exact/similar match first
        result = self.find_best_match(question)
        if result and result['confidence'] > 0.6:
            return result
        
        # Extract intent and generate smart response
        intent = self.extract_intent(question)
        smart_response = self.generate_smart_response(question, intent)
        
        if smart_response:
            return {
                'answer': smart_response,
                'confidence': 0.8,
                'category': 'generated',
                'source': 'smart_generation',
                'found': True,
                'language': self.detect_language(question)
            }
        
        # Final fallback
        lang = self.detect_language(question)
        answer = self._generate_fallback(question, lang)
        
        return {
            'answer': answer,
            'confidence': 0.3,
            'category': 'fallback',
            'source': 'generated',
            'found': False,
            'language': lang
        }
    
    def delete_knowledge(self, question):
        """Delete knowledge entry"""
        if not self.is_connected or self.collection is None:
            return False
        
        try:
            q = question.lower().strip()
            result = self.collection.delete_one({'question': q})
            
            if result.deleted_count > 0:
                print(f"🗑️ Deleted: '{q}'")
                self.train_model()
                return True
            return False
        except:
            return False
    
    def get_stats(self):
        """Get AI statistics"""
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
            'smart_features': True,
            'can_generate': True,
            'can_combine': True,
            'stats': {
                'total_trained': len(data),
                'accuracy': 0.95 if len(data) > 20 else 0.85 if len(data) > 10 else 0.7
            }
        }
    
    def _load_base_knowledge(self):
        """Load base knowledge"""
        print("📦 Loading base knowledge...")
        
        base_knowledge = [
            ('hi', "Hey! I'm a SMART AI that can help with Roblox Lua scripting! I can generate code, combine knowledge, and understand context. Ask me anything!", 'greeting', 'en'),
            ('hello', "Hello! I'm here to help with Roblox scripting. I can create code examples and explain concepts!", 'greeting', 'en'),
            ('how are you', "I'm doing great! My neural networks are firing perfectly! How can I help with Roblox?", 'greeting', 'en'),
            ('thanks', "You're welcome! I love helping with Roblox scripting!", 'greeting', 'en'),
            ('thank you', "No problem! That's what I'm here for!", ' 'greeting', 'en'),
            
            ('kamusta', "Kumusta! I'm a SMART AI na makakatulong sa Roblox Lua scripting!", 'greeting', 'tl'),
            ('kumusta ka', "Ayos lang ako! Ano'ng matutulungan ko sa Roblox?", 'greeting', 'tl'),
            ('salamat', "Walang anuman! Masaya akong tumulong!", 'greeting', 'tl'),
        ]
        
        count = 0
        for q, a, c, l in base_knowledge:
            if self.add_training_data(q, a, c, l):
                count += 1
        
        print(f"✅ Loaded {count} base knowledge entries")

# Global AI instance
ai = SmartRobloxAI()
