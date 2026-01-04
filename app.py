from flask import Flask, render_template, request, jsonify
from ai_brain import ai
import os

app = Flask(__name__)

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({
            'error': 'No question provided'
        }), 400
    
    # Try to get response from AI
    response = ai.get_response(question)
    stats = ai.get_stats()
    
    if response:
        return jsonify({
            'response': response,
            'learned': True,
            'knowledge_count': stats['knowledge_count']
        })
    else:
        return jsonify({
            'response': "🤔 I don't know the answer to that yet. Can you teach me?",
            'learned': False,
            'knowledge_count': stats['knowledge_count']
        })

@app.route('/teach', methods=['POST'])
def teach():
    """Handle teaching requests"""
    data = request.json
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    
    if not question or not answer:
        return jsonify({
            'error': 'Both question and answer are required'
        }), 400
    
    # Teach the AI
    success = ai.add_conversation(question, answer)
    stats = ai.get_stats()
    
    if success:
        return jsonify({
            'success': True,
            'message': '✅ Thanks for teaching me! I learned something new! 🧠',
            'knowledge_count': stats['knowledge_count']
        })
    else:
        return jsonify({
            'error': 'Failed to learn'
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get AI statistics"""
    return jsonify(ai.get_stats())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting AI Training Server...")
    print(f"📚 AI knows {len(ai.training_data)} things")
    print(f"🌍 Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
