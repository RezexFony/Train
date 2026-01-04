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
    """Handle chat requests - AI auto-learns from Groq"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({
            'error': 'No question provided'
        }), 400
    
    # Get response - AI will auto-learn if needed
    response = ai.get_response(question)
    stats = ai.get_stats()
    
    return jsonify({
        'response': response,
        'learned': True,  # Always true since Groq teaches automatically
        'knowledge_count': stats['knowledge_count'],
        'auto_learned': True
    })

@app.route('/teach', methods=['POST'])
def teach():
    """Handle manual teaching requests (optional override)"""
    data = request.json
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    
    if not question or not answer:
        return jsonify({
            'error': 'Both question and answer are required'
        }), 400
    
    # Manually teach the AI (override Groq response)
    success = ai.add_conversation(question, answer)
    stats = ai.get_stats()
    
    if success:
        return jsonify({
            'success': True,
            'message': '✅ Manual override saved! I learned from you directly! 🎓',
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
    print("🚀 Starting AI Training Server with Groq Auto-Learning...")
    print(f"📚 AI knows {len(ai.training_data)} things")
    print(f"🤖 Groq AI will automatically teach new topics!")
    print(f"🌍 Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
