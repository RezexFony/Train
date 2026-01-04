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
    
    # Get response based on current mode
    result = ai.get_response(question)
    stats = ai.get_stats()
    
    return jsonify({
        'response': result['answer'],
        'source': result['source'],
        'found_in_memory': result['found'],
        'knowledge_count': stats['knowledge_count'],
        'current_mode': stats['mode'],
        'afk_progress': stats['afk_progress']
    })

@app.route('/switch-mode', methods=['POST'])
def switch_mode():
    """Switch between Test Mode and Learning Mode"""
    data = request.json
    groq_enabled = data.get('groq_enabled', True)
    
    mode_name = ai.set_mode(groq_enabled)
    stats = ai.get_stats()
    
    return jsonify({
        'success': True,
        'mode': mode_name,
        'groq_enabled': groq_enabled,
        'message': f'Switched to {mode_name}'
    })

@app.route('/toggle-afk', methods=['POST'])
def toggle_afk():
    """Toggle AFK Auto-Training Mode"""
    data = request.json
    afk_enabled = data.get('afk_enabled', False)
    
    ai.set_afk_mode(afk_enabled)
    stats = ai.get_stats()
    
    status_msg = "AFK Mode ENABLED - AI will auto-train every 30 seconds" if afk_enabled else "AFK Mode DISABLED"
    
    return jsonify({
        'success': True,
        'afk_enabled': afk_enabled,
        'afk_progress': stats['afk_progress'],
        'message': status_msg,
        'knowledge_count': stats['knowledge_count']
    })

@app.route('/teach', methods=['POST'])
def teach():
    """Handle manual teaching requests"""
    data = request.json
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    
    if not question or not answer:
        return jsonify({
            'error': 'Both question and answer are required'
        }), 400
    
    # Manually teach the AI
    success = ai.add_conversation(question, answer)
    stats = ai.get_stats()
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Manual override saved! I learned from you directly!',
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
    print("Starting AI Training Server with AFK Mode...")
    print(f"AI knows {len(ai.training_data)} things")
    print(f"Current Mode: Learning Mode (Groq ON)")
    print(f"Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
