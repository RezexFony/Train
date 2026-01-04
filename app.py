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
    
    # Get response
    result = ai.get_response(question)
    stats = ai.get_stats()
    
    return jsonify({
        'response': result['answer'],
        'source': result.get('source', 'unknown'),
        'found_in_memory': result.get('found', False),
        'knowledge_count': stats['training_examples'],
        'current_mode': 'Learning Mode',
        'confidence': result.get('confidence', 0.0)
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
    lang = ai.detect_language(question)
    success = ai.add_training_data(question, answer, 'user_taught', lang)
    
    # Retrain if enough data
    if len(ai.training_data) >= 10:
        ai.train_model()
    
    stats = ai.get_stats()
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Manual override saved! I learned from you directly!',
            'knowledge_count': stats['training_examples']
        })
    else:
        return jsonify({
            'error': 'Failed to learn (duplicate question)'
        }), 400

@app.route('/stats', methods=['GET'])
def stats():
    """Get AI statistics"""
    return jsonify(ai.get_stats())

@app.route('/train', methods=['POST'])
def train():
    """Manually trigger model training"""
    result = ai.train_model()
    stats = ai.get_stats()
    
    return jsonify({
        'success': True,
        'message': result,
        'stats': stats
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting AI Training Server...")
    print(f"AI knows {len(ai.training_data)} things")
    print(f"Running on port {port}")
    # FIXED: Set debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)
