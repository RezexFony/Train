from flask import Flask, render_template, request, jsonify
from ai_brain import ai
import os

app = Flask(__name__)

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/train-page')
def train_page():
    """Bulk training page"""
    return render_template('bulk_train.html')

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
    ai.train_model()
    stats = ai.get_stats()
    
    return jsonify({
        'success': True,
        'message': 'Model retrained successfully!',
        'stats': stats
    })

@app.route('/knowledge', methods=['GET'])
def knowledge():
    """Get all knowledge entries"""
    data = ai.get_all_training_data()
    
    # Convert MongoDB ObjectId to string
    for item in data:
        item['_id'] = str(item['_id'])
        if 'created_at' in item:
            item['created_at'] = str(item['created_at'])
    
    return jsonify({
        'success': True,
        'count': len(data),
        'knowledge': data
    })

@app.route('/delete', methods=['POST'])
def delete():
    """Delete a knowledge entry"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question required'}), 400
    
    success = ai.delete_knowledge(question)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Knowledge deleted'
        })
    else:
        return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting AI Training Server with MongoDB...")
    print(f"AI knows {ai.get_knowledge_count()} things")
    print(f"Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
