import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import gradio as gr
import pickle
import os
from datetime import datetime

# Initialize moves
moves = ["rock", "paper", "scissors"]
move_to_idx = {"rock": 0, "paper": 1, "scissors": 2}
idx_to_move = {0: "rock", 1: "paper", 2: "scissors"}

# Game state
class GameState:
    def __init__(self):
        self.user_moves = []
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists('rps_model.h5'):
                self.model = tf.keras.models.load_model('rps_model.h5')
                print("Loaded existing model")
            else:
                self.model = self.create_model()
                print("Created new model")
        except:
            self.model = self.create_model()
            print("Created new model due to loading error")
    
    def create_model(self):
        """Create LSTM model"""
        model = Sequential([
            LSTM(50, activation="relu", input_shape=(3, 3), return_sequences=False),
            Dense(25, activation="relu"),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def save_model(self):
        """Save the trained model"""
        try:
            self.model.save('rps_model.h5')
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")

# Initialize game state
game_state = GameState()

def encode_moves(moves_list):
    """Encode moves for AI"""
    return np.array([np.eye(3)[move_to_idx[move]] for move in moves_list])

def predict_next_move(user_history):
    """AI Prediction Logic"""
    if len(user_history) < 3:
        return np.random.choice(moves)  # Random choice for insufficient data
    
    try:
        input_seq = encode_moves(user_history[-3:]).reshape(1, 3, 3)
        prediction = game_state.model.predict(input_seq, verbose=0)
        predicted_user_move = idx_to_move[np.argmax(prediction)]
        
        # AI chooses counter move
        counter_moves = {
            "rock": "paper",
            "paper": "scissors", 
            "scissors": "rock"
        }
        return counter_moves.get(predicted_user_move, np.random.choice(moves))
    except:
        return np.random.choice(moves)

def determine_winner(user_move, ai_move):
    """Game Result Logic"""
    if user_move == ai_move:
        return "tie", "🤝 It's a tie!"
    elif (user_move == "rock" and ai_move == "scissors") or \
         (user_move == "paper" and ai_move == "rock") or \
         (user_move == "scissors" and ai_move == "paper"):
        return "win", "🎉 You win!"
    else:
        return "lose", "🤖 AI wins!"

def train_model():
    """Train the model with current user moves"""
    if len(game_state.user_moves) < 4:
        return
    
    try:
        # Prepare training data
        X_train = []
        y_train = []
        
        for i in range(len(game_state.user_moves) - 3):
            sequence = game_state.user_moves[i:i+3]
            next_move = game_state.user_moves[i+3]
            
            X_train.append(encode_moves(sequence))
            y_train.append(np.eye(3)[move_to_idx[next_move]])
        
        if len(X_train) > 0:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train the model
            game_state.model.fit(X_train, y_train, epochs=1, verbose=0, batch_size=1)
            
            # Save model periodically
            if len(game_state.user_moves) % 10 == 0:
                game_state.save_model()
                
    except Exception as e:
        print(f"Training error: {e}")

def play_game(user_move):
    """Main game function"""
    # AI predicts and chooses move
    ai_move = predict_next_move(game_state.user_moves)
    
    # Add user move to history
    game_state.user_moves.append(user_move)
    
    # Determine winner
    result_type, result_message = determine_winner(user_move, ai_move)
    
    # Update stats
    if result_type == "win":
        game_state.wins += 1
    elif result_type == "lose":
        game_state.losses += 1
    else:
        game_state.ties += 1
    
    # Train the model
    train_model()
    
    # Create response
    move_emojis = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
    total_games = game_state.wins + game_state.losses + game_state.ties
    win_rate = (game_state.wins / total_games * 100) if total_games > 0 else 0
    
    response = f"""
🎮 **Game Result:**
Your move: {move_emojis[user_move]} {user_move.title()}
AI move: {move_emojis[ai_move]} {ai_move.title()}

{result_message}

📊 **Statistics:**
Games played: {total_games}
Wins: {game_state.wins} | Losses: {game_state.losses} | Ties: {game_state.ties}
Win rate: {win_rate:.1f}%

🧠 **AI Learning:** The AI has analyzed {len(game_state.user_moves)} of your moves and is getting smarter!
    """
    
    return response

def reset_game():
    """Reset game statistics"""
    game_state.user_moves = []
    game_state.wins = 0
    game_state.losses = 0
    game_state.ties = 0
    game_state.model = game_state.create_model()  # Reset model too
    
    return "🔄 Game reset! The AI has forgotten your patterns. Start fresh!"

def get_stats():
    """Get current game statistics"""
    total_games = game_state.wins + game_state.losses + game_state.ties
    win_rate = (game_state.wins / total_games * 100) if total_games > 0 else 0
    
    stats = f"""
📊 **Current Statistics:**
Total games: {total_games}
Wins: {game_state.wins}
Losses: {game_state.losses}  
Ties: {game_state.ties}
Win rate: {win_rate:.1f}%

🧠 **AI Analysis:**
Moves analyzed: {len(game_state.user_moves)}
Model trained: {'Yes' if len(game_state.user_moves) >= 4 else 'No (needs 4+ moves)'}
    """
    return stats

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="🤖 Rock Paper Scissors AI") as interface:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🤖 Rock Paper Scissors AI</h1>
        <p style="font-size: 18px; color: #666;">
            Play against an AI that learns your patterns and tries to predict your next move!
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<h3>🎮 Make Your Move</h3>")
            move_input = gr.Radio(
                choices=["rock", "paper", "scissors"], 
                label="Choose your weapon:",
                value="rock"
            )
            
            with gr.Row():
                play_btn = gr.Button("🚀 Play!", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset Game", variant="secondary")
                stats_btn = gr.Button("📊 Show Stats", variant="secondary")
        
        with gr.Column(scale=3):
            gr.HTML("<h3>📋 Game Results</h3>")
            output = gr.Markdown(
                "Welcome! Choose your move and click Play to start the game.",
                elem_classes=["game-output"]
            )
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd;">
        <h4>🧠 How the AI Works:</h4>
        <p>The AI uses an LSTM neural network to analyze your move patterns and predict your next move. 
        The more you play, the better it gets at reading your strategy!</p>
        <p><strong>Tip:</strong> Try to be unpredictable to beat the AI!</p>
    </div>
    """)
    
    # Event handlers
    play_btn.click(fn=play_game, inputs=[move_input], outputs=[output])
    reset_btn.click(fn=reset_game, inputs=[], outputs=[output])
    stats_btn.click(fn=get_stats, inputs=[], outputs=[output])

# Launch options
if __name__ == "__main__":
    # For local development
    interface.launch(
        share=True,  # Creates public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        debug=True
    )
