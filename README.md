🤖 Rock Paper Scissors AI
=========================

An intelligent Rock Paper Scissors game that learns your playing patterns using LSTM neural networks and tries to predict your next move!

🎯 Wanna try?
-------

**🚀 [Try the Live Demo Here](https://huggingface.co/spaces/0xUjwal/rock-paper-scissors-ai)**

✨ Features
----------

### 🧠 **AI Intelligence**

-   **LSTM Neural Network**: Uses Long Short-Term Memory networks to learn patterns
-   **Pattern Recognition**: Analyzes your move history to predict future moves
-   **Adaptive Learning**: Gets smarter with every game you play
-   **Model Persistence**: Saves and loads trained models for continuous learning

### 🎮 **Game Features**

-   **Real-time Gameplay**: Instant results and feedback
-   **Comprehensive Statistics**: Track wins, losses, ties, and win rates
-   **Smart AI Strategy**: AI tries to counter-predict your moves
-   **Reset Functionality**: Clear stats and retrain the AI anytime
-   **Move History**: Keeps track of all your moves for pattern analysis

### 🖥️ **User Interface**

-   **Beautiful Gradio Interface**: Modern, responsive web interface
-   **Interactive Controls**: Easy-to-use radio buttons and buttons
-   **Real-time Updates**: Live statistics and game results
-   **Mobile Friendly**: Works perfectly on desktop and mobile devices
-   **Dark/Light Theme**: Supports different visual themes

### 🔬 **Technical Features**

-   **Error Handling**: Robust error handling for training and prediction
-   **Memory Optimization**: Efficient model training and inference
-   **Model Checkpointing**: Automatic model saving every 10 games
-   **Logging**: Comprehensive logging for debugging and monitoring

🚀 Quick Start
--------------

### 🌐 Online (Recommended)

Just click here to play instantly: **[🎮 Play Now](https://huggingface.co/spaces/0xUjwal/rock-paper-scissors-ai)**

No installation required! The game runs directly in your browser.

### 💻 Local Installation

#### Prerequisites

-   Python 3.7 or higher
-   pip package manager

#### Step 1: Clone the Repository

```
git clone https://github.com/yourusername/rock-paper-scissors-ai.git
cd rock-paper-scissors-ai

```

#### Step 2: Create Virtual Environment (Recommended)

```
# Create virtual environment
python -m venv rps_env

# Activate virtual environment
# On Windows:
rps_env\Scripts\activate
# On macOS/Linux:
source rps_env/bin/activate

```

#### Step 3: Install Dependencies

```
pip install -r requirements.txt

```

#### Step 4: Run the Application

```
python app.py

```

#### Step 5: Open in Browser

The application will start and display:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live

```

Open the local URL in your browser to play!

📋 Requirements
---------------

### System Requirements

-   **RAM**: 4GB minimum (8GB recommended)
-   **Storage**: ~500MB for dependencies and models
-   **Python**: 3.7, 3.8, 3.9, 3.10, or 3.11
-   **OS**: Windows, macOS, or Linux

### Python Dependencies

```
gradio>=4.0.0
tensorflow>=2.13.0
numpy>=1.24.0
pickle (built-in)
os (built-in)
datetime (built-in)

```

Full list in [`requirements.txt`](https://github.com/0xUjwal/rock-paper-scissors-ai/blob/main/requirements.txt)

🎮 How to Play
--------------

### 🎯 **Objective**

Try to beat the AI by being unpredictable! The AI learns from your moves and tries to predict what you'll play next.

### 📝 **Rules**

-   **Rock** beats **Scissors** ✂️
-   **Paper** beats **Rock** 🪨
-   **Scissors** beats **Paper** 📄
-   Same moves = **Tie** 🤝

### 🎪 **Gameplay Steps**

1.  **Choose Your Move**: Select Rock, Paper, or Scissors
2.  **Click Play**: The AI will make its prediction and move
3.  **See Results**: View the outcome and updated statistics
4.  **Repeat**: Keep playing to see if you can outsmart the AI!
5.  **Reset**: Start fresh anytime with the reset button

### 💡 **Pro Tips**

-   **Be Unpredictable**: Random patterns are harder for AI to learn
-   **Change Strategies**: Switch up your approach mid-game
-   **Watch the Stats**: Monitor how well you're doing against the AI
-   **Play Long Sessions**: The AI gets smarter with more data

🧠 How the AI Works
-------------------

### 🔬 **Technical Overview**

1.  **Data Collection**: Records every move you make
2.  **Sequence Analysis**: Uses last 3 moves to predict the next
3.  **Neural Network**: LSTM processes sequential patterns
4.  **Prediction**: AI predicts your likely next move
5.  **Counter Strategy**: AI chooses the move that beats its prediction
6.  **Continuous Learning**: Model retrains after every game

### 🏗️ **Architecture**

```
Input Layer (3x3)  →  LSTM Layer (50 units)  →  Dense Layer (25 units)  →  Output Layer (3 units)
     ↓                        ↓                         ↓                        ↓
Encoded moves          Pattern learning           Feature extraction        Move probabilities

```

### 📊 **Learning Process**

-   **Minimum Data**: Needs 4+ moves to start learning
-   **Training Frequency**: Retrains after each game
-   **Model Persistence**: Saves model every 10 games
-   **Prediction Accuracy**: Improves with more gameplay data

👨‍💻 Author
------------

**0xUjwal**

-   GitHub: [@0xUjwal](https://github.com/0xUjwal)
-   Hugging Face: [@0xUjwal](https://huggingface.co/0xUjwal)

<div align="center">

**🎮 [Play Now](https://huggingface.co/spaces/0xUjwal/rock-paper-scissors-ai) | ⭐ [Star on GitHub](https://github.com/yourusername/rock-paper-scissors-ai) | 🐛 [Report Issue](https://github.com/0xUjwal/rock-paper-scissors-ai/issues)**

Made with ❤️ using AI and open-source technologies

</div>
