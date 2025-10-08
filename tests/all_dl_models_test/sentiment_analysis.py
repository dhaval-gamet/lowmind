# sentiment_analysis.py
import numpy as np
import re
from collections import Counter
from lowmind import Tensor, Linear, SGD, cross_entropy_loss

class TextPreprocessor:
    """Simple text preprocessing for sentiment analysis"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        words = []
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            words.extend(tokens)
        
        word_counts = Counter(words)
        common_words = word_counts.most_common(self.vocab_size - 2)
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, count) in enumerate(common_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def text_to_sequence(self, text, max_length=20):
        """Convert text to sequence of indices"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
        if len(sequence) < max_length:
            sequence = sequence + [self.word_to_idx['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        return sequence

class SentimentLSTM:
    """Simple LSTM for sentiment analysis"""
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=16, output_dim=2, device='cpu'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Embedding layer
        self.embedding = Tensor(np.random.randn(vocab_size, embedding_dim) * 0.01, 
                               requires_grad=True, device=device)
        
        # LSTM parameters
        self.W_i = Tensor(np.random.randn(hidden_dim, embedding_dim) * 0.01, requires_grad=True, device=device)
        self.U_i = Tensor(np.random.randn(hidden_dim, hidden_dim) * 0.01, requires_grad=True, device=device)
        self.b_i = Tensor(np.zeros(hidden_dim), requires_grad=True, device=device)
        
        self.W_f = Tensor(np.random.randn(hidden_dim, embedding_dim) * 0.01, requires_grad=True, device=device)
        self.U_f = Tensor(np.random.randn(hidden_dim, hidden_dim) * 0.01, requires_grad=True, device=device)
        self.b_f = Tensor(np.zeros(hidden_dim), requires_grad=True, device=device)
        
        self.W_g = Tensor(np.random.randn(hidden_dim, embedding_dim) * 0.01, requires_grad=True, device=device)
        self.U_g = Tensor(np.random.randn(hidden_dim, hidden_dim) * 0.01, requires_grad=True, device=device)
        self.b_g = Tensor(np.zeros(hidden_dim), requires_grad=True, device=device)
        
        self.W_o = Tensor(np.random.randn(hidden_dim, embedding_dim) * 0.01, requires_grad=True, device=device)
        self.U_o = Tensor(np.random.randn(hidden_dim, hidden_dim) * 0.01, requires_grad=True, device=device)
        self.b_o = Tensor(np.zeros(hidden_dim), requires_grad=True, device=device)
        
        # Output layer
        self.fc = Linear(hidden_dim, output_dim, device=device)
    
    def parameters(self):
        """Get all parameters"""
        params = [
            self.embedding, self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f, self.W_g, self.U_g, self.b_g,
            self.W_o, self.U_o, self.b_o
        ]
        params.extend(self.fc.parameters())
        return params
    
    def lstm_step(self, x, h_prev, c_prev):
        """Single LSTM step"""
        i = (x @ self.W_i.T + h_prev @ self.U_i.T + self.b_i).sigmoid()
        f = (x @ self.W_f.T + h_prev @ self.U_f.T + self.b_f).sigmoid()
        g = (x @ self.W_g.T + h_prev @ self.U_g.T + self.b_g).tanh()
        o = (x @ self.W_o.T + h_prev @ self.U_o.T + self.b_o).sigmoid()
        
        c = f * c_prev + i * g
        h = o * c.tanh()
        
        return h, c
    
    def forward(self, x):
        """Forward pass"""
        batch_size, seq_len = x.shape
        
        h = Tensor(np.zeros((batch_size, self.hidden_dim)), device=self.device)
        c = Tensor(np.zeros((batch_size, self.hidden_dim)), device=self.device)
        
        for t in range(seq_len):
            x_t = self.embedding.data[x[:, t]]
            x_t = Tensor(x_t, device=self.device)
            h, c = self.lstm_step(x_t, h, c)
        
        output = self.fc(h)
        return output

def train_sentiment_model():
    """Train and test sentiment analysis model"""
    print("📝 Training Sentiment Analysis Model")
    print("=" * 50)
    
    # Sample data
    positive_reviews = [
        "this movie is great and wonderful",
        "amazing film with great acting",
        "loved it fantastic story",
        "brilliant performance by actors",
        "excellent movie must watch",
        "wonderful film amazing acting",
        "great movie loved it",
        "fantastic story brilliant"
    ]
    
    negative_reviews = [
        "terrible movie waste of time",
        "boring film bad acting",
        "awful story hated it",
        "poor performance disappointing",
        "bad movie not worth it",
        "terrible acting awful story",
        "boring waste of time",
        "poor film disappointed"
    ]
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    # Preprocess text
    preprocessor = TextPreprocessor(vocab_size=50)
    preprocessor.build_vocab(texts)
    
    # Convert to sequences
    sequences = [preprocessor.text_to_sequence(text) for text in texts]
    X = np.array(sequences, dtype=np.int32)
    y = np.array(labels, dtype=np.int32)
    
    # Model
    vocab_size = len(preprocessor.word_to_idx)
    model = SentimentLSTM(vocab_size=vocab_size)
    
    # Training
    optimizer = SGD(model.parameters(), lr=0.01)
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for i in range(len(X)):
            x_batch = Tensor(X[i:i+1])
            y_batch = Tensor(y[i:i+1])
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = cross_entropy_loss(output, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data[0]
            pred = np.argmax(output.data, axis=1)
            correct += np.sum(pred == y_batch.data)
        
        accuracy = 100. * correct / len(X)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss/len(X):.4f} | Acc: {accuracy:.2f}%")
    
    print(f"\n✅ Training Completed! Final Accuracy: {accuracy:.2f}%")
    
    # Test with new texts
    test_texts = [
        "great movie fantastic acting",
        "terrible awful bad movie",
        "amazing wonderful brilliant",
        "boring waste of time"
    ]
    
    print("\n🧪 Model Predictions:")
    for text in test_texts:
        sequence = preprocessor.text_to_sequence(text)
        x_test = Tensor(np.array([sequence], dtype=np.int32))
        output = model(x_test)
        prediction = np.argmax(output.data, axis=1)[0]
        sentiment = "😊 POSITIVE" if prediction == 1 else "😠 NEGATIVE"
        confidence = np.max(output.data) * 100
        print(f"'{text}' -> {sentiment} ({confidence:.1f}%)")
    
    return model, preprocessor

if __name__ == "__main__":
    model, preprocessor = train_sentiment_model()