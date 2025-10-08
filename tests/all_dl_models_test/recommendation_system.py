# recommendation_system.py
import numpy as np
from lowmind import Tensor, Linear, SGD

class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering for recommendations"""
    def __init__(self, num_users, num_items, embedding_dim=8, hidden_dims=[16, 8], device='cpu'):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        
        # Embeddings
        self.user_embeddings = Tensor(
            np.random.randn(num_users, embedding_dim) * 0.01,
            requires_grad=True, device=device
        )
        self.item_embeddings = Tensor(
            np.random.randn(num_items, embedding_dim) * 0.01,
            requires_grad=True, device=device
        )
        
        # MLP layers
        self.layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layer = Linear(input_dim, hidden_dim, device=device)
            self.layers.append(layer)
            input_dim = hidden_dim
        
        self.output_layer = Linear(input_dim, 1, device=device)
    
    def parameters(self):
        """Get all parameters"""
        params = [self.user_embeddings, self.item_embeddings]
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.output_layer.parameters())
        return params
    
    def forward(self, user_ids, item_ids):
        """Forward pass"""
        user_emb = self.user_embeddings.data[user_ids]
        item_emb = self.item_embeddings.data[item_ids]
        
        user_emb = Tensor(user_emb, device=self.device)
        item_emb = Tensor(item_emb, device=self.device)
        
        x = Tensor(np.concatenate([user_emb.data, item_emb.data], axis=1), device=self.device)
        
        for layer in self.layers:
            x = layer(x).relu()
        
        output = self.output_layer(x).sigmoid()
        return output

def generate_movie_data():
    """Generate synthetic movie rating data"""
    num_users = 50
    num_movies = 100
    num_ratings = 500
    
    # User preferences (genre preferences)
    user_preferences = np.random.randn(num_users, 3)
    
    # Movie features (genre features)
    movie_features = np.random.randn(num_movies, 3)
    
    # Generate ratings based on user-movie compatibility
    ratings = []
    for _ in range(num_ratings):
        user_id = np.random.randint(0, num_users)
        movie_id = np.random.randint(0, num_movies)
        
        # Calculate compatibility score
        compatibility = np.dot(user_preferences[user_id], movie_features[movie_id])
        probability = 1 / (1 + np.exp(-compatibility))
        
        # Generate rating (1 if compatible, 0 otherwise)
        rating = 1 if np.random.random() < probability else 0
        ratings.append((user_id, movie_id, rating))
    
    return ratings, num_users, num_movies

def train_recommendation_model():
    """Train movie recommendation system"""
    print("🎬 Training Movie Recommendation System")
    print("=" * 50)
    
    # Generate data
    ratings, num_users, num_movies = generate_movie_data()
    print(f"Generated {len(ratings)} ratings from {num_users} users for {num_movies} movies")
    
    # Convert to arrays
    user_ids = np.array([r[0] for r in ratings])
    movie_ids = np.array([r[1] for r in ratings])
    ratings_array = np.array([r[2] for r in ratings])
    
    # Model
    model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_movies)
    
    # Training
    optimizer = SGD(model.parameters(), lr=0.01)
    epochs = 150
    batch_size = 32
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(ratings))
        
        for i in range(0, len(ratings), batch_size):
            batch_indices = indices[i:min(i+batch_size, len(ratings))]
            
            batch_users = user_ids[batch_indices]
            batch_movies = movie_ids[batch_indices]
            batch_ratings = ratings_array[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_users, batch_movies)
            targets = Tensor(batch_ratings.reshape(-1, 1))
            
            # Binary cross entropy loss
            loss = - (targets * predictions.log() + (1 - targets) * (1 - predictions).log()).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data[0]
            num_batches += 1
        
        if epoch % 30 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")
    
    print(f"\n✅ Recommendation System Training Completed!")
    
    # Generate recommendations for sample users
    print("\n🎯 Personalized Recommendations:")
    
    for user_id in [0, 1, 2]:
        print(f"\n📋 Recommendations for User {user_id}:")
        
        # Get predictions for all movies
        user_predictions = []
        for movie_id in range(min(20, num_movies)):  # Test first 20 movies
            pred = model(np.array([user_id]), np.array([movie_id]))
            user_predictions.append((movie_id, pred.data[0, 0]))
        
        # Sort by score and get top 3
        user_predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = user_predictions[:3]
        
        for i, (movie_id, score) in enumerate(top_recommendations):
            print(f"  {i+1}. Movie {movie_id} -> Score: {score:.3f}")
    
    # Find similar users (based on embeddings)
    print("\n👥 User Similarity Analysis:")
    user_embeddings = model.user_embeddings.data
    
    for user_id in [0, 1]:
        similarities = []
        user_emb = user_embeddings[user_id]
        
        for other_user_id in range(1, min(10, num_users)):
            if other_user_id != user_id:
                other_emb = user_embeddings[other_user_id]
                similarity = np.dot(user_emb, other_emb) / (
                    np.linalg.norm(user_emb) * np.linalg.norm(other_emb)
                )
                similarities.append((other_user_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar = similarities[0]
        print(f"User {user_id} is most similar to User {most_similar[0]} (score: {most_similar[1]:.3f})")
    
    return model

if __name__ == "__main__":
    model = train_recommendation_model()