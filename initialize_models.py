# initialize_models.py
from engine.train import RecipeRecommender

if __name__ == "__main__":
    print("Initializing models and data...")
    recommender = RecipeRecommender()
    print("Success! Models and data ready.")