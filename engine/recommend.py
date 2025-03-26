import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors


# noinspection PyUnresolvedReferences
def _load_recipe_data():
    """Load recipe data with standardized ingredients"""
    return [
        # Mediterranean
        {
            "name": "Hummus with Pita",
            "ingredients": "chickpeas,tahini,lemon_juice,garlic,olive_oil,pita_bread",
            "steps": "Blend chickpeas,Add tahini and lemon juice,Adjust seasoning,Serve with pita bread",
            "cuisine": "Mediterranean",
            "cooking_time": 15,
            "serves": 4,
            "image": "hummus.jpg"
        },
        # Italian
        {
            "name": "Spaghetti Carbonara",
            "ingredients": "spaghetti,eggs,pancetta,parmesan_cheese,black_pepper",
            "steps": "Cook spaghetti,Fry pancetta,Mix eggs and cheese,Combine all ingredients",
            "cuisine": "Italian",
            "cooking_time": 30,
            "serves": 2,
            "image": "carbonara.jpg"
        },
        # Asian
        {
            "name": "Vegetable Stir Fry",
            "ingredients": "rice,broccoli,carrots,bell_pepper,soy_sauce,ginger",
            "steps": "Cook rice,Stir fry vegetables,Add sauce,Combine with rice",
            "cuisine": "Asian",
            "cooking_time": 25,
            "serves": 3,
            "image": "stirfry.jpg"
        },
        # South Indian
        {
            "name": "Masala Dosa",
            "ingredients": "rice_flour,urad_dal,potatoes,onions,green_chillies,mustard_seeds,curry_leaves",
            "steps": "Prepare dosa batter,Spread batter on hot griddle,Add potato filling,Fold and serve with chutney",
            "cuisine": "South Indian",
            "cooking_time": 40,
            "serves": 2,
            "image": "masala_dosa.jpg"
        },
        {
            "name": "Idli Sambar",
            "ingredients": "idli_rice,urad_dal,toor_dal,tamarind,vegetables,sambar_powder",
            "steps": "Steam idli batter,Prepare sambar with vegetables,Serve hot with chutney",
            "cuisine": "South Indian",
            "cooking_time": 35,
            "serves": 3,
            "image": "idli_sambar.jpg"
        },
        {
            "name": "Vada",
            "ingredients": "urad_dal,green_chillies,ginger,curry_leaves,oil",
            "steps": "Grind dal to batter,Shape into donuts,Deep fry until golden,Serve with chutney",
            "cuisine": "South Indian",
            "cooking_time": 30,
            "serves": 4,
            "image": "vada.jpg"
        },
        {
            "name": "Upma",
            "ingredients": "semolina,onions,green_chillies,ginger,curry_leaves,mustard_seeds",
            "steps": "Roast semolina,Sauté vegetables,Add water and cook,Season with spices",
            "cuisine": "South Indian",
            "cooking_time": 20,
            "serves": 2,
            "image": "upma.jpg"
        },
        {
            "name": "Pongal",
            "ingredients": "rice,moong_dal,pepper,cumin,ginger,curry_leaves",
            "steps": "Pressure cook rice and dal,Temper with spices,Serve hot with chutney",
            "cuisine": "South Indian",
            "cooking_time": 30,
            "serves": 3,
            "image": "pongal.jpg"
        },
        {
            "name": "Bisi Bele Bath",
            "ingredients": "rice,toor_dal,vegetables,bisi_bele_bath_powder,tamarind",
            "steps": "Cook rice and dal together,Add vegetables and spices,Simmer with tamarind pulp",
            "cuisine": "South Indian",
            "cooking_time": 45,
            "serves": 4,
            "image": "bisi_bele_bath.jpg"
        },
        {
            "name": "Rava Kesari",
            "ingredients": "semolina,sugar,ghee,cardamom,cashews,raisins",
            "steps": "Roast semolina in ghee, Add sugar syrup,Mix in dry fruits,Serve warm",
            "cuisine": "South Indian",
            "cooking_time": 25,
            "serves": 4,
            "image": "rava_kesari.jpg"
        }
    ]

class RecipeRecommender:
    def __init__(self):
        """Initialize with comprehensive recipe database"""
        self.BASE_DIR = Path(__file__).parent.parent
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.MODEL_DIR.mkdir(exist_ok=True)

        self.RECIPES = _load_recipe_data()
        self.df = self._initialize_data()
        self.model = self._load_or_train_model()
        self.knn = self._load_or_build_knn()

    def _initialize_data(self):
        """Initialize recipe dataframe with normalized ingredients"""
        df = pd.DataFrame(self.RECIPES)
        df['ingredients'] = df['ingredients'].apply(
            lambda x: [self._normalize_ingredient(i) for i in x.split(',')]
        )
        return df

    @staticmethod
    def _normalize_ingredient(ingredient: str) -> str:
        """Standardize ingredient formatting"""
        return ingredient.strip().lower().replace(' ', '_')

    def _load_or_train_model(self):
        """Load or train Word2Vec model"""
        model_path = self.MODEL_DIR / "word2vec.model"
        if model_path.exists():
            return Word2Vec.load(str(model_path))
        else:
            model = Word2Vec(
                sentences=self.df['ingredients'],
                vector_size=100,
                window=5,
                min_count=1,
                workers=4
            )
            model.save(str(model_path))
            return model

    def _load_or_build_knn(self):
        """Load or build KNN model"""
        knn_path = self.MODEL_DIR / "knn.pkl"
        if knn_path.exists():
            with open(knn_path, 'rb') as f:
                return pickle.load(f)
        else:
            embeddings = [self._get_recipe_embedding(ings) for ings in self.df['ingredients']]
            knn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(embeddings)
            with open(knn_path, 'wb') as f:
                pickle.dump(knn, f)
            return knn

    def _get_recipe_embedding(self, ingredients):
        """Get embedding vector for a recipe"""
        valid_ings = [i for i in ingredients if i in self.model.wv]
        if valid_ings:
            return np.mean([self.model.wv[i] for i in valid_ings], axis=0)
        return np.zeros(self.model.vector_size)

    def recommend(self, user_input: str) -> pd.DataFrame:
        """
        Get recipe recommendations based on ingredients
        Args:
            user_input: Comma-separated ingredient string
        Returns:
            DataFrame of recommended recipes with similarity scores
        """
        try:
            ingredients = self._process_input(user_input)
            if not ingredients:
                return pd.DataFrame()

            avg_vec = self._get_ingredients_vector(ingredients)
            if avg_vec is None:
                return self.df.sample(min(3, len(self.df)))

            distances, indices = self.knn.kneighbors([avg_vec])
            results = self.df.iloc[indices[0]].copy()
            results['similarity'] = 1 - distances[0]
            return results.sort_values('similarity', ascending=False)

        except Exception as e:
            warnings.warn(f"Recommendation error: {str(e)}")
            return self.df.sample(min(3, len(self.df)))

    def _process_input(self, user_input: str) -> list:
        """Process and normalize user input"""
        if not user_input or not isinstance(user_input, str):
            return []
        return [
            self._normalize_ingredient(i)
            for i in user_input.split(',')
            if i.strip()
        ]

    def _get_ingredients_vector(self, ingredients: list):
        """Get average vector for ingredients"""
        valid_ings = [i for i in ingredients if i in self.model.wv]
        if not valid_ings:
            return None
        return np.mean([self.model.wv[i] for i in valid_ings], axis=0)

    def get_recipe_by_name(self, name: str) -> dict:
        """Get complete recipe details by name"""
        try:
            recipe = self.df[self.df['name'].str.lower() == name.lower()].iloc[0].to_dict()
            return {
                'name': recipe['name'],
                'ingredients': recipe['ingredients'],
                'steps': [step.strip() for step in recipe['steps'].split(',')],
                'cuisine': recipe['cuisine'],
                'cooking_time': recipe['cooking_time'],
                'serves': recipe['serves'],
                'image': recipe.get('image', 'default.jpg')
            }
        except IndexError:
            return None