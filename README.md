🚀 Setup & Usage
1. Clone the Repository
bash
Copy
git clone https://github.com/kurraGrushnesh/data_science_project_on_food.git
cd food_recipe_generator
2. Create and Activate Virtual Environment
bash
Copy
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# For macOS/Linux: source .venv/bin/activate
3. Install Dependencies
bash
Copy
pip install -r requirements.txt
4. Run the Recipe Recommender (Optional)
Test the backend logic directly:

bash
Copy
python recommend.py
5. Launch the Streamlit App
bash
Copy
streamlit run app/gui.py
The app will open in your browser at http://localhost:8501.

🔍 Notes:
Virtual Environment: Ensures dependency isolation.

Requirements: Ensure requirements.txt includes:

plaintext
Copy
streamlit
pandas
gensim
scikit-learn
numpy
pickle-mixin
First Run:

The script will generate ./models/word2vec.model and ./models/knn.pkl on first execution.

Subsequent runs will use these cached models.

🐛 Troubleshooting
Missing Images: Place recipe images in ./images/ (e.g., hummus.jpg).

Streamlit Errors: Verify gui.py imports RecipeRecommender correctly from recommend.py.
