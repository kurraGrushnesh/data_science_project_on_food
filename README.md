# 🚀 Setup & Usage

### 1. Clone the Repository
git clone https://github.com/kurraGrushnesh/data_science_project_on_food.git                   
cd food_recipe_generator

### 2. Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
### 3. Install Dependencies
pip install -r requirements.txt
### 4. Run the Recipe Recommender (Optional)
**Test the backend logic directly:**
python recommend.py
### 5. Launch the Streamlit App:
streamlit run app/gui.py
The app will open in your browser at http://localhost:8501.

## **🔍 Notes:**                                                                                      

**Virtual Environment**: Ensures dependency isolation.                                             

**Requirements**: Ensure requirements.txt includes:

streamlit==1.32.0         # GUI                                                                
pandas==2.0.3            # Data handling                                                       
scikit-learn==1.3.0      # ML model                                                            
gensim==4.3.2            # Word embeddings                                                     
nltk==3.8.1              # NLP processing                                                      
python-dotenv==1.0.0     # API keys                                                            
Pillow==9.5.0  # Recommended stable version
numpy==1.26.0  # exact version       

The script will generate ./models/word2vec.model and ./models/knn.pkl on first execution.      
Subsequent runs will use these cached models.                                                  

### 🐛 Troubleshooting:                                                                            
Missing Images: Place recipe images in ./images/ (e.g., hummus.jpg).                           
Streamlit Errors: Verify gui.py imports RecipeRecommender correctly from recommend.py.                                                                                                                           

# **🍳 Vavi food Recipe Generator**
Discover recipes based on ingredients in your kitchen!                                      
Ajanta is an intelligent recipe recommendation system that suggests dishes based on available ingredients, cuisine preferences, cooking time, and serving size. Built with Python and Streamlit, it helps home 
 cooks and food enthusiasts explore diverse cuisines effortlessly.    
## DEMO:
https://github.com/user-attachments/assets/c7358436-5892-442d-b48c-a66d2a231da0
                                                                                              
### ✨ Features
Smart Ingredient Matching: Finds recipes using what's in your kitchen                          
Multi-Cuisine Recipes: Mediterranean, Italian, Asian, Indian, Mexican, American                
Quick Search: Find recipes by name in sidebar                                                  
Complete Instructions: Steps, prep time, servings & images                                     
Personalized: Filters by cooking time and serving size                                         
                                                                                              
# **Project Structure:**    
food_recipe_generator/                                                                         
│                                                                                              
├── .venv/                   # Virtual environment                                             
├── app/                     # Application files                                               
│   ├── api.py               # API endpoints                                                   
│   └── gui.py               # Streamlit interface                                             
├── data/                    # Recipe datasets
│   ├── processed/
│   │   └── recipes.csv      # Processed recipe data
│   └── raw/
│       └── recipes.json     # Raw recipe data
├── engine/                  # Core recommendation engine
│   ├── __init__.py
│   ├── preprocess.py        # Data preprocessing
│   └── recommend.py         # Recommendation logic
├── images/                  # Recipe images
│   ├── mediterranean/
│   ├── asian/
│   └── ... (other cuisines)
├── models/                  # ML models
│   └── embeddings/
│       └── food2vec.model   # Trained Word2Vec model
├── .gitignore
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

## **Key Features of This Structure:**
### 1.Separation of Concerns:
* Frontend (app/)
* Data (data/)
* Business logic (engine/)
* Assets (images/)

### 2.Modular Design:
* API separated from GUI
* Raw vs processed data distinction
* ML models in dedicated folder

### 3.Scalability:
* Easy to add new cuisines (images/)
* Clean model versioning (models/embeddings/)
* Supports both CSV and JSON data formats

### 4.Reproducibility:
* Virtual environment isolated (.venv/)
* Clear data processing pipeline
* Model artifacts stored separately

