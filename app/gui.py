import random
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
sys.path.append(str(PROJECT_ROOT))

try:
    from engine.recommend import RecipeRecommender
except ImportError as e:
    st.error(f"Failed to import RecipeRecommender: {str(e)}. Please check:")
    st.error("1. The 'engine' folder exists in your project")
    st.error("2. It contains a 'recommend.py' file with RecipeRecommender class")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()


def configure_page():
    st.set_page_config(
        page_title="🍳 VAVI RECIPES GENERATOR",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        background-image: linear-gradient(315deg, #f8f9fa 0%, #e9ecef 74%);
    }
    .recipe-card {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 25px;
        background: white;
        transition: transform 0.3s ease;
    }
    .recipe-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4CAF50 0%, #2E7D32 100%) !important;
    }
    .ingredient-chip {
        display: inline-block;
        background: #e8f5e9;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 14px;
    }
    .step-card {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #2E7D32;
    }
    .step-number {
        background: #2E7D32;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def load_image(image_name):
    try:
        image_path = IMAGES_DIR / image_name
        if image_path.exists():
            return Image.open(image_path)
        return Image.open(IMAGES_DIR / "default.jpg")
    except Exception as p:
        st.warning(f"Image loading error: {str(p)}")
        return Image.new('RGB', (400, 300), color=(240, 240, 240))


def display_recipe(recipe, expanded=True):
    """Professional recipe card display with numbered steps"""
    with st.expander(f"🍴 {recipe['name']}", expanded=expanded):
        col1, col2 = st.columns([1, 2])

        with col1:
            img = load_image(recipe.get('image', 'default.jpg'))
            st.image(img, use_column_width=True,
                     caption=f"{recipe['cuisine']} Cuisine • ⏱️ {recipe['cooking_time']} min • 👥 Serves {recipe['serves']}")

        with col2:
            st.markdown(f"### {recipe['name']}")

            # Ingredients with chips
            st.markdown("#### 🛒 Ingredients")
            cols = st.columns(3)
            for i, ing in enumerate(recipe['ingredients']):
                with cols[i % 3]:
                    st.markdown(f'<div class="ingredient-chip">{ing.replace("_", " ").title()}</div>',
                                unsafe_allow_html=True)

            # Preparation steps with clear numbering
            st.markdown("#### 👩‍🍳 Preparation")

            # Handle both string and list input for steps
            if isinstance(recipe['steps'], str):
                # Split steps by numbers (1., 2., etc.) or newlines
                steps = [step.strip() for step in recipe['steps'].split('\n') if step.strip()]
                if len(steps) == 1:  # If still one long string, try splitting by numbers
                    steps = [step.strip() for step in recipe['steps'].split(r'\d+\.') if step.strip()]
            else:
                steps = recipe['steps']

            # Display each step with a numbered circle
            for i, step in enumerate(steps, 1):
                st.markdown(f"""
                <div class="step-card">
                    <div class="step-number">{i}</div>
                    <span style="vertical-align: middle;">{step}</span>
                </div>
                """, unsafe_allow_html=True)


def show_hero():
    """Professional hero section"""
    hero = st.container()
    with hero:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🍳 VAVI RECIPES GENERATOR")
            st.markdown("""
            <p style="font-size:18px;color:#555;">
            Discover perfect recipes tailored to your ingredients. Our AI-powered system reduces 
            food waste while helping you create delicious meals!
            </p>
            """, unsafe_allow_html=True)
        with col2:
            chef_img = load_image("chef_icon.png")
            st.image(chef_img, width=150)


def show_sidebar():
    """Enhanced sidebar with more features"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;margin-bottom:30px;">
            <h2 style="color:white;">Chef's Toolkit</h2>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ Get Pro Tip"):
            tips = [
                "For best results, let dough rest for at least 30 minutes",
                "Always toast spices before grinding for maximum flavor",
                "Use a meat thermometer for perfect doneness every time",
                "Deglaze your pan with wine or stock for amazing sauces",
                "Salt your pasta water like the sea for perfect seasoning"
            ]
            st.success(f"👨‍🍳 **Pro Tip**: {random.choice(tips)}")

        st.markdown("---")

        st.markdown("### 🔍 Find Specific Recipe")
        recipe_name = st.text_input("Enter recipe name:")
        if recipe_name:
            try:
                recommender = RecipeRecommender()
                recipe = recommender.get_recipe_by_name(recipe_name)
                display_recipe(recipe)
            except Exception:
                st.warning("Recipe not found. Try another name.")


def main_interface():
    """Main recommendation interface"""
    st.markdown("## 🔍 What's in your kitchen?")

    with st.form("recipe_form"):
        user_input = st.text_area(
            "Enter ingredients (comma separated):",
            placeholder="e.g., chicken, rice, tomatoes, garlic...",
            height=100,
            help="List what you have available"
        )

        cols = st.columns(2)
        with cols[0]:
            cuisine_pref = st.multiselect(
                "Preferred cuisines:",
                ["Any", "Asian", "Italian", "Indian", "Mediterranean", "American"],
                default=["Any"]
            )
        with cols[1]:
            max_time = st.slider(
                "Max cooking time (minutes):",
                10, 180, 60,
                help="Filter by preparation time"
            )

        submitted = st.form_submit_button("✨ Generate Recipes")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter some ingredients to get started")
            st.image(load_image("empty_kitchen.jpg"),
                     caption="Your kitchen is empty! Add some ingredients.")
        else:
            with st.spinner("🧑‍🍳 Consulting our culinary AI..."):
                try:
                    time.sleep(1)  # Artificial delay for better UX

                    recommender = RecipeRecommender()
                    results = recommender.recommend(user_input)

                    if results.empty:
                        st.info("No exact matches found. Try these delicious alternatives:")
                        st.image(load_image("no_recipes.jpg"),
                                 caption="Try different ingredients or broaden your filters")
                        sample_recipes = recommender.df.sample(2).to_dict('records')
                        for recipe in sample_recipes:
                            display_recipe(recipe)
                    else:
                        # Apply filters
                        if max_time < 180:
                            results = results[results['cooking_time'] <= max_time]
                        if "Any" not in cuisine_pref and cuisine_pref:
                            results = results[results['cuisine'].isin(cuisine_pref)]

                        if not results.empty:
                            st.success(f"🍽️ Found {len(results)} perfect recipes for you!")
                            for _, recipe in results.iterrows():
                                display_recipe(recipe.to_dict())
                        else:
                            st.info("No recipes match your filters. Try being more flexible!")
                            st.image(load_image("adapt_recipe.jpg"),
                                     caption="Great chefs adapt! Try different combinations")

                except Exception as p:
                    st.error("Our culinary AI encountered an issue!")
                    st.image(load_image("kitchen_problem.jpg"),
                             caption="Our team is working on it")
                    st.error(f"Technical details: {str(p)}")


def main():
    configure_page()
    show_hero()
    show_sidebar()
    main_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:30px;background:#f8f9fa;border-radius:10px;">
        <h3 style="color:#2E7D32;">Happy Cooking! 👨‍🍳👩‍🍳</h3>
        <p>The secret ingredient is always passion and love ❤️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
