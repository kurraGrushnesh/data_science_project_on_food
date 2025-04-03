import logging
import random
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st
from PIL import Image

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
ENGINE_DIR = PROJECT_ROOT / "src"
sys.path.append(str(ENGINE_DIR))  # Ensure 'src' is in path

# Import RecipeRecommender safely
try:
    from train import RecipeRecommender
except ImportError as e:
    st.error("⚠️ Failed to load RecipeRecommender. Ensure 'src/train.py' exists.")
    st.error(f"Error details: {e}")
    st.stop()

# Initialize recommender and get recipes
recommender = RecipeRecommender()
RECIPES = recommender.get_all_recipes()


def configure_page() -> None:
    """Sets Streamlit page config and styles"""
    st.set_page_config(
        page_title="VAVI Recipes Generator",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
            .stApp {background: linear-gradient(315deg, black 25%, white 25%);}
            .recipe-card {border-radius: 35px; box-shadow: 0 15px 45px rgba(0,0,0,0.15);
                          padding: 20px; margin-bottom: 75px; 
                          transition: transform 0.3s ease;   color: white;
                           background: linear-gradient(135deg, black, black);}
            .recipe-card:hover {transform: translateY(-5px);
                                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                                 }
            .stButton>button {background: linear-gradient(135deg, black, #8BC34A);
                              color: white; border-radius: 8px; padding: 10px 24px;
                              font-weight: 600; transition: all 0.3s;}
            .stButton>button:hover {transform: scale(1.05);
                                    box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);}
            [data-testid="stSidebar"] {background: linear-gradient(180deg, #4CAF50, #2E7D32) !important;}
            .ingredient-chip {display: inline-block; padding: 4px 8px; margin: 2px;
                              background: #e3f2fd; border-radius: 16px; font-size: 0.9em;}
            .sidebar-buttons {display: flex; gap: 10px;}
        </style>
    """, unsafe_allow_html=True)


# noinspection PyUnresolvedReferences
def load_image(image_name: str) -> Image.Image:
    """Safely loads an image from the images directory"""
    image_path = IMAGES_DIR / image_name
    try:
        if image_path.exists():
            return Image.open(image_path)
        return Image.new('RGB', (400, 300), color=(240, 240, 240))  # Default placeholder
    except Exception as p:
        st.warning(f"Couldn't load image {image_name}: {p}")
        return Image.new('RGB', (400, 300), color=(240, 240, 240))


def display_recipe(recipe: Dict[str, Any], expanded: bool = True) -> None:
    """Displays a recipe in an expandable card with enhanced layout"""
    with st.expander(f"🍴 {recipe['name']}", expanded=expanded):
        col1, col2 = st.columns([1, 2])

        with col1:
            img = load_image(recipe.get('image', 'default.jpg'))
            st.image(img, use_container_width=True,
                     caption=f"{recipe['cuisine']} Cuisine • ⏱️ {recipe['cooking_time']} min • 👥 Serves {recipe['serves']}")

        with col2:
            st.markdown(f"### {recipe['name']}")

            # Display ingredients with chips
            st.markdown("#### 🛒 Ingredients")
            cols = st.columns(3)
            for i, ing in enumerate(
                    recipe['ingredients'] if isinstance(recipe['ingredients'], list) else recipe['ingredients'].split(
                        ',')):
                with cols[i % 3]:
                    st.markdown(f'<div class="ingredient-chip">{ing.strip().title()}</div>', unsafe_allow_html=True)

            # Display steps with better formatting
            st.markdown("#### 👩‍🍳 Preparation")
            for i, step in enumerate(
                    recipe['steps'] if isinstance(recipe['steps'], list) else recipe['steps'].split(','), 1):
                st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <span style="font-weight: bold; color: #2E7D32;">{i}.</span> 
                        <span>{step.strip()}</span>
                    </div>
                """, unsafe_allow_html=True)


def show_sidebar() -> None:
    """Enhanced sidebar with improved layout and functionality"""
    with st.sidebar:
        st.markdown("## 🧑‍🍳 Chef's Toolkit")

        # Initialize session state for search input
        if 'recipe_search_input' not in st.session_state:
            st.session_state.recipe_search_input = ""

        # Cooking tips section
        if st.button("✨ Get Pro Tip", key="pro_tip_button"):
            tips = [
                "Toast your spices before grinding for better flavor.",
                "Use room temperature ingredients for even baking.",
                "Always let your meat rest before slicing.",
                "Salt pasta water like the sea for better seasoning.",
                "Deglaze your pan with wine for a delicious sauce."
            ]
            st.success(f"👨‍🍳 **Pro Tip**: {random.choice(tips)}")

        st.markdown("---")

        # Enhanced recipe search section
        st.markdown("## 🔍 Find Recipe by Name")

        # Create the search input widget
        recipe_name = st.text_input(
            "Enter recipe name:",
            value=st.session_state.recipe_search_input,
            key="recipe_search_input_widget",  # Different key for the widget
            placeholder="e.g., Vada, Carbonara..."
        )

        # Use HTML/CSS for side-by-side buttons instead of columns
        st.markdown('<div class="sidebar-buttons">', unsafe_allow_html=True)

        if st.button("🔍 Search", key="search_button"):
            if recipe_name:
                try:
                    matching_recipes = [r for r in RECIPES if recipe_name.lower() in r['name'].lower()]
                    if matching_recipes:
                        # Move display to main area
                        st.session_state.sidebar_search_results = matching_recipes
                        st.session_state.recipe_search_input = recipe_name  # Update session state
                    else:
                        st.warning("Recipe not found. Try another name.")
                except Exception as p:
                    st.error(f"Error fetching recipe: {str(p)}")
            else:
                st.warning("Please enter a recipe name")

        if st.button("🗑️ Clear", key="clear_search_button"):
            # Clear through callback rather than direct modification
            st.session_state.recipe_search_input = ""
            if 'sidebar_search_results' in st.session_state:
                del st.session_state.sidebar_search_results
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Enhanced ingredient suggestions
        st.markdown("## 🛒 Common Ingredients")
        all_ingredients = list(set(ing for recipe in RECIPES for ing in recipe['ingredients'].split(',')))
        if st.button("💡 Suggest Ingredients", key="suggest_button"):
            st.session_state.suggested_ingredients = ", ".join(random.sample(all_ingredients, 5))

        if 'suggested_ingredients' in st.session_state:
            st.text_area("Suggested ingredients:",
                         value=st.session_state.suggested_ingredients,
                         height=68,
                         key="suggested_ingredients_display")


def main_interface() -> None:
    """Main user input and recipe recommendation interface"""
    # Display sidebar search results in main area if they exist
    if 'sidebar_search_results' in st.session_state:
        st.markdown("## 🔍 Search Results")
        for recipe in st.session_state.sidebar_search_results:
            display_recipe(recipe)
        st.markdown("---")

    st.markdown("## 🔍 What's in your kitchen?")

    # Initialize session state for ingredients if not exists
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = ""

    with st.form("recipe_form", clear_on_submit=False):
        user_input = st.text_area(
            "Enter ingredients (comma separated):",
            value=st.session_state.ingredients,
            placeholder="e.g., rice, tomatoes, potatoes...",
            height=100,
            key="ingredient_input"
        )

        cols = st.columns(2)
        with cols[0]:
            cuisine_options = ["Any"] + sorted(list(set(recipe['cuisine'] for recipe in RECIPES)))
            cuisine_pref = st.multiselect(
                "Preferred Cuisines:",
                cuisine_options,
                default=["Any"]
            )
        with cols[1]:
            max_time = st.slider(
                "Max Cooking Time (minutes):",
                min_value=10,
                max_value=120,
                value=60,
                step=5,
                help="Filter recipes by maximum preparation time"
            )

        submitted = st.form_submit_button("✨ Find Matching Recipes")

    if submitted:
        st.session_state.ingredients = user_input  # Save to session state

        if not user_input.strip():
            st.warning("Please enter ingredients to get started!")
            st.image(load_image("empty_kitchen.jpg"), width=300)
        else:
            with st.spinner("🧑‍🍳 Finding matching recipes..."):
                try:
                    input_ingredients = [ing.strip().lower() for ing in user_input.split(',') if ing.strip()]

                    # Find recipes that contain at least one of the input ingredients
                    matching_recipes = []
                    for recipe in RECIPES:
                        recipe_ingredients = [ing.strip().lower() for ing in recipe['ingredients'].split(',')]
                        if any(ing in recipe_ingredients for ing in input_ingredients):
                            if ("Any" in cuisine_pref or recipe['cuisine'] in cuisine_pref) and recipe[
                                'cooking_time'] <= max_time:
                                matching_recipes.append(recipe)

                    if matching_recipes:
                        st.success(f"🍽️ Found {len(matching_recipes)} matching recipes!")
                        for recipe in matching_recipes:
                            display_recipe(recipe)
                    else:
                        st.info(
                            "No recipes match your ingredients and filters. Try different ingredients or broaden your filters.")
                        # Show sample recipes
                        st.markdown("### Here are some sample recipes:")
                        for recipe in random.sample(RECIPES, min(3, len(RECIPES))):
                            display_recipe(recipe)
                except Exception as p:
                    st.error(f"⚠️ Error finding recipes: {str(p)}")
                    logging.error(f"Recipe search error: {str(p)}")


def main() -> None:
    """Main application function"""
    configure_page()

    # Header section
    st.title("🍳 VAVI Recipes Generator")
    st.markdown("""
        <div style="margin-bottom: 30px;">
            Discover delicious recipes tailored to your ingredients and preferences! 
            Enter what you have in your kitchen, and we'll suggest the perfect dishes.
        </div>
    """, unsafe_allow_html=True)

    show_sidebar()
    main_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <h3>Happy Cooking! 👨‍🍳👩‍🍳</h3>
            <p>Made with ❤️ by VAVI Recipes</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
