from absl import app


@app.post("/generate_recipe")
def generate_recipe(ingredients: list):
    recipe = model.predict(ingredients)
    return {"recipe": recipe}