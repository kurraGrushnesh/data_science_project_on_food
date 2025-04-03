import pickle
import warnings
from pathlib import Path
from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors


def _load_recipe_data() -> List[Dict[str, Any]]:
    """Load recipe data with standardized ingredients"""
    return [
        # Mediterranean (10 recipes)
        {

            "name": "Hummus with Pita",
            "ingredients": "chickpeas,tahini,lemon_juice,garlic,olive_oil,pita_bread",
            "steps": "Blend chickpeas,Add tahini and lemon juice,Adjust seasoning,Serve with pita bread",
            "cuisine": "Mediterranean",
            "cooking_time": 15,
            "serves": 4,
            "image": "hummus.jpg"
        },
        {
            "name": "Greek Salad",
            "ingredients": "cucumber,tomatoes,red_onion,feta_cheese,kalamata_olives,olive_oil,oregano",
            "steps": "Chop vegetables,Add feta and olives,Drizzle with oil and oregano",
            "cuisine": "Mediterranean",
            "cooking_time": 10,
            "serves": 2,
            "image": "greek_salad.jpg"
        },
        {
            "name": "Tzatziki Sauce",
            "ingredients": "yogurt,cucumber,garlic,dill,lemon_juice,olive_oil",
            "steps": "Grate cucumber,Mix with yogurt,Add garlic and herbs,Chill before serving",
            "cuisine": "Mediterranean",
            "cooking_time": 15,
            "serves": 4,
            "image": "tzatziki.jpg"
        },
        {
            "name": "Falafel",
            "ingredients": "chickpeas,parsley,garlic,cumin,coriander,flour,oil",
            "steps": "Blend ingredients,Form into balls,Fry until golden,Serve with tahini",
            "cuisine": "Mediterranean",
            "cooking_time": 30,
            "serves": 4,
            "image": "falafel.jpg"
        },
        {
            "name": "Baba Ganoush",
            "ingredients": "eggplant,tahini,lemon_juice,garlic,olive_oil,parsley",
            "steps": "Roast eggplant,Scoop out flesh,Mix with ingredients,Garnish with parsley",
            "cuisine": "Mediterranean",
            "cooking_time": 40,
            "serves": 4,
            "image": "baba_ganoush.jpg"
        },
        {
            "name": "Tabouli Salad",
            "ingredients": "bulgur,parsley,mint,tomatoes,lemon_juice,olive_oil",
            "steps": "Soak bulgur,Chop vegetables,Mix with herbs,Dress with lemon and oil",
            "cuisine": "Mediterranean",
            "cooking_time": 20,
            "serves": 4,
            "image": "tabouli.jpg"
        },
        {
            "name": "Dolma",
            "ingredients": "grape_leaves,rice,onion,pine_nuts,dill,mint,lemon_juice",
            "steps": "Prepare filling,Wrap in grape leaves,Steam until cooked,Serve with lemon",
            "cuisine": "Mediterranean",
            "cooking_time": 60,
            "serves": 6,
            "image": "dolma.jpg"
        },
        {
            "name": "Moussaka",
            "ingredients": "eggplant,potatoes,ground_lamb,tomatoes,cinnamon,béchamel_sauce",
            "steps": "Layer vegetables and meat,Top with béchamel,Bake until golden",
            "cuisine": "Mediterranean",
            "cooking_time": 90,
            "serves": 6,
            "image": "moussaka.jpg"
        },
        {
            "name": "Spanakopita",
            "ingredients": "spinach,feta_cheese,phyllo_dough,eggs,dill,olive_oil",
            "steps": "Prepare filling,Layer with phyllo,Brush with oil,Bake until crispy",
            "cuisine": "Mediterranean",
            "cooking_time": 45,
            "serves": 8,
            "image": "spanakopita.jpg"
        },
        {
            "name": "Shakshuka",
            "ingredients": "eggs,tomatoes,bell_peppers,onion,garlic,paprika,cumin",
            "steps": "Sauté vegetables,Add tomatoes,Make wells for eggs,Poach eggs in sauce",
            "cuisine": "Mediterranean",
            "cooking_time": 30,
            "serves": 4,
            "image": "shakshuka.jpg"
        },

        # Italian (15 recipes)
        {
            "name": "Spaghetti Carbonara",
            "ingredients": "spaghetti,eggs,pancetta,parmesan_cheese,black_pepper",
            "steps": "Cook spaghetti,Fry pancetta,Mix eggs and cheese,Combine all ingredients",
            "cuisine": "Italian",
            "cooking_time": 30,
            "serves": 2,
            "image": "carbonara.jpg"
        },
        {
            "name": "Margherita Pizza",
            "ingredients": "pizza_dough,tomato_sauce,mozzarella_cheese,basil,olive_oil",
            "steps": "Roll out dough,Spread sauce,Add cheese,Bake,Garnish with basil",
            "cuisine": "Italian",
            "cooking_time": 25,
            "serves": 4,
            "image": "margherita.jpg"
        },
        {
            "name": "Risotto Milanese",
            "ingredients": "arborio_rice,saffron,chicken_stock,onion,parmesan_cheese,butter",
            "steps": "Sauté onion,Toast rice,Add stock gradually,Stir in saffron,Finish with butter and cheese",
            "cuisine": "Italian",
            "cooking_time": 40,
            "serves": 4,
            "image": "risotto.jpg"
        },
        {
            "name": "Lasagna",
            "ingredients": "lasagna_noodles,ground_beef,tomato_sauce,ricotta_cheese,mozzarella_cheese",
            "steps": "Layer noodles,meat sauce,and cheeses,Repeat layers,Bake until bubbly",
            "cuisine": "Italian",
            "cooking_time": 75,
            "serves": 8,
            "image": "lasagna.jpg"
        },
        {
            "name": "Minestrone Soup",
            "ingredients": "vegetable_stock,tomatoes,carrots,celery,zucchini,pasta,beans",
            "steps": "Sauté vegetables,Add stock,Simmer with beans and pasta,Season to taste",
            "cuisine": "Italian",
            "cooking_time": 45,
            "serves": 6,
            "image": "minestrone.jpg"
        },
        {
            "name": "Tiramisu",
            "ingredients": "ladyfingers,mascarpone_cheese,espresso,cocoa_powder,eggs,sugar",
            "steps": "Dip ladyfingers in coffee,Layer with mascarpone mixture,Dust with cocoa,Chill",
            "cuisine": "Italian",
            "cooking_time": 30,
            "serves": 8,
            "image": "tiramisu.jpg"
        },
        {
            "name": "Bruschetta",
            "ingredients": "baguette,tomatoes,basil,garlic,olive_oil,balsamic_vinegar",
            "steps": "Toast bread,Top with tomato mixture,Drizzle with balsamic",
            "cuisine": "Italian",
            "cooking_time": 15,
            "serves": 4,
            "image": "bruschetta.jpg"
        },
        {
            "name": "Osso Buco",
            "ingredients": "veal_shanks,carrots,celery,onion,white_wine,chicken_stock",
            "steps": "Brown veal shanks,Sauté vegetables,Deglaze with wine,Braise until tender",
            "cuisine": "Italian",
            "cooking_time": 120,
            "serves": 4,
            "image": "osso_buco.jpg"
        },
        {
            "name": "Panna Cotta",
            "ingredients": "heavy_cream,sugar,vanilla,gelatin,berries",
            "steps": "Heat cream with sugar,Add gelatin,Pour into molds,Chill,Serve with berries",
            "cuisine": "Italian",
            "cooking_time": 20,
            "serves": 6,
            "image": "panna_cotta.jpg"
        },
        {
            "name": "Fettuccine Alfredo",
            "ingredients": "fettuccine,heavy_cream,parmesan_cheese,butter,garlic,nutmeg",
            "steps": "Cook pasta,Make sauce with cream and cheese,Toss with pasta",
            "cuisine": "Italian",
            "cooking_time": 25,
            "serves": 4,
            "image": "alfredo.jpg"
        },
        {
            "name": "Caprese Salad",
            "ingredients": "tomatoes,fresh_mozzarella,basil,olive_oil,balsamic_glaze",
            "steps": "Slice tomatoes and mozzarella,Arrange with basil,Drizzle with oil and glaze",
            "cuisine": "Italian",
            "cooking_time": 10,
            "serves": 2,
            "image": "caprese.jpg"
        },
        {
            "name": "Cannoli",
            "ingredients": "cannoli_shells,ricotta_cheese,powdered_sugar,chocolate_chips,pistachios",
            "steps": "Mix filling ingredients,Pipe into shells,Garnish with pistachios",
            "cuisine": "Italian",
            "cooking_time": 30,
            "serves": 12,
            "image": "cannoli.jpg"
        },
        {
            "name": "Arancini",
            "ingredients": "risotto,eggs,breadcrumbs,mozzarella_cheese,oil",
            "steps": "Form risotto balls,Insert cheese,Dredge in egg and crumbs,Fry until golden",
            "cuisine": "Italian",
            "cooking_time": 30,
            "serves": 6,
            "image": "arancini.jpg"
        },
        {
            "name": "Gnocchi",
            "ingredients": "potatoes,flour,egg,parmesan_cheese,butter,sage",
            "steps": "Mash potatoes,Mix with flour and egg,Form gnocchi,Boil,Sauté with butter and sage",
            "cuisine": "Italian",
            "cooking_time": 45,
            "serves": 4,
            "image": "gnocchi.jpg"
        },
        {
            "name": "Pesto Pasta",
            "ingredients": "pasta,basil,pine_nuts,parmesan_cheese,garlic,olive_oil",
            "steps": "Blend pesto ingredients,Cook pasta,Toss with pesto",
            "cuisine": "Italian",
            "cooking_time": 20,
            "serves": 4,
            "image": "pesto.jpg"
        },

        # Asian (20 recipes)
        {
            "name": "Vegetable Stir Fry",
            "ingredients": "rice,broccoli,carrots,bell_pepper,soy_sauce,ginger",
            "steps": "Cook rice,Stir fry vegetables,Add sauce,Combine with rice",
            "cuisine": "Asian",
            "cooking_time": 25,
            "serves": 3,
            "image": "stirfry.jpg"
        },
        {
            "name": "Chicken Teriyaki",
            "ingredients": "chicken_breasts,soy_sauce,mirin,brown_sugar,ginger,garlic",
            "steps": "Make teriyaki sauce,Grill chicken,Glaze with sauce",
            "cuisine": "Asian",
            "cooking_time": 30,
            "serves": 4,
            "image": "teriyaki.jpg"
        },
        {
            "name": "Beef Bulgogi",
            "ingredients": "beef,soy_sauce,pear,garlic,sesame_oil,brown_sugar",
            "steps": "Marinate beef,Cook on high heat,Serve with rice",
            "cuisine": "Asian",
            "cooking_time": 40,
            "serves": 4,
            "image": "bulgogi.jpg"
        },
        {
            "name": "Sushi Rolls",
            "ingredients": "sushi_rice,nori,avocado,cucumber,imitation_crab,wasabi",
            "steps": "Prepare rice,Lay ingredients on nori,Roll tightly,Slice",
            "cuisine": "Asian",
            "cooking_time": 60,
            "serves": 4,
            "image": "sushi.jpg"
        },
        {
            "name": "Pad Thai",
            "ingredients": "rice_noodles,shrimp,tofu,bean_sprouts,peanuts,pad_thai_sauce",
            "steps": "Soak noodles,Stir fry ingredients,Add noodles and sauce,Garnish",
            "cuisine": "Asian",
            "cooking_time": 35,
            "serves": 4,
            "image": "pad_thai.jpg"
        },
        {
            "name": "Pho",
            "ingredients": "beef_broth,rice_noodles,beef,onion,ginger,star_anise",
            "steps": "Simmer broth,Cook noodles,Slice beef thinly,Assemble bowls",
            "cuisine": "Asian",
            "cooking_time": 120,
            "serves": 6,
            "image": "pho.jpg"
        },
        {
            "name": "Dumplings",
            "ingredients": "dumpling_wrappers,ground_pork,cabbage,ginger,soy_sauce",
            "steps": "Make filling,Fold dumplings,Steam or fry",
            "cuisine": "Asian",
            "cooking_time": 45,
            "serves": 6,
            "image": "dumplings.jpg"
        },
        {
            "name": "Bibimbap",
            "ingredients": "rice,beef,spinach,carrots,bean_sprouts,egg,gochujang",
            "steps": "Prepare toppings,Arrange on rice,Top with egg and sauce",
            "cuisine": "Asian",
            "cooking_time": 40,
            "serves": 2,
            "image": "bibimbap.jpg"
        },
        {
            "name": "Tom Yum Soup",
            "ingredients": "shrimp,lemongrass,kaffir_lime_leaves,galangal,chili,tomatoes",
            "steps": "Simmer aromatics,Add shrimp,Adjust seasoning with lime and fish sauce",
            "cuisine": "Asian",
            "cooking_time": 30,
            "serves": 4,
            "image": "tom_yum.jpg"
        },
        {
            "name": "Ramen",
            "ingredients": "ramen_noodles,pork_broth,chashu_pork,soft_boiled_egg,green_onions",
            "steps": "Cook noodles,Heat broth,Assemble toppings",
            "cuisine": "Asian",
            "cooking_time": 45,
            "serves": 2,
            "image": "ramen.jpg"
        },
        {
            "name": "Spring Rolls",
            "ingredients": "rice_paper,shrimp,rice_vermicelli,lettuce,mint,dipping_sauce",
            "steps": "Soak rice paper,Layer ingredients,Roll tightly,Serve with sauce",
            "cuisine": "Asian",
            "cooking_time": 30,
            "serves": 4,
            "image": "spring_rolls.jpg"
        },
        {
            "name": "Korean Fried Chicken",
            "ingredients": "chicken_wings,flour,cornstarch,gochujang,honey,garlic",
            "steps": "Coat chicken,Fry until crispy,Toss in sauce",
            "cuisine": "Asian",
            "cooking_time": 45,
            "serves": 4,
            "image": "korean_chicken.jpg"
        },
        {
            "name": "Miso Soup",
            "ingredients": "dashi_stock,miso_paste,tofu,wakame,green_onions",
            "steps": "Heat stock,Dissolve miso,Add tofu and seaweed,Garnish",
            "cuisine": "Asian",
            "cooking_time": 15,
            "serves": 4,
            "image": "miso_soup.jpg"
        },
        {
            "name": "Chow Mein",
            "ingredients": "egg_noodles,chicken,bean_sprouts,carrots,celery,soy_sauce",
            "steps": "Cook noodles,Stir fry chicken and vegetables,Combine with noodles",
            "cuisine": "Asian",
            "cooking_time": 25,
            "serves": 4,
            "image": "chow_mein.jpg"
        },
        {
            "name": "Banh Mi",
            "ingredients": "baguette,pork,pate,pickled_vegetables,cucumber,cilantro",
            "steps": "Slice bread,Layer ingredients,Add sauce",
            "cuisine": "Asian",
            "cooking_time": 20,
            "serves": 2,
            "image": "banh_mi.jpg"
        },
        {
            "name": "Peking Duck",
            "ingredients": "duck,honey,soy_sauce,hoisin_sauce,pancakes,scallions",
            "steps": "Prepare duck,Roast until crispy,Serve with pancakes and sauce",
            "cuisine": "Asian",
            "cooking_time": 180,
            "serves": 6,
            "image": "peking_duck.jpg"
        },
        {
            "name": "Hot Pot",
            "ingredients": "broth,thinly_sliced_meat,tofu,vegetables,noodles,dipping_sauces",
            "steps": "Heat broth,Cook ingredients at the table,Dip in sauces",
            "cuisine": "Asian",
            "cooking_time": 60,
            "serves": 6,
            "image": "hot_pot.jpg"
        },
        {
            "name": "Satay",
            "ingredients": "chicken,peanut_butter,soy_sauce,lime_juice,garlic,cumin",
            "steps": "Marinate chicken,Skewer,Grill,Serve with peanut sauce",
            "cuisine": "Asian",
            "cooking_time": 40,
            "serves": 4,
            "image": "satay.jpg"
        },
        {
            "name": "Kimchi Fried Rice",
            "ingredients": "rice,kimchi,spam,egg,green_onions,sesame_oil",
            "steps": "Sauté kimchi and spam,Add rice,Fry egg on top",
            "cuisine": "Asian",
            "cooking_time": 20,
            "serves": 2,
            "image": "kimchi_rice.jpg"
        },
        {
            "name": "Char Siu",
            "ingredients": "pork_shoulder,hoisin_sauce,honey,soy_sauce,five_spice,red_fermented_tofu",
            "steps": "Marinate pork,Roast,Baste with sauce",
            "cuisine": "Asian",
            "cooking_time": 120,
            "serves": 6,
            "image": "char_siu.jpg"
        },
        {
            "name": "Mooncakes",
            "ingredients": "flour,golden_syrup,lotus_seed_paste,egg_yolk,salt",
            "steps": "Make dough,Form around filling,Press in mold,Bake",
            "cuisine": "Asian",
            "cooking_time": 90,
            "serves": 12,
            "image": "mooncakes.jpg"
        },

        # South Indian (20 recipes)
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
            "steps": "Roast semolina in ghee,Add sugar syrup,Mix in dry fruits,Serve warm",
            "cuisine": "South Indian",
            "cooking_time": 25,
            "serves": 4,
            "image": "rava_kesari.jpg"
        },
        {
            "name": "Medu Vada",
            "ingredients": "urad_dal,pepper,cumin,ginger,curry_leaves,oil",
            "steps": "Grind dal to batter,Add spices,Shape and deep fry,Serve hot",
            "cuisine": "South Indian",
            "cooking_time": 30,
            "serves": 4,
            "image": "medu_vada.jpg"
        },
        {
            "name": "Pesarattu",
            "ingredients": "moong_dal,rice,ginger,green_chillies,cumin,oil",
            "steps": "Grind dal and rice,Make thin crepes,Serve with chutney",
            "cuisine": "South Indian",
            "cooking_time": 30,
            "serves": 4,
            "image": "pesarattu.jpg"
        },
        {
            "name": "Avial",
            "ingredients": "mixed_vegetables,yogurt,coconut,green_chillies,curry_leaves,coconut_oil",
            "steps": "Cook vegetables,Grind coconut mixture,Combine with yogurt,Season with curry leaves",
            "cuisine": "South Indian",
            "cooking_time": 40,
            "serves": 6,
            "image": "avial.jpg"
        },
        {
            "name": "Rasam",
            "ingredients": "tamarind,tomatoes,rasam_powder,pepper,cumin,garlic,curry_leaves",
            "steps": "Extract tamarind juice,Add spices and tomatoes,Simmer,Garnish with curry leaves",
            "cuisine": "South Indian",
            "cooking_time": 25,
            "serves": 4,
            "image": "rasam.jpg"
        },
        {
            "name": "Puliyodarai",
            "ingredients": "rice,tamarind,peanuts,sesame_oil,mustard_seeds,curry_leaves",
            "steps": "Prepare tamarind paste,Cook with spices,Mix with rice,Garnish",
            "cuisine": "South Indian",
            "cooking_time": 45,
            "serves": 6,
            "image": "puliyodarai.jpg"
        },
        {
            "name": "Kootu",
            "ingredients": "vegetables,toor_dal,coconut,cumin,pepper,curry_leaves",
            "steps": "Cook dal and vegetables,Grind coconut mixture,Combine and season",
            "cuisine": "South Indian",
            "cooking_time": 35,
            "serves": 4,
            "image": "kootu.jpg"
        },
        {
            "name": "Thengai Sadam",
            "ingredients": "rice,coconut,green_chillies,ginger,curry_leaves,mustard_seeds",
            "steps": "Cook rice,Grind coconut mixture,Temper with spices,Mix with rice",
            "cuisine": "South Indian",
            "cooking_time": 30,
            "serves": 4,
            "image": "thengai_sadam.jpg"
        },
        {
            "name": "Paruppu Usili",
            "ingredients": "toor_dal,beans,red_chillies,asafoetida,curry_leaves,mustard_seeds",
            "steps": "Grind dal with spices,Crumble and cook with beans,Season with tempering",
            "cuisine": "South Indian",
            "cooking_time": 45,
            "serves": 4,
            "image": "paruppu_usili.jpg"
        },
        {
            "name": "Vatha Kuzhambu",
            "ingredients": "tamarind,shallots,sambar_powder,fenugreek,garlic,sesame_oil",
            "steps": "Soak tamarind,Sauté shallots,Add spices and tamarind,Simmer",
            "cuisine": "South Indian",
            "cooking_time": 40,
            "serves": 4,
            "image": "vatha_kuzhambu.jpg"
        },
        {
            "name": "Poriyal",
            "ingredients": "vegetables,coconut,mustard_seeds,urad_dal,green_chillies,curry_leaves",
            "steps": "Sauté vegetables,Add coconut,Season with tempering",
            "cuisine": "South Indian",
            "cooking_time": 20,
            "serves": 4,
            "image": "poriyal.jpg"
        },
        {
            "name": "Appam",
            "ingredients": "rice,coconut,yeast,sugar,salt",
            "steps": "Grind rice and coconut,Ferment with yeast,Cook in appam pan",
            "cuisine": "South Indian",
            "cooking_time": 60,
            "serves": 4,
            "image": "appam.jpg"
        },
        {
            "name": "Kozhukattai",
            "ingredients": "rice_flour,jaggery,coconut,cardamom,ghee",
            "steps": "Prepare filling,Make rice dough,Shape dumplings,Steam",
            "cuisine": "South Indian",
            "cooking_time": 45,
            "serves": 6,
            "image": "kozhukattai.jpg"
        },
        {
            "name": "Semiya Payasam",
            "ingredients": "vermicelli,milk,sugar,cardamom,cashews,raisins",
            "steps": "Roast vermicelli,Cook in milk,Add sugar,Garnish with nuts",
            "cuisine": "South Indian",
            "cooking_time": 30,
            "serves": 6,
            "image": "payasam.jpg"
        },
        {
            "name": "Murukku",
            "ingredients": "rice_flour,urad_dal_flour,sesame_seeds,cumin,butter,oil",
            "steps": "Mix flours with spices,Shape into spirals,Deep fry until crisp",
            "cuisine": "South Indian",
            "cooking_time": 60,
            "serves": 8,
            "image": "murukku.jpg"
        },

        # Mexican (15 recipes)
        {
            "name": "Tacos al Pastor",
            "ingredients": "pork,achiote,pineapple,onion,cilantro,corn_tortillas",
            "steps": "Marinate pork,Grill with pineapple,Chop and serve on tortillas",
            "cuisine": "Mexican",
            "cooking_time": 90,
            "serves": 6,
            "image": "al_pastor.jpg"
        },
        {
            "name": "Guacamole",
            "ingredients": "avocados,tomatoes,onion,cilantro,lime_juice,jalapeño",
            "steps": "Mash avocados,Mix with ingredients,Season to taste",
            "cuisine": "Mexican",
            "cooking_time": 15,
            "serves": 4,
            "image": "guacamole.jpg"
        },
        {
            "name": "Enchiladas",
            "ingredients": "corn_tortillas,chicken,enchilada_sauce,cheese,onion,cream",
            "steps": "Fill tortillas with chicken,Roll and place in dish,Cover with sauce and cheese,Bake",
            "cuisine": "Mexican",
            "cooking_time": 45,
            "serves": 6,
            "image": "enchiladas.jpg"
        },
        {
            "name": "Chiles Rellenos",
            "ingredients": "poblano_peppers,cheese,eggs,flour,tomato_sauce",
            "steps": "Roast peppers,Stuff with cheese,Batter and fry,Top with sauce",
            "cuisine": "Mexican",
            "cooking_time": 60,
            "serves": 4,
            "image": "chiles_rellenos.jpg"
        },
        {
            "name": "Pozole",
            "ingredients": "hominy,pork,chili_powder,garlic,oregano,radishes",
            "steps": "Cook pork and hominy,Season with spices,Serve with garnishes",
            "cuisine": "Mexican",
            "cooking_time": 180,
            "serves": 8,
            "image": "pozole.jpg"
        },
        {
            "name": "Quesadillas",
            "ingredients": "flour_tortillas,cheese,chicken,peppers,onion,salsa",
            "steps": "Fill tortillas with cheese and fillings,Cook until melted,Serve with salsa",
            "cuisine": "Mexican",
            "cooking_time": 20,
            "serves": 4,
            "image": "quesadillas.jpg"
        },
        {
            "name": "Churros",
            "ingredients": "flour,water,sugar,cinnamon,oil,chocolate_sauce",
            "steps": "Make dough,Pipe into hot oil,Fry until golden,Roll in cinnamon sugar",
            "cuisine": "Mexican",
            "cooking_time": 30,
            "serves": 6,
            "image": "churros.jpg"
        },
        {
            "name": "Mole Poblano",
            "ingredients": "chicken,chocolate,chilies,almonds,raisins,sesame_seeds",
            "steps": "Make mole sauce,Simmer chicken in sauce,Garnish with sesame seeds",
            "cuisine": "Mexican",
            "cooking_time": 150,
            "serves": 8,
            "image": "mole.jpg"
        },
        {
            "name": "Tamales",
            "ingredients": "masa,chicken,chili_sauce,corn_husks,lard",
            "steps": "Prepare masa,Spread on husks,Add filling,Steam until cooked",
            "cuisine": "Mexican",
            "cooking_time": 120,
            "serves": 12,
            "image": "tamales.jpg"
        },
        {
            "name": "Horchata",
            "ingredients": "rice,cinnamon,vanilla,sugar,milk",
            "steps": "Soak rice with cinnamon,Blend and strain,Add milk and sugar,Chill",
            "cuisine": "Mexican",
            "cooking_time": 15,
            "serves": 4,
            "image": "horchata.jpg"
        },
        {
            "name": "Ceviche",
            "ingredients": "fish,shrimp,lime_juice,tomatoes,onion,cilantro",
            "steps": "Marinate seafood in lime juice,Add vegetables,Chill before serving",
            "cuisine": "Mexican",
            "cooking_time": 30,
            "serves": 4,
            "image": "ceviche.jpg"
        },
        {
            "name": "Flan",
            "ingredients": "eggs,condensed_milk,vanilla,sugar",
            "steps": "Make caramel,Custard mixture,Bake in water bath,Chill",
            "cuisine": "Mexican",
            "cooking_time": 60,
            "serves": 8,
            "image": "flan.jpg"
        },
        {
            "name": "Sopes",
            "ingredients": "masa,beans,cheese,lettuce,cream,salsa",
            "steps": "Form masa disks,Fry,Top with beans and garnishes",
            "cuisine": "Mexican",
            "cooking_time": 45,
            "serves": 6,
            "image": "sopes.jpg"
        },
        {
            "name": "Barbacoa",
            "ingredients": "beef,achiote,garlic,cumin,orange_juice,bay_leaves",
            "steps": "Marinate beef,Slow cook until tender,Shred and serve",
            "cuisine": "Mexican",
            "cooking_time": 240,
            "serves": 8,
            "image": "barbacoa.jpg"
        },
        {
            "name": "Mexican Street Corn",
            "ingredients": "corn,mayonnaise,cotija_cheese,chili_powder,lime_juice",
            "steps": "Grill corn,Coat with mayo,Sprinkle cheese and chili,Squeeze lime",
            "cuisine": "Mexican",
            "cooking_time": 20,
            "serves": 4,
            "image": "street_corn.jpg"
        },

        # American (20 recipes)
        {
            "name": "Classic Burger",
            "ingredients": "ground_beef,buns,cheese,lettuce,tomato,onion,pickles",
            "steps": "Form patties,Cook to desired doneness,Toast buns,Assemble with toppings",
            "cuisine": "American",
            "cooking_time": 25,
            "serves": 4,
            "image": "burger.jpg"
        },
        {
            "name": "Mac and Cheese",
            "ingredients": "macaroni,cheddar_cheese,milk,butter,flour,breadcrumbs",
            "steps": "Cook pasta,Make cheese sauce,Combine,Bake with breadcrumb topping",
            "cuisine": "American",
            "cooking_time": 45,
            "serves": 6,
            "image": "mac_cheese.jpg"
        },
        {
            "name": "Fried Chicken",
            "ingredients": "chicken,buttermilk,flour,paprika,garlic_powder,oil",
            "steps": "Marinate chicken in buttermilk,Coat with flour mixture,Fry until golden",
            "cuisine": "American",
            "cooking_time": 60,
            "serves": 6,
            "image": "fried_chicken.jpg"
        },
        {
            "name": "BBQ Ribs",
            "ingredients": "pork_ribs,bbq_sauce,brown_sugar,paprika,garlic_powder",
            "steps": "Season ribs,Slow cook,Baste with sauce,Finish on grill",
            "cuisine": "American",
            "cooking_time": 240,
            "serves": 6,
            "image": "bbq_ribs.jpg"
        },
        {
            "name": "Apple Pie",
            "ingredients": "apples,sugar,cinnamon,pie_crust,butter,lemon_juice",
            "steps": "Prepare filling,Line pie dish,Add filling,Top with crust,Bake",
            "cuisine": "American",
            "cooking_time": 90,
            "serves": 8,
            "image": "apple_pie.jpg"
        },
        {
            "name": "Pancakes",
            "ingredients": "flour,milk,eggs,baking_powder,sugar,butter",
            "steps": "Mix dry ingredients,Add wet ingredients,Cook on griddle,Serve with syrup",
            "cuisine": "American",
            "cooking_time": 20,
            "serves": 4,
            "image": "pancakes.jpg"
        },
        {
            "name": "Clam Chowder",
            "ingredients": "clams,potatoes,onion,bacon,heavy_cream,thyme",
            "steps": "Cook bacon,Sauté vegetables,Add clams and potatoes,Finish with cream",
            "cuisine": "American",
            "cooking_time": 45,
            "serves": 6,
            "image": "clam_chowder.jpg"
        },
        {
            "name": "Buffalo Wings",
            "ingredients": "chicken_wings,hot_sauce,butter,blue_cheese,celery",
            "steps": "Bake or fry wings,Toss in sauce,Serve with blue cheese and celery",
            "cuisine": "American",
            "cooking_time": 45,
            "serves": 4,
            "image": "buffalo_wings.jpg"
        },
        {
            "name": "Cornbread",
            "ingredients": "cornmeal,flour,buttermilk,eggs,baking_powder,sugar",
            "steps": "Mix dry ingredients,Add wet ingredients,Bake until golden",
            "cuisine": "American",
            "cooking_time": 30,
            "serves": 8,
            "image": "cornbread.jpg"
        },
        {
            "name": "Cobb Salad",
            "ingredients": "lettuce,chicken,bacon,avocado,eggs,tomatoes,blue_cheese",
            "steps": "Chop ingredients,Arrange in sections,Top with dressing",
            "cuisine": "American",
            "cooking_time": 30,
            "serves": 2,
            "image": "cobb_salad.jpg"
        },
        {
            "name": "Pot Roast",
            "ingredients": "beef_roast,potatoes,carrots,onion,beef_broth,rosemary",
            "steps": "Sear meat,Add vegetables and broth,Slow cook until tender",
            "cuisine": "American",
            "cooking_time": 240,
            "serves": 6,
            "image": "pot_roast.jpg"
        },
        {
            "name": "Cheesecake",
            "ingredients": "cream_cheese,eggs,sugar,graham_crackers,butter,vanilla",
            "steps": "Make crust,Mix filling,Bake in water bath,Chill",
            "cuisine": "American",
            "cooking_time": 90,
            "serves": 10,
            "image": "cheesecake.jpg"
        },
        {
            "name": "Reuben Sandwich",
            "ingredients": "rye_bread,corned_beef,sauerkraut,swiss_cheese,russian_dressing",
            "steps": "Layer ingredients,Grill until cheese melts",
            "cuisine": "American",
            "cooking_time": 15,
            "serves": 2,
            "image": "reuben.jpg"
        },
        {
            "name": "Chocolate Chip Cookies",
            "ingredients": "flour,butter,sugar,eggs,chocolate_chips,vanilla",
            "steps": "Cream butter and sugar,Add dry ingredients,Mix in chips,Bake",
            "cuisine": "American",
            "cooking_time": 25,
            "serves": 24,
            "image": "chocolate_chip.jpg"
        },
        {
            "name": "Meatloaf",
            "ingredients": "ground_beef,breadcrumbs,eggs,onion,ketchup,mustard",
            "steps": "Mix ingredients,Form loaf,Top with glaze,Bake",
            "cuisine": "American",
            "cooking_time": 75,
            "serves": 6,
            "image": "meatloaf.jpg"
        },
        {
            "name": "Brownies",
            "ingredients": "chocolate,butter,eggs,sugar,flour,walnuts",
            "steps": "Melt chocolate,Mix ingredients,Bake until set,Cool before cutting",
            "cuisine": "American",
            "cooking_time": 35,
            "serves": 12,
            "image": "brownies.jpg"
        },
        {
            "name": "Chicken Pot Pie",
            "ingredients": "chicken,vegetables,pie_crust,chicken_stock,cream,thyme",
            "steps": "Cook filling,Transfer to dish,Top with crust,Bake until golden",
            "cuisine": "American",
            "cooking_time": 60,
            "serves": 6,
            "image": "pot_pie.jpg"
        },
        {
            "name": "Biscuits and Gravy",
            "ingredients": "flour,buttermilk,baking_powder,sausage,milk,pepper",
            "steps": "Make biscuits,Cook sausage,Make gravy,Serve together",
            "cuisine": "American",
            "cooking_time": 35,
            "serves": 4,
            "image": "biscuits_gravy.jpg"
        },
        {
            "name": "Peanut Butter Cookies",
            "ingredients": "peanut_butter,sugar,egg,flour,baking_soda,vanilla",
            "steps": "Mix ingredients,Form cookies,Press with fork,Bake",
            "cuisine": "American",
            "cooking_time": 20,
            "serves": 24,
            "image": "pb_cookies.jpg"
        },
        {
            "name": "Pulled Pork Sandwich",
            "ingredients": "pork_shoulder,bbq_sauce,buns,coleslaw,pickle_spears",
            "steps": "Slow cook pork,Shred and mix with sauce,Serve on buns with coleslaw",
            "cuisine": "American",
            "cooking_time": 360,
            "serves": 8,
            "image": "pulled_pork.jpg"
        }
    ]


class RecipeRecommender:
    def __init__(self):
        """Initialize with comprehensive recipe database"""
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.IMAGES_DIR = self.BASE_DIR / "images"  # Path for recipe images
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.IMAGES_DIR.mkdir(exist_ok=True, parents=True)

        self.RECIPES = _load_recipe_data()
        self.RECIPE_IMAGES = [r['image'] for r in self.RECIPES]  # List of all image filenames
        self.df = self._initialize_data()
        self.model = self._load_or_train_model()
        self.knn = self._load_or_build_knn()

    def check_missing_images(self) -> List[str]:
        """Check which recipe images are missing from the images directory

        Returns:
            List of missing image filenames
        """
        return [img for img in self.RECIPE_IMAGES
                if not (self.IMAGES_DIR / img).exists()]

    def _initialize_data(self) -> pd.DataFrame:
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

    def _load_or_train_model(self) -> Word2Vec:
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

    def _load_or_build_knn(self) -> NearestNeighbors:
        """Load or build KNN model"""
        knn_path = self.MODEL_DIR / "knn.pkl"
        if knn_path.exists():
            with open(knn_path, 'rb') as f:
                return pickle.load(f)
        else:
            embeddings = [self._get_recipe_embedding(ings) for ings in self.df['ingredients']]
            knn = NearestNeighbors(n_neighbors=min(5, len(embeddings)), metric='cosine').fit(embeddings)
            with open(knn_path, 'wb') as f:
                pickle.dump(knn, f)
            return knn

    def _get_recipe_embedding(self, ingredients: List[str]) -> np.ndarray:
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

    def _process_input(self, user_input: str) -> List[str]:
        """Process and normalize user input"""
        if not user_input or not isinstance(user_input, str):
            return []
        return [
            self._normalize_ingredient(i)
            for i in user_input.split(',')
            if i.strip()
        ]

    def _get_ingredients_vector(self, ingredients: List[str]) -> Optional[np.ndarray]:
        """Get average vector for ingredients"""
        valid_ings = [i for i in ingredients if i in self.model.wv]
        if not valid_ings:
            return None
        return np.mean([self.model.wv[i] for i in valid_ings], axis=0)

    def get_recipe_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get complete recipe details by name (case-insensitive)"""
        try:
            recipe_df = self.df[self.df['name'].str.lower() == name.lower()]
            if recipe_df.empty:
                return None

            recipe = recipe_df.iloc[0].to_dict()
            return {
                'name': recipe['name'],
                'ingredients': recipe['ingredients'],
                'steps': [step.strip() for step in recipe['steps'].split(',')],
                'cuisine': recipe['cuisine'],
                'cooking_time': recipe['cooking_time'],
                'serves': recipe['serves'],
                'image': recipe.get('image', 'default.jpg')
            }
        except Exception as e:
            warnings.warn(f"Error getting recipe by name: {str(e)}")
            return None

    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """Get all recipes in the database"""
        return self.RECIPES

    def get_recipes_by_ingredients(self, ingredients: List[str]) -> List[Dict[str, Any]]:
        """Get recipes that contain any of the specified ingredients"""
        normalized_ingredients = [self._normalize_ingredient(i) for i in ingredients]
        return [
            recipe for recipe in self.RECIPES
            if any(ing in [self._normalize_ingredient(i) for i in recipe['ingredients'].split(',')]
                   for ing in normalized_ingredients)
        ]
