import random
import json
import os
import math  # Added for calculating GCD for simplified ratios

#  Step 1: Load Dataset
default_dataset_path = "D:/final_year_project/final_diet_plan_dataset.json"

if not os.path.exists(default_dataset_path):
    print(f"⚠️ Warning: Default dataset not found at {default_dataset_path}")
    dataset_path = input("Enter the correct dataset path: ").strip()
else:
    dataset_path = default_dataset_path

try:
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    print(f"✅ Dataset Loaded Successfully from {dataset_path}")
except Exception as e:
    print(f"❌ Error Loading Dataset: {e}")
    exit()

# ✅ Remove zero-calorie foods
dataset = [food for food in dataset if food["calories_per_100g"] > 0]

# ✅ Step 2: Meal Planning Functions
def meets_goal_criteria(food, goal):
    """User goal ke hisaab se food filter kare."""
    goal_filters = {
        "muscle_gain": {"tags": ["high-protein", "muscle-gain"], "min_protein": 20},
        "fat_loss": {"tags": ["low-calorie", "high-fiber"], "max_cal_per_100g": 150},
        "bulking": {"tags": ["high-calorie", "energy-dense"], "min_cal_per_100g": 300}
    }

    if goal not in goal_filters:
        return True  

    filters = goal_filters[goal]
    if any(tag in food["tags"] for tag in filters["tags"]):
        if "min_protein" in filters and food["protein_g"] < filters["min_protein"]:
            return False
        if "max_cal_per_100g" in filters and food["calories_per_100g"] > filters["max_cal_per_100g"]:
            return False
        if "min_cal_per_100g" in filters and food["calories_per_100g"] < filters["min_cal_per_100g"]:
            return False
        return True
    return False

def get_candidate_foods(goal, dataset):
    """Filter foods based on user goal."""
    return [food for food in dataset if meets_goal_criteria(food, goal)]

def compute_fit_score(food, goal):
    """Food ka fitness score calculate kare based on goal."""
    if goal == "muscle_gain":
        return food["protein_g"] / max(food["calories_per_100g"], 1)
    elif goal == "fat_loss":
        return food["fiber_g"] / max(food["calories_per_100g"], 1)
    elif goal == "bulking":
        return food["calories_per_100g"] / 100
    return 1

def assign_percentage_scores(candidate_foods, goal):
    """Har food ka contribution percentage assign kare."""
    scores = [compute_fit_score(food, goal) for food in candidate_foods]
    total = sum(scores)
    percentages = [score / total for score in scores] if total else [1 / len(candidate_foods)] * len(candidate_foods)
    for food, pct in zip(candidate_foods, percentages):
        food["assigned_pct"] = pct
    return candidate_foods

def calculate_serving_size(food, meal_cal_target):
    """Serving size calculate kare based on assigned percentage."""
    return (food["assigned_pct"] * meal_cal_target / food["calories_per_100g"]) * 100

def generate_meal_plan(goal, total_cal, num_meals, dataset):
    """User ke goal & calorie requirement ke hisaab se personalized meal plan generate kare."""
    meal_plan = []
    meal_cal_target = total_cal / num_meals  

    for meal_num in range(1, num_meals + 1):
        num_items = random.randint(2, 5)  
        candidates = get_candidate_foods(goal, dataset)

        if len(candidates) < num_items:
            num_items = len(candidates)

        selected_items = random.sample(candidates, num_items)
        selected_items = assign_percentage_scores(selected_items, goal)

        meal_items = []
        total_protein = total_carbs = total_fats = total_calories = 0

        for food in selected_items:
            serving = calculate_serving_size(food, meal_cal_target)
            protein = round(food["protein_g"] * (serving / 100), 1)
            carbs = round(food["carbohydrates_g"] * (serving / 100), 1)
            fats = round(food["fats_g"] * (serving / 100), 1)
            calories = round(food["calories_per_100g"] * (serving / 100), 1)

            total_protein += protein
            total_carbs += carbs
            total_fats += fats
            total_calories += calories

            meal_items.append({
                "food_name": food["name"],
                "category": food["category"],
                "serving_size_g": round(serving, 1),
                "calories": calories,
                "protein_g": protein,
                "carbs_g": carbs,
                "fats_g": fats
            })

        meal_plan.append({
            "meal_name": f"Meal {meal_num}",
            "items": meal_items,
            "total_protein": round(total_protein, 1),
            "total_carbs": round(total_carbs, 1),
            "total_fats": round(total_fats, 1),
            "total_calories": round(total_calories, 1)
        })

    return meal_plan

# ✅ Step 3: Feedback System
def reward_function(user_feedback):
    """User feedback ke basis pe reward assign kare."""
    rewards = {"love": 5, "like": 2, "neutral": 0, "dislike": -2, "hate": -5}
    return rewards.get(user_feedback, 0)

def update_meal_policy(meal_plan, feedback_list, dataset, goal, total_cal, num_meals):
    """RL-based meal policy update with food exclusion"""
    disliked_foods = set()
    
    # First pass: Collect disliked foods
    for meal, feedback in zip(meal_plan, feedback_list):
        if feedback in ["dislike", "hate"]:
            disliked_foods.update(item["food_name"] for item in meal["items"])

    # Second pass: Regenerate meals with exclusions
    for i, (meal, feedback) in enumerate(zip(meal_plan, feedback_list)):
        reward = reward_function(feedback)
        
        if reward < 1:
            print(f"🔄 Updating {meal['meal_name']} | Removing disliked foods: {list(disliked_foods)}")
            
            # Create filtered dataset excluding disliked foods
            filtered_dataset = [food for food in dataset if food["name"] not in disliked_foods]
            
            # Regenerate meal with filtered dataset
            try:
                new_meal_plan = generate_meal_plan(goal, total_cal, 1, filtered_dataset)
                if new_meal_plan:
                    new_meal = new_meal_plan[0]
                    new_meal["meal_name"] = meal["meal_name"]  # Keep the original meal name
                    meal_plan[i] = new_meal
            except ValueError:
                print(f"⚠️ Couldn't regenerate {meal['meal_name']} - not enough alternatives available")

    return meal_plan

# ✅ Step 4: Run the Meal Planner
if __name__ == "__main__":
    user_goal = input("Enter Your Goal (muscle_gain / fat_loss / bulking): ").strip().lower()
    daily_calories = int(input("Enter Your Daily Calorie Requirement: "))
    meals_per_day = int(input("How Many Meals Do You Want Per Day? "))

    # ✅ Generate Initial Meal Plan
    meal_plan = generate_meal_plan(user_goal, daily_calories, meals_per_day, dataset)

    print("\n🍽️ **Generated AI Meal Plan:**")
    for meal in meal_plan:
        print(f"🔹 {meal['meal_name']}:")
        for item in meal["items"]:
            print(f"  ✅ {item['food_name']} | {item['serving_size_g']}g, {item['calories']} kcal, Protein: {item['protein_g']}g, Carbs: {item['carbs_g']}g, Fats: {item['fats_g']}g")
        print(f"   🔥 **Meal Total**: {meal['total_calories']} kcal, Protein: {meal['total_protein']}g, Carbs: {meal['total_carbs']}g, Fats: {meal['total_fats']}g\n")

    # ✅ Collect Feedback
    print("\n💬 Give Feedback for Meals (love / like / neutral / dislike / hate)")
    feedback_list = []
    for meal in meal_plan:
        feedback = input(f"How was {meal['meal_name']}? ").strip().lower()
        feedback_list.append(feedback)

    # ✅ Generate Optimized Meal Plan based on feedback
    optimized_meal_plan = update_meal_policy(meal_plan, feedback_list, dataset, user_goal, daily_calories, meals_per_day)

    # ✅ Print Final Optimized Meal Plan
    print("\n🍽️ **Final Optimized AI Meal Plan (After Feedback):**")
    for meal in optimized_meal_plan:
        print(f"🔹 {meal['meal_name']}:")
        for item in meal["items"]:
            print(f"  ✅ {item['food_name']} | {item['serving_size_g']}g, {item['calories']} kcal, Protein: {item['protein_g']}g, Carbs: {item['carbs_g']}g, Fats: {item['fats_g']}g")
        print(f"   🔥 **Meal Total**: {meal['total_calories']} kcal, Protein: {meal['total_protein']}g, Carbs: {meal['total_carbs']}g, Fats: {meal['total_fats']}g\n")

    # ✅ Calculate and Display Daily Totals
    total_calories_day = 0.0
    total_protein_day = 0.0
    total_carbs_day = 0.0
    total_fats_day = 0.0

    for meal in optimized_meal_plan:
        total_calories_day += meal["total_calories"]
        total_protein_day += meal["total_protein"]
        total_carbs_day += meal["total_carbs"]
        total_fats_day += meal["total_fats"]

    print("\n📊 **Daily Nutrition Summary**:")
    print(f"🔥 Total Calories: {round(total_calories_day, 1)} kcal")
    print(f"💪 Total Protein: {round(total_protein_day, 1)}g")
    print(f"🍞 Total Carbohydrates: {round(total_carbs_day, 1)}g")
    print(f"🥑 Total Fats: {round(total_fats_day, 1)}g")
    
    # Calculate and print simplified macronutrient ratio
    p_int = round(total_protein_day)
    c_int = round(total_carbs_day)
    f_int = round(total_fats_day)
    gcd_val = math.gcd(math.gcd(p_int, c_int), f_int)
    if gcd_val == 0:
        simplified_ratio = f"{p_int}:{c_int}:{f_int}"
    else:
        simplified_ratio = f"{p_int // gcd_val}:{c_int // gcd_val}:{f_int // gcd_val}"
    
    print(f"⚖️  Macronutrient Ratio (P:C:F): {round(total_protein_day, 1)}g:{round(total_carbs_day, 1)}g:{round(total_fats_day, 1)}g")
    print(f"⚖️  Simplified Macronutrient Ratio (P:C:F): {simplified_ratio}")