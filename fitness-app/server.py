from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import importlib.util
import sys
from functools import lru_cache

app = Flask(__name__, static_folder=".")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# Load meal planner
meal_planner_path = os.path.join(PARENT_DIR, "Final_Dataset", "ai_meal_planner.py")
spec1 = importlib.util.spec_from_file_location("ai_meal_planner", meal_planner_path)
meal_planner = importlib.util.module_from_spec(spec1)
sys.modules["ai_meal_planner"] = meal_planner
spec1.loader.exec_module(meal_planner)

# Load exercise trainer
exercise_path = os.path.join(BASE_DIR, "ExerciseAiTrainer_2.py")
spec2 = importlib.util.spec_from_file_location("ExerciseAiTrainer", exercise_path)
exercise_trainer = importlib.util.module_from_spec(spec2)
sys.modules["ExerciseAiTrainer"] = exercise_trainer
spec2.loader.exec_module(exercise_trainer)

@lru_cache(maxsize=1)
def create_exercise_processor():
    return exercise_trainer.create_exercise_processor()

processor = create_exercise_processor()

print("✅ ExerciseAiTrainer module loaded successfully!")
print(f"Available functions: {dir(exercise_trainer)}")

# Load meal dataset
dataset_path = os.path.join(PARENT_DIR, "final_diet_plan_dataset.json")
with open(dataset_path, "r") as f:
    meal_dataset = json.load(f)


@app.route("/generate_meal_plan", methods=["POST"])
def generate_meal_plan():
    data = request.get_json()
    try:
        meal_plan = meal_planner.generate_meal_plan(
            data.get("goal"),
            int(data.get("dailyCalories")),
            int(data.get("mealsPerDay")),
            meal_dataset
        )
        return jsonify({"status": "success", "meal_plan": meal_plan})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/classify_exercise", methods=["POST"])
def classify_exercise():
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request must be JSON",
            "error_type": "INVALID_REQUEST"
        }), 400

    data = request.get_json()
    frame_data = data.get("frame")
    if not frame_data:
        return jsonify({
            "status": "error",
            "message": "Missing 'frame' in request",
            "error_type": "INVALID_DATA"
        }), 400

    try:
        result = processor.process_frame(frame_data)
        
        landmarks = result.get("landmarks", [])
        max_idx = max(idx.value for idx in exercise_trainer.LANDMARK_INDICES.values())
        if len(landmarks) <= max_idx:
            # no full skeleton yet
            return jsonify({ "status": "success", "result": { "error": "No pose detected yet" } })

        # If processor returned an error key, forward it
        if "error" in result:
            return jsonify({"status": "success", "result": result})

        landmarks = result.get("landmarks", [])
        angles = result.get("angles", [])
        exercise_type = str(result.get("exercise_type", "unknown"))

        # Build landmarks dict using correct indices (normalized 0–1 coords)
        landmarks_dict = {
            name: {
                "x": float(landmarks[exercise_trainer.LANDMARK_INDICES[name].value][0]),
                "y": float(landmarks[exercise_trainer.LANDMARK_INDICES[name].value][1]),
                "z": float(landmarks[exercise_trainer.LANDMARK_INDICES[name].value][2])
            }
            for name in exercise_trainer.LANDMARK_INDICES.keys()
        }

        return jsonify({
            "status": "success",
            "result": {
                "exercise_type": exercise_type,
                "form_score": int(result.get("form_score", 0)),
                "counters": {k: int(v) for k, v in result.get("counters", {}).items()},
                "counter": int(result.get("counters", {}).get(exercise_type.lower(), 0)),
                "feedback": result.get("feedback", []),
                "stage": str(result.get("stage", "")),
                "angle_thresholds": result.get("angle_thresholds", {}),
                "landmarks": landmarks_dict,
                "angles": [float(a) for a in angles]
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "error_type": "SERVER_ERROR"
        }), 500


@app.route("/exercise.html")
def serve_exercise():
    return send_from_directory(BASE_DIR, "exercise.html")


@app.route("/exercise_2.html")
def serve_exercise_2():
    return send_from_directory(BASE_DIR, "exercise_2.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)