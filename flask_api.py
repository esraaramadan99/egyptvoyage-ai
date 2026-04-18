from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("recommendation_model.pkl")
encoders = joblib.load("label_encoders.pkl")

interaction_weights = {'review': 3, 'favorite': 2, 'view': 1}

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_id = str(data["user_id"])
    entity_id = str(data["entity_id"])
    interaction_type = data.get("interaction_type", "view")

    try:
        user = encoders['user_id'].transform([user_id])[0]
    except:
        user = -1  # Unknown user

    try:
        entity = encoders['entity_id'].transform([entity_id])[0]
    except:
        entity = -1  # Unknown entity

    itype = encoders['interaction_type'].transform([interaction_type])[0]
    weight = interaction_weights.get(interaction_type, 0)

    df = pd.DataFrame([[user, entity, weight]], columns=['user_id', 'entity_id', 'interaction_weight'])

    pred = model.predict(df)[0]

    return jsonify({"prediction": float(pred)})



@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_id = str(data["user_id"])
    entity_ids = data["entity_ids"]          
    interaction_type = data.get("interaction_type", "view")
    top_n = data.get("top_n", None)          

   
    try:
        user = encoders['user_id'].transform([user_id])[0]
    except:
        user = -1  

    weight = interaction_weights.get(interaction_type, 1)
    results = []

    for eid in entity_ids:
        try:
            entity = encoders['entity_id'].transform([str(eid)])[0]
        except:
    
            continue

        df = pd.DataFrame(
            [[user, entity, weight]],
            columns=['user_id', 'entity_id', 'interaction_weight']
        )
        score = model.predict(df)[0]
        results.append({"entity_id": eid, "score": float(score)})

   
    results.sort(key=lambda x: x["score"], reverse=True)

    
    if top_n:
        results = results[:int(top_n)]

    return jsonify({"recommendations": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)