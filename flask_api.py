from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone

# ── Load model ──────────────────────────────────────────────
model    = joblib.load("recommendation_model.pkl")
encoders = joblib.load("label_encoders.pkl")

interaction_weights = {'review': 3, 'favorite': 2, 'view': 1}

# ── MongoDB (same connection as trainmodel.py) ───────────────
mongo_client = MongoClient(
    "mongodb+srv://esraaramadan:esraa12345@cluster0.yg2odvv.mongodb.net/?appName=Cluster0"
)
db = mongo_client["EgyptVoyageDb"]

app = Flask(__name__)


# ────────────────────────────────────────────────────────────
#  /predict  — single entity score
# ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data            = request.json
    user_id         = str(data["user_id"])
    entity_id       = str(data["entity_id"])
    interaction_type = data.get("interaction_type", "view")

    try:
        user = encoders['user_id'].transform([user_id])[0]
    except Exception:
        user = -1

    try:
        entity = encoders['entity_id'].transform([entity_id])[0]
    except Exception:
        entity = -1

    weight = interaction_weights.get(interaction_type, 1)
    df     = pd.DataFrame([[user, entity, weight]],
                          columns=['user_id', 'entity_id', 'interaction_weight'])
    pred   = model.predict(df)[0]
    return jsonify({"prediction": float(pred)})


# ────────────────────────────────────────────────────────────
#  /recommend  — ranked list for a user
# ────────────────────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    data            = request.json
    user_id         = str(data["user_id"])
    entity_ids      = data["entity_ids"]
    interaction_type = data.get("interaction_type", "view")
    top_n           = data.get("top_n", None)

    try:
        user = encoders['user_id'].transform([user_id])[0]
    except Exception:
        user = -1

    weight  = interaction_weights.get(interaction_type, 1)
    results = []

    for eid in entity_ids:
        try:
            entity = encoders['entity_id'].transform([str(eid)])[0]
        except Exception:
            results.append({"entity_id": eid, "score": 0.5})
            continue

        df    = pd.DataFrame([[user, entity, weight]],
                             columns=['user_id', 'entity_id', 'interaction_weight'])
        score = model.predict(df)[0]
        results.append({"entity_id": eid, "score": float(score)})

    results.sort(key=lambda x: x["score"], reverse=True)
    if top_n:
        results = results[:int(top_n)]

    return jsonify({"recommendations": results})


# ────────────────────────────────────────────────────────────
#  /trending  — NEW
#  Returns entity IDs ranked by how many UNIQUE users
#  favorited or reviewed them in the last 7 days.
# ────────────────────────────────────────────────────────────
@app.route("/trending", methods=["GET"])
def trending():
    limit      = int(request.args.get("limit", 8))
    days       = int(request.args.get("days", 7))
    since      = datetime.now(timezone.utc) - timedelta(days=days)

    # Count interactions per entity from both collections
    entity_counts: dict[str, dict] = {}   # entityId -> {count, uniqueUsers}

    def add_entry(entity_id: str, tourist_id: str, weight: int):
        if not entity_id or not tourist_id:
            return
        eid = str(entity_id)
        uid = str(tourist_id)
        if eid not in entity_counts:
            entity_counts[eid] = {"count": 0, "unique_users": set()}
        entity_counts[eid]["count"]        += weight
        entity_counts[eid]["unique_users"].add(uid)

    # ── Favorites (last N days) ──────────────────────────────
    # FavoriteLists doesn't have a date per item, so we check
    # the updatedAt field of the document as a proxy.
    for fav in db["FavoriteLists"].find(
        {"isDeleted": False, "updatedAt": {"$gte": since}}
    ):
        tourist_id = fav.get("touristId")
        for eid in fav.get("hotelIds",      []): add_entry(str(eid), tourist_id, 2)
        for eid in fav.get("restaurantIds", []): add_entry(str(eid), tourist_id, 2)
        for eid in fav.get("landmarkIds",   []): add_entry(str(eid), tourist_id, 2)
        for eid in fav.get("programIds",    []): add_entry(str(eid), tourist_id, 2)

    # ── Reviews (last N days) ────────────────────────────────
    for rev in db["Reviews"].find(
        {"isDeleted": False, "createdAt": {"$gte": since}}
    ):
        add_entry(str(rev.get("entityId", "")), str(rev.get("touristId", "")), 3)

    if not entity_counts:
        return jsonify({"trending": []})

    # ── Score = weighted_count × log(unique_users + 1) ──────
    import math
    ranked = []
    for eid, data in entity_counts.items():
        unique = len(data["unique_users"])
        score  = data["count"] * math.log(unique + 1 + 1)
        ranked.append({
            "entity_id":    eid,
            "score":        round(score, 4),
            "unique_users": unique,
            "total_interactions": data["count"],
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"trending": ranked[:limit]})


# ────────────────────────────────────────────────────────────
#  /health
# ────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)