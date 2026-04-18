import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
import random

logging.basicConfig(level=logging.INFO)

client = MongoClient("mongodb+srv://esraaramadan:esraa12345@cluster0.yg2odvv.mongodb.net/?appName=Cluster0")
db = client["EgyptVoyageDb"]

rows = []

logging.info("Fetching favorites...")
for fav in db["FavoriteLists"].find({"isDeleted": False}):
    user_id = str(fav.get("touristId"))
    
    entity_lists = [
        fav.get("hotelIds", []),
        fav.get("restaurantIds", []),
        fav.get("landmarkIds", []),
        fav.get("programIds", [])
    ]
    
    for entity_group in entity_lists:
        for entity in entity_group:
            rows.append({
                "user_id": user_id,
                "entity_id": str(entity),
                "interaction_weight": 1 
            })

logging.info("Fetching reviews...")
for rev in db["Reviews"].find({"isDeleted": False}):
    uid = str(rev.get("touristId"))
    eid = str(rev.get("entityId"))
    
    if uid and eid:
        rows.append({
            "user_id": uid,
            "entity_id": eid,
            "interaction_weight": 1  
        })

if not rows:
    logging.error("No interaction data found in database.")
    exit()

df = pd.DataFrame(rows)

df = df.drop_duplicates(subset=['user_id', 'entity_id'])

logging.info(f"Total positive interactions: {len(df)}")


logging.info("Generating negative samples...")

all_users = df['user_id'].unique()
all_entities = set(df['entity_id'].unique())

positive_interactions = df.groupby('user_id')['entity_id'].apply(set).to_dict()

negative_rows = []
for user in all_users:
    user_positives = positive_interactions.get(user, set())
    
    possible_negatives = list(all_entities - user_positives)
    
    num_positives = len(user_positives)
    if possible_negatives:
        num_negatives = min(len(possible_negatives), num_positives * 2)
        sampled_negatives = random.sample(possible_negatives, num_negatives)
        
        for neg_entity in sampled_negatives:
            negative_rows.append({
                "user_id": user,
                "entity_id": neg_entity,
                "interaction_weight": 0  # وزن صفر لأن مفيش تفاعل
            })

neg_df = pd.DataFrame(negative_rows)


df = pd.concat([df, neg_df], ignore_index=True)

logging.info(f"Total interactions after negatives: {len(df)}")


df['user_id'] = df['user_id'].astype('category')
df['entity_id'] = df['entity_id'].astype('category')


joblib.dump({"users": df['user_id'].cat.categories, "entities": df['entity_id'].cat.categories}, "categories.pkl")


y = df['interaction_weight']


X = df[['user_id', 'entity_id']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


logging.info("Training XGBoost model...")

# لازم tree_method='hist' و enable_categorical=True
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    enable_categorical=True, 
    tree_method='hist'
)

param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logging.info(f"MAE: {mae}")
logging.info(f"MSE: {mse}")
logging.info(f"R2 Score: {r2}")


joblib.dump(best_model, "recommendation_model.pkl")

logging.info("Model saved successfully")

client.close()