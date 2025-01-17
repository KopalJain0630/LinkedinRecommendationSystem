import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Step 1: Load JSON Data
def load_profiles(folder_path):
    profiles = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            user_id = filename[:-5]  # Extract user_id from filename
            with open(os.path.join(folder_path, filename), "r") as file:
                profile = json.load(file)
                profile["id"] = user_id  # Add user_id to profile
                profiles.append(profile)
    return profiles


# Step 2: Process Profiles into Feature Text
def process_field(field):
    if isinstance(field, list):
        processed_items = []
        for item in field:
            if isinstance(item, dict):
                processed_items.append(" ".join(str(value) for value in item.values()))
            elif isinstance(item, str):
                processed_items.append(item)
        return " ".join(processed_items)
    return str(field) if isinstance(field, str) else ""


def extract_features(profiles):
    user_ids = []
    user_features = []
    for profile in profiles:
        user_ids.append(profile["id"])

        # Combine fields into a single text feature
        skills_text = process_field(profile.get("skills", []))
        experience_text = process_field(profile.get("experience", []))
        education_text = process_field(profile.get("education", []))
        feature_text = f"{skills_text} {experience_text} {education_text}"
        user_features.append(feature_text)
        
    return user_ids, user_features


# Step 3: Compute Recommendations for a Single User with Similarity Scores
def recommend_single_user(target_user_id, user_ids, user_features, top_k=5):
    # Convert text to TF-IDF embeddings
    vectorizer = TfidfVectorizer(max_features=300)
    embeddings = vectorizer.fit_transform(user_features).toarray()

    # Find the index of the target user
    if target_user_id not in user_ids:
        raise ValueError(f"User ID {target_user_id} not found in the dataset.")
    target_idx = user_ids.index(target_user_id)

    # Compute cosine similarity
    similarity_scores = cosine_similarity([embeddings[target_idx]], embeddings).flatten()

    # Get the top_k most similar users (excluding the target user itself)
    similar_indices = similarity_scores.argsort()[-top_k - 1 : -1][::-1]
    recommendations = [
        {"user_id": user_ids[i], "similarity": similarity_scores[i]} for i in similar_indices
    ]

    return recommendations


# Main Pipeline
if __name__ == "__main__":
    folder_path = "final_user_profiles"  # Replace with your JSON folder path
    target_user_id = "ravi-kiran-47284928"  # Replace with the target user ID

    profiles = load_profiles(folder_path)

    # Extract features
    user_ids, user_features = extract_features(profiles)

    # Generate recommendations for the target user
    recommendations = recommend_single_user(target_user_id, user_ids, user_features)

    # Display recommendations with similarity scores
    print(f"Recommended users for {target_user_id}:")
    for rec in recommendations:
        print(f"User ID: {rec['user_id']}, Similarity: {rec['similarity']:.4f}")
