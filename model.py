# model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP SIMILARITY 
def get_text_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


# MBTI LOGIC 
mbti_compatibility = {
    ('INTJ', 'ENFP'): 1.0,
    ('ENFP', 'INTJ'): 1.0,
    ('INFJ', 'ENTP'): 0.9,
    ('ENTP', 'INFJ'): 0.9,
}

def get_mbti_score(m1, m2):
    return mbti_compatibility.get((m1, m2), 0.5)


# LOCATION SCORE 
def get_location_score(loc1, loc2):
    return 1.0 if loc1 == loc2 else 0.5


# FINAL SCORE 
def calculate_score(userA, userB, weights):
    text_sim = get_text_similarity(userA['bio'], userB['bio'])
    mbti = get_mbti_score(userA['mbti'], userB['mbti'])
    location = get_location_score(userA['location'], userB['location'])

    total = (
        weights['w1'] * text_sim +
        weights['w2'] * mbti +
        weights['w3'] * location
    )

    return total, {
        'text': text_sim,
        'mbti': mbti,
        'location': location
    }


# FEEDBACK LEARNING 
def update_weights(weights, feedback, features):
    lr = 0.1  # learning rate

    predicted = (
        weights['w1'] * features['text'] +
        weights['w2'] * features['mbti'] +
        weights['w3'] * features['location']
    )

    error = feedback - predicted

    weights['w1'] += lr * error * features['text']
    weights['w2'] += lr * error * features['mbti']
    weights['w3'] += lr * error * features['location']

    return weights