from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import random


def generate_mcqs(text, num_mcqs=5):
    # Fallback to predefined questions if generation fails
    default_mcqs = [
        {
            "question": "What is the main topic of this document?",
            "options": ["Topic A", "Topic B", "Topic C", "Topic D"],
            "correct_answer": "Topic A",
        }
    ]

    try:
        # Extract key terms using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()

        mcqs = []
        for _ in range(num_mcqs):
            # Select random key term
            key_term = random.choice(feature_names) if feature_names else "key concept"

            # Generate question
            question = f"What best describes '{key_term}'?"

            # Generate options
            options = [
                f"A detailed explanation about {key_term}",
                f"A brief overview of {key_term}",
                f"An alternative perspective on {key_term}",
                f"A historical context of {key_term}",
            ]

            random.shuffle(options)

            mcqs.append(
                {"question": question, "options": options, "correct_answer": options[0]}
            )

        return mcqs

    except Exception as e:
        print(f"MCQ generation error: {e}")
        return default_mcqs
