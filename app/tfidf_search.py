from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from app.intent import get_terms, detect_intent

INDEX_FILE = Path("data/tfidf_index.joblib")
INDEX_DATA = None

QUERY_EXPANSIONS = {
    "kontakt": "telefon email e-mail adres dziekanat rektorat",
    "skontaktować": "kontakt telefon email e-mail adres dziekanat rektorat",
    "progi": "progi punktowe punkty minimalne rekrutacja",
    "informatyka": "informatyka wydział informatyki data science",
    "akademik": "akademik dom studenta zakwaterowanie",
    "stypendium": "stypendium socjalne naukowe zapomoga pomoc materialna",
    "termin": "terminy harmonogram rekrutacja daty",
    "irk": "internetowa rekrutacja kandydatów zapisy konto",
}

def load_index(path=INDEX_FILE):
    if not path.exists():
        print(f"Nie znaleziono indeksu TF-IDF")
        print("Uruchom końcową komórkę w tf_idf.ipynb, aby go utworzyć.")
        raise SystemExit(1)

    return joblib.load(path)


def expand_query(query):
    terms = get_terms(query)
    additions = []

    for key, expansion in QUERY_EXPANSIONS.items():
        if key in query.lower() or key in terms:
            additions.append(expansion)

    if additions:
        return query + " " + " ".join(additions)

    return query


def search(query, k=3):
    global INDEX_DATA

    if INDEX_DATA is None:
        INDEX_DATA = load_index()

    intent = detect_intent(query)
    vectorizer = INDEX_DATA["vectorizer"]
    matrix = INDEX_DATA["matrix"]
    chunks = INDEX_DATA["chunks"]

    if intent != "nieznana":
        selected_indices = [
            idx for idx, chunk in enumerate(chunks)
            if chunk.get("category") == intent
        ]
    else:
        selected_indices = []

    if not selected_indices:
        selected_indices = list(range(len(chunks)))

    if not selected_indices:
        return []

    selected_chunks = [chunks[idx] for idx in selected_indices]
    selected_matrix = matrix[selected_indices]
    expanded_query = expand_query(query)

    q = vectorizer.transform([expanded_query])
    
    scores = cosine_similarity(q, selected_matrix).flatten()
    best = scores.argsort()[::-1][:k]

    results = []

    for idx in best:
        chunk = selected_chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "title": chunk["title"],
            "url": chunk["url"],
            "category": chunk["category"],
            "text": chunk["text"],
        })

    return results
