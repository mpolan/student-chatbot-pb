import json
from pathlib import Path

import joblib
import spacy
from sklearn.metrics.pairwise import cosine_similarity


PAGES_FILE = Path("pages.jsonl")
INDEX_FILE = Path("data/tfidf_index.joblib")
INDEX_FILE_DISPLAY = "data/tfidf_index.joblib"
INDEX_DATA = None

INTENTS = {
    "rekrutacja": ["rekrutacja", "kandydat", "matura", "nabór", "terminy"],
    "studia": ["kierunek", "studia", "informatyka", "program", "przedmioty"],
    "kontakt": ["kontakt", "dziekanat", "email", "telefon", "adres"],
    "stypendia": ["stypendium", "socjalne", "naukowe", "zapomoga"],
    "akademik": [
        "akademik",
        "akademika",
        "akademiki",
        "akademikach",
        "dom",
        "student",
        "pokój",
        "zakwaterowanie",
    ],
}

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

NOISE = [
    "Facebook Instagram youtube linkedin tiktok",
    "Strona główna",
    "Aplikuj online",
]

try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    print("Brak modelu spaCy: pl_core_news_sm")
    print("Zainstaluj go komendą:")
    print("python -m spacy download pl_core_news_sm")
    raise SystemExit(1)


def get_terms(text: str) -> set:
    doc = nlp(text.lower())

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    return set(words + lemmas)


def detect_intent(text):
    terms = get_terms(text)
    scores = {}

    for intent, keywords in INTENTS.items():
        score = 0

        for keyword in keywords:
            if keyword in terms:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "nieznana"

    return best_intent


def clean_doc_text(text):
    for noise in NOISE:
        text = text.replace(noise, " ")
    return " ".join(text.split())


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def load_docs(path=PAGES_FILE):
    if not path.exists():
        print(f"Nie znaleziono pliku bazy wiedzy: {path}")
        print("Najpierw uruchom scraper i utwórz pages.jsonl.")
        raise SystemExit(1)

    docs = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    return docs


def build_chunks(docs):
    chunks = []

    for doc in docs:
        text = clean_doc_text(doc.get("text", ""))

        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "chunk_id": i,
                "text": chunk,
            })

    return chunks


def load_index(path=INDEX_FILE):
    if not path.exists():
        print(f"Nie znaleziono indeksu TF-IDF: {INDEX_FILE_DISPLAY}")
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


def main():
    global INDEX_DATA

    INDEX_DATA = load_index()

    print("Student Assistant CLI")

    while True:
        query = input("\nTy: ").strip()

        if query.lower() == "q":
            break

        intent = detect_intent(query)
        results = search(query, k=3)

        print(f"Wykryta intencja: {intent}")

        if not results:
            print("Nie znalazłem pasujących wyników.")
            continue

        for i, result in enumerate(results, start=1):
            fragment = result["text"][:500]

            print(f"\nWynik {i}")
            print(result["title"])
            print(result["url"])
            print(f"Score: {result['score']:.4f}")
            print(fragment)


if __name__ == "__main__":
    main()
