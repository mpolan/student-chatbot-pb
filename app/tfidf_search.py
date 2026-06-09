from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from app.intent import get_terms, detect_intent

INDEX_FILE = Path("data/tfidf_index.joblib")
INDEX_DATA = None
MIN_TFIDF_SCORE = 0.03

QUERY_EXPANSIONS = {
    "kontakt": "kontakt telefon email e-mail adres dziekanat dział spraw studenckich",
    "skontaktować": "kontakt telefon email e-mail adres dziekanat rektorat",
    "progi": "progi punktowe punkty minimalne rekrutacja",
    "informatyka": "informatyka wydział informatyki data science",
    "akademik": "akademik akademiki dom studenta zakwaterowanie miejsce w akademiku osiedle akademickie",
    "stypendium": "stypendium stypendia socjalne rektora zapomoga świadczenia student",
    "termin": "terminy harmonogram rekrutacja daty",
    "irk": "internetowa rekrutacja kandydatów zapisy konto",
    "studia": "studia kierunek kierunki program studiów przedmioty sylabus",
    "rekrutacja": "rekrutacja krok po kroku harmonogram dokumenty kandydat studia pierwszego stopnia irk",
    "legitymacja": "legitymacja studencka dokument student",
    "usos": "usos usosweb system student",
    "praktyka": "praktyka praktyki zawodowe student",
    "akademicki": "rok akademicki harmonogram organizacja kształcenia",
    "opłata": "opłata opłaty studia student",
}

IMPORTANT_URL_BOOSTS = {
    "rekrutacja-krok-po-kroku": 0.30,
    "studenci/akademiki-pb": 0.30,
    "studenci/stypendia": 0.25,
    "kontakt/dane-teleadresowe": 0.15,
}

BAD_URL_PENALTIES = {
    "irk.pb.edu.pl": 0.15,
    "irk2.uci.pb.edu.pl": 0.10,
    "kontakt": 0.05,
    "2025/": 0.10,
    "2026/": 0.10,
    "dni-otwarte": 0.10,
    "pralnie": 0.10,
    "redakcja-serwisu-www": 0.10,
    "faq-obsluga-informatyczna": 0.10,
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
    
    tfidf_scores = cosine_similarity(q, selected_matrix).flatten()
    scores = tfidf_scores.copy()

    for i, chunk in enumerate(selected_chunks):
        url = chunk["url"].lower()
        title = chunk["title"].lower()

        for pattern, boost in IMPORTANT_URL_BOOSTS.items():
            if pattern in url:
                scores[i] += boost

        for pattern, penalty in BAD_URL_PENALTIES.items():
            if pattern in url:
                scores[i] -= penalty

        if "krok po kroku" in title:
            scores[i] += 0.10

        if "akademiki pb" in title:
            scores[i] += 0.10

        if "stypendia" in title:
            scores[i] += 0.10

    best = scores.argsort()[::-1]

    results = []
    seen_urls = set()

    for idx in best:
        if tfidf_scores[idx] < MIN_TFIDF_SCORE:
            continue

        chunk = selected_chunks[idx]
        url = chunk["url"].rstrip("/")

        if url in seen_urls:
            continue

        seen_urls.add(url)

        results.append({
            "score": float(scores[idx]),
            "title": chunk["title"],
            "url": url,
            "category": chunk["category"],
            "text": chunk["text"],
        })

        if len(results) >= k:
            break

    return results
