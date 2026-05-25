import spacy
import json
from pprint import pprint

INTENTS = {
    "rekrutacja": ["rekrutacja", "kandydat", "matura", "nabór", "terminy"],
    "studia": ["kierunek", "studia", "informatyka", "program", "przedmioty"],
    "kontakt": ["kontakt", "dziekanat", "email", "telefon", "adres"],
    "stypendia": ["stypendium", "socjalne", "naukowe", "zapomoga"],
    "akademik": ["akademik", "akademika", "akademiki", "akademikach", "dom", "student", "pokój", "zakwaterowanie"]
}

nlp = spacy.load("pl_core_news_sm")

def detect_intent(text):
    terms = get_terms(text)

    # print(f"\tDEBUG (TERMS): {terms}")

    scores = {}

    for intent, keywords in INTENTS.items():
        score = 0

        for keyword in keywords:
            if keyword in terms:
                score += 1

        scores[intent] = score

    # print(f"\tDEBUG (SCORES): {scores}")

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "nieznana"

    return best_intent

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

def search_knowledge_base(text):
    with open("data/knowledge_base.json", "r", encoding="utf-8") as file:
        base = json.load(file)

    query_terms = get_terms(text)

    results = []

    for item in base:
        item_tags = set(item.get('intents', []))
        item_text = f"{item.get('title', '')} {item.get('text', '')}"
        item_terms = get_terms(item_text)

        tag_score = len(query_terms & item_tags) * 2
        text_score = len(query_terms & item_terms)

        score = tag_score + text_score

        results.append({
            'score': score,
            'item': item,
        })

    results = sorted(results, key = lambda x: x['score'], reverse=True)

    return results


def main():
    print("Student Assistant CLI")
    print("'q', aby zakończyć.\n")

    while True:
        text = input("Ty: ")

        if text == 'q':
            break
        if text == 'a':
            text = "Gdzie znajdę informację o rekrutacji na studia po maturze?"
        
        intent = detect_intent(text)

        results = search_knowledge_base(text)
        pprint(f"DEBUG (results): {results}")

        if results and results[0]["score"] > 0:
            best = results[0]["item"]

            print(f"""
        Bot:
        Wykryta intencja: {intent}

        Najlepszy wynik:
        Tytuł: {best.get("title")}
        Opis: {best.get("text")}
        Url: {best.get("url")}
        """)
        else:
            print("Bot: Nie znalazłem pasującej informacji.")

if __name__ == "__main__":
    main()