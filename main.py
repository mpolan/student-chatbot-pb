import spacy

INTENTS = {
    "rekrutacja": ["rekrutacja", "kandydat", "matura", "nabór", "terminy"],
    "studia": ["kierunek", "studia", "informatyka", "program", "przedmioty"],
    "kontakt": ["kontakt", "dziekanat", "email", "telefon", "adres"],
    "stypendia": ["stypendium", "socjalne", "naukowe", "zapomoga"],
    "akademik": ["akademik", "dom studenta", "pokój", "zakwaterowanie"],
}

nlp = spacy.load("pl_core_news_sm")

def detect_intent(user_text):
    doc = nlp(user_text.lower())

    lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    print(f"\tDEBUG (LEMMAS): {lemmas}")

    scores = {}

    for intent, keywords in INTENTS.items():
        score = 0
        for lemma in lemmas:
            if lemma in keywords:
                score += 1
        scores[intent] = score

    print(f"\tDEBUG (SCORES): {scores}")

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "nieznana"
    
    return best_intent

def main():
    print("Student Assistant CLI")
    print("Napisz 'exit', aby zakończyć.\n")

    while True:
        text = input("Ty: ")

        if text == 'q':
            break

        intent = detect_intent(text)

        print(f"Bot: {intent}\n")

if __name__ == "__main__":
    main()