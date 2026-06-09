import spacy

INTENTS = {
    "rekrutacja": [
        "rekrutacja",
        "kandydat",
        "matura",
        "nabór",
        "terminy",
        "próg",
        "progi",
    ],
    "studia": ["kierunek", "studia", "informatyka", "program", "przedmioty"],
    "kontakt": ["kontakt", "dziekanat", "email", "telefon", "adres"],
    "stypendia": [
        "stypendium",
        "stypendia",
        "socjalne",
        "naukowe",
        "zapomoga",
        "świadczenie",
        "świadczenia",
    ],
    "sprawy_studenckie": [
        "legitymacja",
        "usos",
        "praktyka",
        "praktyki",
        "tok",
        "opłata",
        "opłaty",
        "akademicki",
        "regulamin",
        "kredyt",
    ],
    "akademik": [
        "akademik",
        "akademika",
        "akademiki",
        "akademikach",
        "dom",
        "pokój",
        "zakwaterowanie",
        "zakwaterować",
    ],
}

INTENT_PHRASES = {
    "rekrutacja": {
        "opłata rekrutacyjna",
        "dokumenty rekrutacyjne",
        "limity miejsc",
        "ile miejsc",
        "wzór rekrutacyjny",
    },
    "stypendia": {
        "sprawie stypendium",
        "świadczenia dla studentów",
    },
    "sprawy_studenckie": {
        "dział spraw studenckich",
        "organizacji kształcenia",
        "organizacja kształcenia",
    },
}

SPECIFIC_INTENTS = {
    "rekrutacja": {"próg", "progi"},
    "sprawy_studenckie": {
        "legitymacja",
        "usos",
        "praktyka",
        "praktyki",
        "akademicki",
        "regulamin",
        "kredyt",
    },
}

SMALL_TALK = {
    "powitanie": {
        "cześć",
        "czesc",
        "hej",
        "hejka",
        "dzień dobry",
        "dzien dobry",
        "dobry wieczór",
        "dobry wieczor",
        "witam",
    },
    "chitchat": {
        "dziękuję",
        "dziekuje",
        "dzięki",
        "dzieki",
        "do widzenia",
        "dobranoc",
        "miłego dnia",
        "milego dnia",
    },
}

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
    normalized_text = " ".join(
        token.text.lower()
        for token in nlp(text)
        if token.is_alpha
    )

    for intent, phrases in INTENT_PHRASES.items():
        if any(phrase in normalized_text for phrase in phrases):
            return intent

    terms = get_terms(text)
    scores = {}

    for intent, keywords in SPECIFIC_INTENTS.items():
        if terms & keywords:
            return intent

    for intent, keywords in INTENTS.items():
        score = 0

        for keyword in keywords:
            if keyword in terms:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] > 0:
        return best_intent

    for intent, phrases in SMALL_TALK.items():
        if normalized_text in phrases:
            return intent

    return "nieznana"
