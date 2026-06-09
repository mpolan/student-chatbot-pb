import ollama

MODEL_NAME = "llama3.2"


def get_relevant_chunk(text, query, window=3000):
    """
    Funkcja przeszukuje dokument i wycina fragment wokół kluczowych słów.
    Dzięki temu eliminujemy nagłówki i menu, podając LLM samą treść/tabelę.
    """
    text_lower = text.lower()
    # Szukamy słów kluczowych z zapytania (odrzucamy bardzo krótkie słowa)
    keywords = [w for w in query.lower().split() if len(w) > 3]
    
    start_idx = 0
    for kw in keywords:
        idx = text_lower.find(kw)
        if idx != -1:
            start_idx = max(0, idx - 200)
            break
            
    return text[start_idx : start_idx + window]


def build_context(results, query):
    context_parts = []
    for i, result in enumerate(results, start=1):
        text = get_relevant_chunk(result["text"], query, window=3000)
        context_parts.append(
            f"""
DOKUMENT {i}
Tytuł: {result["title"]}
Treść:
{text}
"""
        )
    return "\n".join(context_parts)


def rewrite_query(current_query, chat_history):
    """
    Decyduje, czy zapytanie wymaga rozwinięcia. 
    Jeśli to gotowe zdanie, zwraca je bez zmian.
    """
    if not chat_history:
        return current_query

    history_text = ""
    for turn in chat_history:
        history_text += f"Użytkownik: {turn['user']}\nBot: {turn['bot']}\n"

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś programem optymalizującym zapytania do wyszukiwarki tekstowej.\n"
                "Twoim jedynym zadaniem jest ocena najnowszego zapytania użytkownika w kontekście historii rozmowy.\n\n"
                
                "ZASADY:\n"
                "1. Jeśli najnowsze zapytanie zawiera już konkretną nazwę kierunku lub zagadnienia (np. 'Jaki jest próg wejściowy na informatyke?'), "
                "jest ono w pełni zrozumiałe samodzielnie. Wtedy NIE ZMIENIAJ GO ANI O JEDNO SŁOWO. Zwróć je dokładnie tak, jak je wpisano.\n"
                "2. Jeśli zapytanie jest skrótem myślowym i nawiązaniem do historii (np. 'A na cyberbezpieczeństwo?', 'A jakie wyposażenie tam jest?'), "
                "wtedy przekształć je w pełne, samodzielne zdanie pytające (np. 'Jaki jest próg wejściowy na cyberbezpieczeństwo?').\n\n"
                
                "ZASADA KRYTYCZNA: Odpowiedz WYŁĄCZNIE przetworzonym zapytaniem. Nie dodawaj żadnych wyjaśnień, komentarzy ani cudzysłowów."
            )
        },
        {
            "role": "user",
            "content": f"""
Historia rozmowy:
{history_text}

Najnowsze zapytanie użytkownika:
{current_query}

Zoptymalizowane zapytanie:
"""
        }
    ]

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": 0.0} # Pełny determinizm
    )
    
    return response.message.content.strip()


def generate_answer(question, results):
    if not results:
        return "Nie znalazłem dokładnej informacji w bazie wiedzy."

    context = build_context(results, question)

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś oficjalnym chatbotem Politechniki Białostockiej o nazwie Student Assistant. "
                "Odpowiadasz po polsku, krótko, uprzejmie i konkretnie.\n\n"
                
                "ZASADY ODPOWIADANIA:\n"
                "1. Odpowiadasz wyłącznie na podstawie podanego kontekstu.\n"
                "2. W kontekście dane mogą być podane w formie skrótowej lub tabelarycznej (np. 'informatyka: 140'). "
                "Musisz poprawnie odczytać to powiązanie jako próg punktowy (np. próg na informatykę to 140 punktów).\n"
                "3. Nie wymyślaj informacji, liczb ani terminów, których nie ma fizycznie w tekście.\n"
                "4. Jeżeli w podanym tekście nie ma informacji o szukanym kierunku, napisz: 'Nie znalazłem dokładnej informacji w bazie wiedzy.'\n"
                "5. Nie dodawaj sekcji 'Źródła'."
            )
        },
        {
            "role": "user",
            "content": f"""
Kontekst:
{context}

Pytanie użytkownika:
{question}
"""
        }
    ]

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.1,
            "top_p": 0.9
        }
    )

    return response.message.content