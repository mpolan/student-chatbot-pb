import ollama

MODEL_NAME = "llama3.2"


def build_context(results, query):
    """
    Zamiast wycinać okno wokół słowa kluczowego (co często trafiało w nagłówki),
    dajemy LLM CAŁY tekst chunku — chunki z TF-IDF są już odpowiednio małe.
    Dodajemy też URL bezpośrednio w kontekście żeby LLM mógł go cytować.
    """
    context_parts = []
    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"DOKUMENT {i}\n"
            f"Tytuł: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Treść:\n{result['text']}\n"
        )
    return "\n---\n".join(context_parts)


def rewrite_query(current_query, chat_history):
    """
    Przepisuje zapytanie tylko jeśli jest ewidentnym skrótem myślowym
    (bardzo krótkie + brak rzeczownika). W pozostałych przypadkach zwraca oryginał.
    Zamiast LLM używamy prostej heurystyki — LLM tu bardziej szkodził niż pomagał.
    """
    words = current_query.strip().split()

    # Krótkie zapytania które zaczynają się od "a ", "i ", "to " 
    # to prawie zawsze follow-up do poprzedniego pytania
    followup_starts = ("a ", "i co", "to co", "a co", "i jak", "a jak", "to jak")
    is_short = len(words) <= 4
    is_followup = any(current_query.lower().startswith(f) for f in followup_starts)

    if not (is_short and is_followup):
        return current_query

    # Mamy follow-up — doklejamy kontekst z ostatniej tury
    if not chat_history:
        return current_query

    last_turn = chat_history[-1]
    previous_query = last_turn.get("search_query", last_turn["user"])

    return f"{previous_query} {current_query}"


def generate_answer(question, results):
    if not results:
        return (
            "Nie znalazłem informacji na ten temat w bazie wiedzy PB. "
            "Spróbuj zajrzeć bezpośrednio na pb.edu.pl lub skontaktuj się z dziekanatem."
        )

    context = build_context(results, question)

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś oficjalnym chatbotem Politechniki Białostockiej o nazwie Student Assistant.\n"
                "Odpowiadasz po polsku, konkretnie i krótko.\n\n"
                "ZASADY:\n"
                "1. Odpowiadasz wyłącznie na podstawie dostarczonych dokumentów.\n"
                "2. Jeżeli dokument zawiera dane w formie listy lub tabeli, odczytaj dokładnie powiązanie nazwa → wartość.\n"
                "3. Nie wymyślaj liczb, dat, nazw ani adresów, których nie ma w dokumentach.\n"
                "4. Jeśli informacji nie ma w dokumentach, napisz dokładnie: 'Nie znalazłem tej informacji w bazie wiedzy.'\n"
                "5. Nie dodawaj sekcji 'Źródła', URL-i ani komentarzy typu 'Oto odpowiedź'.\n"
                "6. Odpowiedź ma mieć maksymalnie 3 zdania."
            ),
        },
        {
            "role": "user",
            "content": f"Dokumenty:\n{context}\n\nPytanie: {question}",
        },
    ]

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.0,  # pełny determinizm — mniej halucynacji
            "top_p": 0.9,
        },
    )

    return response.message.content