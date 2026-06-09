import ollama

MODEL_NAME = "llama3.2"


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        text = result["text"][:4000]

        context_parts.append(
            f"""
DOKUMENT {i}
Tytuł: {result["title"]}
Treść:
{text}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, results):
    if not results:
        return "Nie znalazłem dokładnej informacji w bazie wiedzy."

    context = build_context(results)

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś chatbotem Politechniki Białostockiej o nazwie Student Assistant. "
                "Odpowiadasz po polsku, krótko, uprzejmie i konkretnie.\n\n"
                
                "ZASADY:\n"
                "1. Odpowiadasz wyłącznie na podstawie podanego kontekstu.\n"
                "2. W kontekście dane mogą być zapisane w formacie tabelarycznym lub skrótowym (np. 'kierunek: liczba'). "
                "Twoim zadaniem jest poprawne odczytanie tych powiązań i podanie ich użytkownikowi "
                "(np. jeśli widzisz 'informatyka: 140', oznacza to, że próg na informatykę wynosi 140 punktów).\n"
                "3. Pod żadnym pozorem nie wymyślaj liczb ani dat, których NIE MA fizycznie w tekście.\n"
                "4. Jeżeli w kontekście naprawdę nie ma żadnej wzmianki o szukanym kierunku lub temacie, "
                "napisz: 'Nie znalazłem dokładnej informacji w bazie wiedzy.'\n"
                "5. Nie dodawaj sekcji 'Źródła'."
            )
        },
        {
            "role": "user",
            "content": f"""
Pytanie użytkownika:
{question}

Kontekst:
{context}

Odpowiedz na pytanie użytkownika.
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