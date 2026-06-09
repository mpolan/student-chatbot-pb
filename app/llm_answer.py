import ollama

MODEL_NAME = "llama3.2"


def build_context(results):
    context_parts = []
    for i, result in enumerate(results, start=1):
        text = result["text"][:1500]
        context_parts.append(
            f"""
ŹRÓDŁO {i}
Tytuł: {result["title"]}
URL: {result["url"]}
Treść:
{text}
"""
        )
    return "\n".join(context_parts)


def generate_answer(question, results):
    # USUNĘLIŚMY: 'if not results: return ...' 
    # Pozwalamy Ollamie zareagować na puste wyniki (np. przy powitaniu)

    context = build_context(results) if results else "BRAK DANYCH W BAZIE WIEDZY"

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś chatbotem Politechniki Białostockiej o nazwie Student Assistant. "
                "Odpowiadasz po polsku, krótko, uprzejmie i konkretnie. "
                
                # Kluczowa zmiana: instrukcja dla chitchatu
                "Jeżeli użytkownik po prostu się wita, prowadzi luźną rozmowę (chitchat) lub pisze wiadomości grzecznościowe, "
                "odpowiedz mu naturalnie i sympatycznie od siebie, nie wspominając nic o bazie wiedzy ani źródłach. "
                
                # Instrukcja dla pytań o uczelnię
                "Jeśli użytkownik zadaje konkretne pytanie o uczelnię, studia lub procedury, korzystaj wyłącznie z podanego kontekstu. "
                "Nie wymyślaj informacji spoza kontekstu. Jeżeli w kontekście nie ma odpowiedzi na konkretne pytanie, "
                "napisz, że nie znalazłeś tej informacji w bazie wiedzy. "
                "Gdy odpowiadasz na podstawie kontekstu, na samym końcu odpowiedzi dodaj sekcję 'Źródła' z linkami z kontekstu."
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
        messages=messages
    )

    return response.message.content