from app.intent import detect_intent
from app.tfidf_search import search
from app.llm_answer import generate_answer, rewrite_query


def print_sources(results):
    print("\n[Źródła z bazy danych]:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   {result['url']}")
        print(f"   score={result['score']:.4f}")


def is_threshold_question(text):
    text = text.lower()
    return any(word in text for word in ["próg", "prog", "progi", "punktowy", "punktowe"])


def filter_results(query, results):
    if "podyplomow" not in query.lower():
        results = [
            r for r in results
            if "podyplomowe" not in r["title"].lower()
            and "podyplomowe" not in r["url"].lower()
        ]

    if is_threshold_question(query):
        threshold_results = [
            r for r in results
            if "progi-punktowe" in r["url"].lower()
            or "progi punktowe" in r["title"].lower()
        ]

        if threshold_results:
            results = threshold_results

    results = results[:3]
    results = [r for r in results if r["score"] > 0.12]

    return results


def main():
    print("Student Assistant CLI")

    chat_history = []
    last_search_query = None

    while True:
        query = input("\nTy: ").strip()

        if query.lower() == "q":
            break

        search_query = query

        if chat_history:
            search_query = rewrite_query(query, chat_history)

        intent = detect_intent(search_query)
        print(f"\nWykryta intencja: {intent}")

        if intent == "powitanie":
            answer = (
                "Cześć! Jestem Student Assistant PB. "
                "Mogę pomóc w sprawach rekrutacji, studiów, kontaktu, stypendiów i akademików."
            )
            results = []

        elif intent == "chitchat":
            answer = "Nie ma sprawy. W czym jeszcze mogę pomóc?"
            results = []

        else:
            results = search(search_query, k=5)
            results = filter_results(search_query, results)

            # Ważne: LLM odpowiada na search_query, nie na surowe query.
            answer = generate_answer(search_query, results)

        print("\nBot:")
        print(answer)

        if results:
            print_sources(results)

        chat_history.append(
            {
                "user": query,
                "search_query": search_query,
                "bot": answer,
            }
        )

        if len(chat_history) > 10:
            chat_history.pop(0)


if __name__ == "__main__":
    main()