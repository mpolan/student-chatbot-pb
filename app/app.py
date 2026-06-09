from app.intent import detect_intent
from app.tfidf_search import search
from app.llm_answer import generate_answer, rewrite_query


def print_sources(results):
    print("\n[Źródła z bazy danych]:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   {result['url']}")
        print(f"   score={result['score']:.4f}")


def main():
    print("Student Assistant CLI")
    
    chat_history = [] 

    while True:
        query = input("\nTy: ").strip()

        if query.lower() == "q":
            break

        search_query = query
        if chat_history:
            search_query = rewrite_query(query, chat_history)
        intent = detect_intent(search_query)
        print(f"\nWykryta intencja: {intent}")

        if intent in ["powitanie", "chitchat"]:
            results = []
        else:
            results = search(search_query, k=5)
            if "podyplomow" not in search_query.lower():
                results = [r for r in results if "podyplomowe" not in r["title"].lower() and "podyplomowe" not in r["url"].lower()]
            results = results[:3]
            results = [r for r in results if r["score"] > 0.12]

        answer = generate_answer(query, results)

        print("\nBot:")
        print(answer)

        if results:
            print_sources(results)

        chat_history.append({"user": query, "bot": answer})
        
        if len(chat_history) > 10:
            chat_history.pop(0)


if __name__ == "__main__":
    main()