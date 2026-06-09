from app.intent import detect_intent
from app.tfidf_search import search
from app.llm_answer import generate_answer


def print_sources(results):
    print("\nŹródła:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   {result['url']}")
        print(f"   score={result['score']:.4f}")


def main():
    print("Student Assistant CLI")

    while True:
        query = input("\nTy: ").strip()

        if query.lower() == "q":
            break

        intent = detect_intent(query)
        print(f"\nWykryta intencja: {intent}")

        # Tylko small talk omija bazę. Dla nieznanej intencji szukamy globalnie.
        if intent in ["powitanie", "chitchat"]:
            results = []
        else:
            results = search(query, k=3)

        answer = generate_answer(query, results)

        print("\nBot:")
        print(answer)

        # Drukowanie źródeł tylko, jeśli faktycznie coś znaleźliśmy
        if results:
            print_sources(results)


if __name__ == "__main__":
    main()
