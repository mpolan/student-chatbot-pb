from app.intent import detect_intent
from app.tfidf_search import search

def print_results(intent, results):
    print(f"Wykryta intencja: {intent}")

    if not results:
        print("Nie znalazłem pasujących wyników.")
        return

    for i, result in enumerate(results, start=1):
        print(f"\nWynik {i}")
        print(result["title"])
        print(result["url"])
        print(f"Score: {result['score']:.4f}")
        print(result["text"][:500])

def main():
    print("Student Assistant CLI")

    while True:
        query = input("\nTy: ").strip()

        if query.lower() == "q":
            break

        intent = detect_intent(query)
        results = search(query, k=3)
        print_results(intent, results)

if __name__ == "__main__":
    main()