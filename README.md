# Student Assistant PB

Prosty chatbot odpowiadający na pytania dotyczące Politechniki Białostockiej. Projekt korzysta z klasyfikacji intencji, wyszukiwania TF-IDF oraz lokalnego modelu językowego uruchamianego przez Ollamę.

## Uruchomienie

Wymagany jest Python 3.10 oraz zainstalowana [Ollama](https://ollama.com/).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
python -m app.app
```

Program działa w terminalu. Aby go zamknąć, wpisz `q`.

## Jak to działa

1. Pytanie użytkownika jest przypisywane do jednej z podstawowych intencji.
2. W indeksie TF-IDF wyszukiwane są pasujące fragmenty stron PB.
3. Znalezione fragmenty są przekazywane jako kontekst do modelu `llama3.2`.
4. Model generuje odpowiedź i podaje wykorzystane źródła.

## Pliki

- `app/` - kod chatbota,
- `data/` - pobrane strony oraz gotowy indeks TF-IDF,
- `scrape.ipynb` - pobieranie danych ze stron PB,
- `tf_idf.ipynb` - budowa indeksu,
- `evaluate.ipynb` - podstawowa ewaluacja intencji i wyszukiwania.

Gotowy indeks znajduje się w repozytorium, więc do zwykłego uruchomienia aplikacji nie trzeba wykonywać notebooków.
