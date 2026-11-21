# 🎓 Szkolny asystent AI – nauczmy model Twoich notatek i regulaminu

**Tutorial dla uczestników warsztatów** · Google Colab · darmowa karta graficzna T4

- `RAG_dla_licealistow.ipynb` - pełny plik warsztatu, z komentarzami i wyjaśnieniami
- `RAG_workshop_CNT.ipynb` - wersja skrócona, mniej komentarzy i wyjaśnień (dobra baza do dalszej pracy)
---

## Co dziś zbudujemy?

Duże modele językowe (ChatGPT, Gemma, Gemini…) wiedzą mnóstwo, ale **nie znają**:

- regulaminu *Twojej* szkoły,
- Twoich notatek z lekcji,
- niczego, co powstało po dniu zakończenia ich „nauki".

Gdy zapytasz o coś, czego nie wiedzą, często **zmyślają** – i robią to bardzo pewnym tonem.

Dziś nauczymy model **Twoich dokumentów** bez żadnego trenowania. Technika nazywa się **RAG**
(*Retrieval-Augmented Generation* – generowanie wzbogacone o wyszukiwanie) i działa jak **egzamin z otwartymi notatkami**:

```
 Pytanie ──► 1. WYSZUKAJ najlepszy fragment w bazie wiedzy
                      │
                      ▼
             2. WKLEJ go do polecenia („odpowiedz na podstawie tego tekstu")
                      │
                      ▼
             3. MODEL ODPOWIADA na podstawie tego, co dostał
```

Warsztat ma trzy części:

| Część | O czym | Czas (orientacyjnie) |
|---|---|---|
| **A** | Budujemy działającego asystenta: pytania o regulamin i notatki | ok. 40 min |
| **B** | Zaglądamy do środka: jak komputer „rozumie" znaczenie zdań (embeddingi) | ok. 30 min |
| **C** | Twój własny asystent – na Twoich notatkach | ok. 20 min |

### Jak czytać ten tutorial

- ⌨️ **Przepisz** – krótki kod, który wpisujesz ręcznie. Przepisując, najwięcej się uczysz! Literówka? Komputer powie Ci, gdzie.
- 📋 **Skopiuj** – długi kod lub dane. Kopiujesz i wklejasz, nie ma sensu tracić na to czasu.
- 👀 **Co powinieneś zobaczyć** – opis oczekiwanego wyniku. Odpowiedzi modelu mogą się różnić słowami – ważny jest sens.
- 🧪 **Twoja kolej** – zadanie do samodzielnego wykonania.

> Każdy blok kodu to **jedna komórka** w Colabie. Uruchamiasz ją klawiszami **Shift + Enter**.
> Komórki uruchamiaj **po kolei** – kolejne korzystają z tego, co zrobiły poprzednie.

---

## Krok 0 – Przygotowanie Colaba

1. Wejdź na **[colab.research.google.com](https://colab.research.google.com)** i zaloguj się kontem Google.
2. Kliknij **Nowy notatnik** (*New notebook*). Nazwij go np. `Szkolny asystent AI`.
3. Włącz kartę graficzną: menu **Środowisko wykonawcze → Zmień typ środowiska wykonawczego → T4 GPU → Zapisz**.
   *(Angielski interfejs: Runtime → Change runtime type → T4 GPU.)*
4. Nowa komórka kodu: przycisk **+ Kod** (*+ Code*) na górze notatnika.

### Token Hugging Face (dostęp do modelu Gemma)

Model Gemma jest darmowy, ale trzeba zaakceptować jego licencję. Zrób to **jednorazowo**:

1. Załóż konto na **[huggingface.co](https://huggingface.co)** (jeśli nie masz).
2. Wejdź na stronę modelu **[huggingface.co/google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)** i kliknij zgodę na licencję.
3. Wygeneruj token: *Settings → Access Tokens → Create new token* (typ: **Read**). Skopiuj go.
4. W Colabie kliknij ikonę **🔑 Secrets** (pasek po lewej) → **Add new secret**:
   - Name: `HF_TOKEN`
   - Value: *Twój token*
   - włącz przełącznik **Notebook access**.

> 🔒 Token to Twoje hasło. Nie wklejaj go do kodu ani na czat, nie pokazuj kolegom. W Secrets jest bezpieczny.
>
> ℹ️ Jeśli prowadzący da Ci inną instrukcję (np. wspólny token), zrób to, co mówi prowadzący.

---

# CZĘŚĆ A – Budujemy asystenta

## Krok 1 – Instalacja bibliotek i sprawdzenie GPU ⌨️

```python
!pip install -q transformers accelerate sentence-transformers

import os
import torch
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

print("GPU:", torch.cuda.get_device_name(0))
```

👀 **Co powinieneś zobaczyć:** po minucie instalacji napis `GPU: Tesla T4`.

> ❗ Jeśli pojawił się błąd o braku sekretu (`SecretNotFoundError`) – nie dodałeś tokenu (patrz Krok 0) albo nie włączyłeś *Notebook access*.
> ❗ Jeśli ostatnia linijka zgłasza błąd o braku GPU/CUDA – włącz T4 w ustawieniach środowiska wykonawczego i uruchom komórkę ponownie.

## Krok 2 – Ładujemy model językowy ⌨️

Mały model **Gemma** od Google (ok. 2 mld parametrów) zmieści się w darmowej karcie T4.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-4-E2B-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model załadowany!")
```

👀 **Co powinieneś zobaczyć:** paski pobierania plików, a po kilku minutach `Model załadowany!`.
To dobry moment, żeby przeczytać Krok 3 i zrobić sobie przerwę ☕

> 💡 **Co tu się dzieje?** *Tokenizer* tnie tekst na małe kawałki (tokeny), które model rozumie. *Model* to sieć neuronowa z miliardami liczb, które „nauczyły się" języka.

## Krok 3 – Funkcja, która pyta model ⌨️

Zamiast za każdym razem pisać wszystko od nowa, zbudujemy funkcję `ask_llm`: dajesz jej tekst, a ona zwraca odpowiedź modelu.

```python
def ask_llm(prompt, max_new_tokens=150):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # odcinamy nasze pytanie, zostawiamy samą odpowiedź
    start = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][start:], skip_special_tokens=True).strip()


print(ask_llm("Podaj stolicę Polski jednym zdaniem."))
```

👀 **Co powinieneś zobaczyć:** coś w stylu `Stolicą Polski jest Warszawa.`

## Krok 4 – Baza wiedzy, której model NIE zna 📋

Poniżej jest **regulamin wymyślonego liceum w Bajtowie**, trochę informacji o szkole i dwie notatki z lekcji.
Ta szkoła nie istnieje, więc żaden model nie mógł jej regulaminu „podejrzeć" w internecie.
Dzięki temu będziemy mieć pewność: jeśli odpowie dobrze, to dlatego, że **dostał od nas wiedzę**.

Każdy element listy to jeden *dokument*.

```python
documents = [
    "Regulamin: uczeń, który spóźni się na lekcję więcej niż 15 minut, otrzymuje w dzienniku nieobecność. Spóźnienia do 15 minut są odnotowywane osobno.",
    "Regulamin: nieobecność trzeba usprawiedliwić w ciągu 7 dni od powrotu do szkoły – przez dziennik elektroniczny lub pisemnie (kartka od rodzica).",
    "Regulamin: w czasie lekcji telefon musi być wyłączony i schowany w plecaku. Nauczyciel może pozwolić na jego użycie do konkretnego zadania.",
    "Regulamin: sprawdzian musi być zapowiedziany z tygodniowym wyprzedzeniem. W jednym dniu mogą odbyć się najwyżej dwa sprawdziany.",
    "Regulamin: ocenę ze sprawdzianu można poprawić tylko jeden raz, w ciągu dwóch tygodni od jej otrzymania.",
    "Biblioteka szkolna jest czynna od 7:30 do 15:30. Można wypożyczyć 4 książki na 21 dni i przedłużyć termin jeden raz.",
    "Obiady w stołówce są wydawane od 11:30 do 14:00. Obiad można odwołać najpóźniej do godziny 8:00 w tym samym dniu.",
    "Koło informatyczne spotyka się we wtorki o 15:00 w sali 12. Prowadzi je mgr Anna Bajtowska, a udział jest dobrowolny.",
    "Próbna matura z matematyki odbędzie się w pierwszym tygodniu grudnia i potrwa 180 minut. Wolno używać kalkulatora prostego i tablic matematycznych.",
    "Notatka z biologii: mitochondrium to organellum komórkowe, w którym powstaje ATP w procesie oddychania komórkowego. Nazywa się je „elektrownią komórki”.",
    "Notatka z historii: Mieszko I przyjął chrzest w 966 roku w obrządku łacińskim, za pośrednictwem Czech. Wydarzenie to wprowadziło Polskę do kręgu zachodniego chrześcijaństwa.",
]

print(f"Załadowano {len(documents)} dokumentów.")
```

👀 **Co powinieneś zobaczyć:** `Załadowano 11 dokumentów.`

## Krok 5 – Eksperyment: model BEZ RAG ⌨️

Zapytajmy model o regulamin, **którego nie zna**:

```python
pytanie = "Ile mam dni na usprawiedliwienie nieobecności w naszej szkole?"

print("Pytanie:", pytanie)
print("Odpowiedź BEZ RAG:", ask_llm(pytanie))
```

👀 **Co powinieneś zobaczyć:** model albo przyzna, że nie zna regulaminu Twojej szkoły, albo poda coś ogólnego lub **zmyślonego** (np. „zwykle 7 lub 14 dni"). Poprawnej, pewnej odpowiedzi „7 dni, przez dziennik lub kartką od rodzica" **nie ma skąd wziąć**.

> 💬 **Zapisz sobie:** co odpowiedział model? Zmyślił, czy się przyznał? Porównaj z sąsiadem – odpowiedzi mogą się różnić!

## Krok 6 – Wyszukiwarka znaczeniowa (retriever)

Jak znaleźć w bazie właściwy dokument? Nie możemy szukać po identycznych słowach – w pytaniu użyjesz „absencji", a w regulaminie jest „nieobecność".

Użyjemy **modelu embeddingów**: zamienia on każdy tekst na **listę liczb** (wektor) tak, że teksty o **podobnym znaczeniu** dostają **podobne liczby**. Wtedy wyszukiwanie to tylko porównywanie liczb.

W Części B zajrzymy do środka tego mechanizmu. Na razie używamy go jak czarnej skrzynki.

### 6a. Ładujemy model embeddingów i zamieniamy dokumenty na wektory ⌨️

```python
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

doc_embeddings = embed_model.encode(documents, normalize_embeddings=True)

print(doc_embeddings.shape)
```

👀 **Co powinieneś zobaczyć:** `(11, 384)` – 11 dokumentów, każdy zamieniony na 384 liczby.

> 💡 Wybraliśmy model **wielojęzyczny**, bo nasze dokumenty są po polsku. Dlaczego to ważne – przekonasz się w Części B!

### 6b. Funkcja `search` ⌨️

```python
def search(query, docs, doc_embs, top_k=1):
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    scores = np.dot(doc_embs, query_emb.T).flatten()   # podobieństwo do każdego dokumentu
    best = np.argsort(scores)[::-1][:top_k]            # od najlepszego do najgorszego
    return [(float(scores[i]), docs[i]) for i in best]


for score, doc in search("Co grozi za spóźnienie na lekcję?", documents, doc_embeddings, top_k=3):
    print(f"{score:.2f} | {doc[:80]}...")
```

👀 **Co powinieneś zobaczyć:** trzy dokumenty z liczbą (podobieństwem) – **na górze dokument o spóźnieniach** z wynikiem ok. 0,8. Im wynik bliższy 1, tym bardziej podobne znaczenie.

## Krok 7 – Składamy RAG w całość ⌨️

Łączymy wszystko: **wyszukaj → wklej do polecenia → zapytaj model**.
Najważniejsze jest **polecenie (prompt)**: mówimy modelowi, żeby odpowiadał **tylko** na podstawie dostarczonego tekstu.

```python
def ask_rag(query, docs, doc_embs, top_k=1, min_score=None):
    # 1. WYSZUKAJ
    found = search(query, docs, doc_embs, top_k=top_k)
    for score, doc in found:
        print(f"   🔎 podobieństwo {score:.2f}: {doc[:70]}...")

    if min_score is not None and found[0][0] < min_score:
        return "Nie wiem (nie ma nic podobnego w bazie wiedzy)."

    context = "\n".join(doc for _, doc in found)

    # 2. WKLEJ DO POLECENIA
    prompt = (
        "Odpowiedz na pytanie użytkownika, opierając się TYLKO na podanym kontekście. "
        "Jeśli w kontekście nie ma odpowiedzi, napisz 'Nie wiem'.\n\n"
        f"Kontekst: {context}\n\n"
        f"Pytanie: {query}"
    )

    # 3. WYGENERUJ ODPOWIEDŹ
    return ask_llm(prompt)


pytanie = "Ile mam dni na usprawiedliwienie nieobecności w naszej szkole?"
print("Pytanie:", pytanie)
print("Odpowiedź Z RAG:", ask_rag(pytanie, documents, doc_embeddings))
```

👀 **Co powinieneś zobaczyć:** znaleziony dokument o usprawiedliwianiu i odpowiedź w stylu *„Masz 7 dni od powrotu do szkoły…"*.

🎉 **To samo pytanie i ten sam model co w Kroku 5 – a odpowiedź inna.** Różnica? Teraz model dostał „ściągę".

## Krok 8 – Testujemy asystenta ⌨️

```python
pytania = [
    "Co się stanie, jeśli spóźnię się na lekcję o pół godziny?",
    "Ile razy mogę poprawiać sprawdzian?",
    "Czy mogę używać telefonu na lekcji?",
    "Do której mogę odwołać obiad?",
    "Ile trwa próbna matura z matematyki?",
    "Jaka będzie jutro pogoda w Bajtowie?",
    "Ile kosztuje obiad w stołówce?",
]

for p in pytania:
    print("❓", p)
    print("💬", ask_rag(p, documents, doc_embeddings))
    print("-" * 60)
```

👀 **Co powinieneś zobaczyć:** pierwsze 5 pytań – poprawne odpowiedzi. **Pytanie o pogodę** – w bazie nie ma nic na ten temat, więc model powinien napisać `Nie wiem`.
**Pytanie o cenę obiadu** jest podchwytliwe: obiady są w bazie, ale **kwoty nikt nie podał**. Czy model przyznał się, że nie wie, czy zmyślił cenę?

### 🧪 Twoja kolej – zadania do Części A

1. **Dopisz własny dokument.** Dodaj do listy `documents` nowy wpis (np. zasady z Twojej prawdziwej szkoły), a potem **uruchom jeszcze raz**:
   ```python
   doc_embeddings = embed_model.encode(documents, normalize_embeddings=True)
   ```
   Zadaj pytanie o ten dokument. *To cała „aktualizacja wiedzy" w RAG – bez trenowania modelu!*
2. **Zepsuj prompt.** W funkcji `ask_rag` usuń słowa „opierając się TYLKO na podanym kontekście". Zadaj pytanie o cenę obiadu. Czy model zaczyna zmyślać?
3. **Zmień `top_k`** na 3. Czy odpowiedzi są lepsze, czy gorsze?
4. **Próg pewności.** Uruchom `ask_rag(pytanie, documents, doc_embeddings, min_score=0.4)` dla pytania o pogodę i o cenę obiadu. Podglądnij podobieństwa (`🔎`). Który przypadek próg wyłapie, a którego nie? Dlaczego?

---

# CZĘŚĆ B – Embeddingi: jak komputer „rozumie" znaczenie?

W Części A wyszukiwarka zadziałała „magicznie". Sprawdźmy, jak to działa.

> **Embedding** = tekst zamieniony na punkt w przestrzeni o wielu wymiarach. Teksty o podobnym znaczeniu leżą **blisko siebie**, o różnym – **daleko**.
> Wyobraź sobie mapę, na której każde zdanie ma współrzędne GPS. Zdania o kotach mieszkają w jednej dzielnicy, o pizzy w drugiej. Tylko że ta mapa ma nie 2 wymiary, ale 384.

## B1 – Jak wygląda embedding? ⌨️

```python
wektor = embed_model.encode("Kot śpi na kanapie.")

print("Długość wektora:", len(wektor))
print("Pierwsze 10 liczb:", np.round(wektor[:10], 3))
```

👀 **Co powinieneś zobaczyć:** `Długość wektora: 384` i dziesięć liczb (u Ciebie wyglądających mniej więcej tak: `0.368 -0.316 -0.139 …`).
Pojedyncza liczba nic nie znaczy – dopiero **porównywanie** wektorów ma sens.

## B2 – Podobieństwo zdań ⌨️

Podobieństwo mierzymy liczbą od ok. **0** (brak związku) do **1** (to samo znaczenie).

```python
zdania = [
    "Kot śpi na kanapie.",
    "Mruczący kociak drzemie na sofie.",
    "Pies goni piłkę w parku.",
    "Kwantowy procesor działa w temperaturze pokojowej.",
]

emb = embed_model.encode(zdania, normalize_embeddings=True)

for i in range(1, len(zdania)):
    print(f"{float(np.dot(emb[0], emb[i])):5.2f}  ← {zdania[i]}")
```

👀 **Co powinieneś zobaczyć:** najwyższy wynik (ok. 0,4) dla zdania o kociaku, a dla pozostałych wartości bliskie zera.
Zdanie o kociaku **nie ma ani jednego wspólnego słowa** z „Kot śpi na kanapie" – a model i tak widzi, że chodzi o to samo. To właśnie siła embeddingów.

### 🧪 Twoja kolej
Sprawdź podobieństwo własnych par zdań:
- tego samego znaczenia, innymi słowami,
- podobnych słów, ale **przeciwnego** sensu („Lubię matematykę" / „Nie lubię matematyki"),
- całkiem niezwiązanych.

Który wynik Cię zaskoczył?

## B3 – Szukanie po słowach (Ctrl+F) kontra szukanie po znaczeniu ⌨️

```python
import re

def slowa(tekst):
    return set(re.findall(r"\w+", tekst.lower()))

def search_keywords(query, docs):
    """Wygrywa dokument z największą liczbą wspólnych słów."""
    wspolne = [len(slowa(query) & slowa(d)) for d in docs]
    best = int(np.argmax(wspolne))
    return docs[best], wspolne[best]


for p in ["Do kiedy rodzic ma wytłumaczyć moją absencję?", "Skąd komórka bierze energię?"]:
    print("❓", p)
    doc, n = search_keywords(p, documents)
    print(f"   📝 po słowach ({n} wspólnych): {doc[:60]}...")
    score, doc = search(p, documents, doc_embeddings)[0]
    print(f"   🧠 po znaczeniu ({score:.2f}):  {doc[:60]}...")
```

👀 **Co powinieneś zobaczyć:** wyszukiwanie po słowach trafia w **zły dokument** (albo wcale nic nie znajduje – 0 wspólnych słów), a wyszukiwanie po znaczeniu wskazuje właściwy: o usprawiedliwianiu nieobecności i o mitochondriach.
Polszczyzna ma odmianę (komórka / komórkowe), synonimy (absencja / nieobecność) – dla Ctrl+F to koszmar, dla embeddingów nie.

## B4 – Nie każdy model embeddingów jest równie dobry 📋

Różne modele uczono na różnych danych. Model uczony głównie na **angielskim** może słabo radzić sobie z **polskim**.
Zróbmy **test jakości**: mamy pytania i wiemy, który dokument jest właściwy. Sprawdzamy, ile razy model go wskaże.

```python
testy = [
    ("Co się stanie, jeśli przyjdę na lekcję pół godziny po dzwonku?", 0),
    ("Do kiedy rodzic ma wytłumaczyć moją absencję?", 1),
    ("Czy mogę mieć smartfon na ławce podczas lekcji?", 2),
    ("Ile klasówek może być jednego dnia?", 3),
    ("Dostałem jedynkę ze sprawdzianu, czy mogę ją zmienić?", 4),
    ("Do której czynna jest czytelnia?", 5),
    ("Do której mogę zrezygnować z obiadu?", 6),
    ("Kiedy mogę pójść na zajęcia z programowania?", 7),
    ("Ile trwa egzamin próbny z matematyki?", 8),
    ("Skąd komórka bierze energię?", 9),
    ("Kiedy Mieszko został ochrzczony?", 10),
]

def evaluate(model_name, query_prefix="", doc_prefix=""):
    m = SentenceTransformer(model_name)
    d_embs = m.encode([doc_prefix + d for d in documents], normalize_embeddings=True)
    trafienia = 0
    for pytanie, poprawny in testy:
        q_emb = m.encode([query_prefix + pytanie], normalize_embeddings=True)
        ok = int(np.argmax(np.dot(d_embs, q_emb.T))) == poprawny
        trafienia += ok
        print(f"  {'✅' if ok else '❌'} {pytanie}")
    print(f"→ {model_name}: {trafienia}/{len(testy)}\n")


print("Model ANGIELSKI:")
evaluate("sentence-transformers/all-MiniLM-L6-v2")

print("Model WIELOJĘZYCZNY:")
evaluate("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

👀 **Co powinieneś zobaczyć:** model angielski trafia w ok. **połowę** pytań (u nas 5/11), a wielojęzyczny w zdecydowaną większość (u nas 9/11). Oba są podobnej wielkości – różnica wynika z tego, **na czym je uczono**.

> 🔑 **Wniosek:** RAG jest tak dobry, jak jego wyszukiwarka. Jeśli retriever poda zły fragment, nawet najlepszy model odpowie źle (albo „Nie wiem"), bo dostał niewłaściwą ściągę.

### 🧪 Twoja kolej
1. Zobacz, na jakich pytaniach wielojęzyczny model się pomylił. Dlaczego Twoim zdaniem? (Podpowiedź: „klasówka" a „sprawdzian"…)
2. Dopisz **własne, podchwytliwe pytania** do listy `testy` – parafrazy, żargon uczniowski, pytania bez wspólnych słów z regulaminem.
3. Sprawdź inne modele:
   ```python
   evaluate("intfloat/multilingual-e5-small", query_prefix="query: ", doc_prefix="passage: ")
   evaluate("BAAI/bge-m3")
   ```
   Który wygrywa? Tak właśnie inżynierowie wybierają modele do prawdziwych systemów – **testując na własnych danych**, nie na wyczucie.

## B5 – Mapa znaczeń (dla chętnych) 📋

Wektor ma 384 liczby, a ekran 2 wymiary. Metoda **PCA** „spłaszcza" dane do 2D, starając się zachować odległości.
Weźmy zdania z czterech przedmiotów i zobaczmy, czy model sam je pogrupuje (nie mówiliśmy mu, jakie są przedmioty!).

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

przedmioty = {
    "matematyka": [
        "Pole kwadratu to bok razy bok.", "Rozwiąż równanie kwadratowe za pomocą delty.",
        "Suma kątów w trójkącie wynosi 180 stopni.", "Funkcja liniowa ma wykres w kształcie prostej.",
        "Oblicz pochodną tej funkcji.",
    ],
    "biologia": [
        "Chloroplasty prowadzą fotosyntezę.", "DNA zawiera informację genetyczną.",
        "Serce pompuje krew do całego ciała.", "Mitochondria produkują energię dla komórki.",
        "Żaby są płazami.",
    ],
    "historia": [
        "Bitwa pod Grunwaldem odbyła się w 1410 roku.", "Konstytucja 3 maja została uchwalona w 1791 roku.",
        "Napoleon przegrał pod Waterloo.", "Mieszko I przyjął chrzest w 966 roku.",
        "II wojna światowa zaczęła się w 1939 roku.",
    ],
    "informatyka": [
        "Pętla for powtarza instrukcje wiele razy.", "Zmienna przechowuje wartość w pamięci.",
        "Algorytm sortowania porządkuje listę liczb.", "Błąd w programie nazywamy bugiem.",
        "Sieć neuronowa uczy się na danych.",
    ],
}
kolory = {"matematyka": "tab:blue", "biologia": "tab:green", "historia": "tab:orange", "informatyka": "tab:red"}

wszystkie = [z for lista in przedmioty.values() for z in lista]
etykiety = [k for k, lista in przedmioty.items() for _ in lista]

punkty = PCA(n_components=2, random_state=0).fit_transform(
    embed_model.encode(wszystkie, normalize_embeddings=True)
)

plt.figure(figsize=(9, 6))
for nazwa, kolor in kolory.items():
    idx = [i for i, e in enumerate(etykiety) if e == nazwa]
    plt.scatter(punkty[idx, 0], punkty[idx, 1], c=kolor, s=90, label=nazwa, alpha=0.8)
    for i in idx:
        plt.annotate(wszystkie[i].split()[0], punkty[i], textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.title("Mapa znaczeń zdań (embeddingi spłaszczone do 2D)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

👀 **Co powinieneś zobaczyć:** wykres z kolorowymi punktami. Zdania z tego samego przedmiotu **zazwyczaj tworzą skupiska** – choć model nigdy nie dostał etykiet. Nauczył się tego, czytając ogromne ilości tekstu.
Uwaga: to tylko *cień* 384-wymiarowej przestrzeni; coś może się nakładać na wykresie, choć w pełnej przestrzeni jest daleko.

### 🧪 Twoja kolej
Dopisz piąty przedmiot (np. „geografia", „chemia") i sprawdź, gdzie trafią jego zdania. Dodaj też zdanie „na granicy" dwóch przedmiotów (np. o biologii obliczeniowej) – gdzie wyląduje?

---

# CZĘŚĆ C – Twój własny asystent do nauki

Teraz to, co może Ci się naprawdę przydać: **asystent odpowiadający z Twoich notatek**. Wklej do niego fragment lektury, notatkę z lekcji albo rozdział z podręcznika – i zadawaj pytania, tak jak przed sprawdzianem.

Długiego tekstu nie wklejamy do modelu w całości (ma ograniczoną „pamięć roboczą"), a embedding całego rozdziału byłby rozmyty. Dlatego **tniemy tekst na krótkie fragmenty** (*chunki*) i każdy zamieniamy na wektor osobno.

## C1 – Dzielenie na fragmenty ⌨️

```python
def chunk_text(text, sentences_per_chunk=2):
    zdania = re.split(r"(?<=[.!?])\s+", text.strip())
    return [" ".join(zdania[i:i + sentences_per_chunk]) for i in range(0, len(zdania), sentences_per_chunk)]
```

## C2 – Wklej własny tekst i zadaj pytanie ⌨️

Poniżej jest tekst przykładowy. **Zamień go na własny** (np. notatkę z historii, biologii, fragment lektury) – pamiętaj o zachowaniu potrójnych cudzysłowów `"""` na początku i końcu.

```python
moj_tekst = """
Wisła jest najdłuższą rzeką w Polsce. Ma długość około 1047 kilometrów. Jej źródła znajdują się w Beskidzie Śląskim.
Wisła przepływa między innymi przez Kraków, Warszawę, Płock, Toruń i Gdańsk. Uchodzi do Morza Bałtyckiego.
Toruń słynie z pierników oraz z tego, że urodził się w nim Mikołaj Kopernik. Starówka Torunia znajduje się na liście UNESCO.
Pierniki toruńskie wypieka się według receptur znanych od średniowiecza. Ich ciasto dojrzewa nawet kilka miesięcy.
"""

chunki = chunk_text(moj_tekst, sentences_per_chunk=2)
chunki_emb = embed_model.encode(chunki, normalize_embeddings=True)

print(f"Tekst podzielony na {len(chunki)} fragmentów.")

pytanie = "Jak długo dojrzewa ciasto na pierniki?"
print("❓", pytanie)
print("💬", ask_rag(pytanie, chunki, chunki_emb, top_k=1))
```

👀 **Co powinieneś zobaczyć:** `Tekst podzielony na 5 fragmentów.` i odpowiedź, że ciasto dojrzewa nawet kilka miesięcy.

### 🧪 Twoja kolej
1. Wklej **własną notatkę** i zadaj 3 pytania „sprawdzianowe". Czy asystent się nie myli?
2. Zmień `sentences_per_chunk` na 1, potem na 4. Co się dzieje z jakością odpowiedzi?
   *(Za małe fragmenty → brakuje kontekstu, np. „Ich ciasto dojrzewa…" – czyje?; za duże → rozmyty sens i dużo „szumu".)*
3. Zadaj pytanie wymagające informacji z **dwóch różnych fragmentów** i spróbuj `top_k=2`.
4. Zadaj pytanie, na które w tekście **nie ma odpowiedzi**. Czy asystent mówi „Nie wiem"?

---

## Podsumowanie – czego się nauczyłeś/aś?

- 🧠 Modele językowe **nie znają Twoich danych** i potrafią zmyślać. RAG daje im „ściągę" z wiarygodnego źródła.
- 📚 **Aktualizacja wiedzy** w RAG to dopisanie dokumentu – bez trenowania modelu.
- 🔢 **Embedding** = tekst zamieniony na wektor; podobne znaczenie → podobne wektory.
- 🌍 **Wybór modelu embeddingów ma znaczenie** (np. język!) i trzeba go **testować** na własnych danych.
- ✂️ **Chunking** – długie teksty tniemy na fragmenty.
- ⚠️ RAG nie jest magiczny: jeśli wyszukiwarka poda zły fragment albo w bazie jest błąd, odpowiedź będzie zła – i to pewnym tonem.

### RAG a trenowanie modelu (fine-tuning)

| | **Fine-tuning** | **RAG** |
|---|---|---|
| Analogia | uczysz się na pamięć | piszesz z otwartymi notatkami |
| Co się zmienia? | sam model | tylko polecenie, które mu dajemy |
| Koszt | duży (GPU, czas, dane) | mały |
| Aktualizacja wiedzy | trening od nowa | dopisz dokument |
| Skąd odpowiedź? | trudno sprawdzić | **możesz pokazać źródło** |

### 💭 Pytania do dyskusji

1. Do czego w Twojej szkole przydałby się taki asystent? (Regulamin? Plan lekcji? Notatki klasowe? FAQ dla rodziców?)
2. Jak sprawdzić, czy odpowiedź **na pewno** pochodzi z bazy wiedzy, a nie z „głowy" modelu?
3. Co się stanie, jeśli w bazie ktoś umieści **fałszywą informację**? Kto powinien mieć prawo dopisywać dokumenty?
4. Czy warto wysyłać do chmury prywatne dokumenty (dziennik, oceny, dane osobowe)? Co daje uruchomienie modelu **lokalnie**, jak dziś?

### 🚀 Zadania dodatkowe (po warsztatach)

- **Pokaż źródło:** zmodyfikuj `ask_rag`, by oprócz odpowiedzi zwracała też tekst użytego dokumentu.
- **Asystent na sprawdzian:** wklej 10–20 zdań z jednego działu i poproś model: *„Na podstawie kontekstu ułóż 3 pytania sprawdzające"*.
- **Własny test jakości:** dopisz po 10 pytań do `testy` i wybierz najlepszy model embeddingów dla swojego tekstu.
- **Skalowanie:** do 1000 dokumentów wystarczy `np.dot`. Co zrobić przy milionie? Poszukaj haseł: *baza wektorowa*, *FAISS*, *ChromaDB*.

---

## 🆘 Co zrobić, gdy coś nie działa?

| Objaw | Co zrobić |
|---|---|
| `Cannot access gated repo` / `401` / `403` przy ładowaniu modelu | Nie zaakceptowałeś licencji Gemma na Hugging Face albo token jest zły. Zrób Krok 0 jeszcze raz i **uruchom Krok 1 ponownie**. |
| `SecretNotFoundError` | Brak sekretu `HF_TOKEN` albo wyłączony *Notebook access*. |
| `CUDA out of memory` | Model został załadowany dwa razy. **Środowisko wykonawcze → Uruchom ponownie sesję**, potem uruchom komórki po kolei od Kroku 1. |
| `NameError: name '...' is not defined` | Pominąłeś/aś komórkę albo uruchomiłeś/aś je w złej kolejności. Uruchom brakującą komórkę (najczęściej z wcześniejszego kroku). |
| Błąd `SyntaxError` / `IndentationError` | Literówka lub złe wcięcia. W Pythonie **wcięcia (spacje na początku linii) mają znaczenie** – porównaj ze wzorem linijka po linijce. |
| Wynik „Nie wiem" mimo że odpowiedź jest w bazie | Po dopisaniu dokumentów **przelicz embeddingi** (`doc_embeddings = embed_model.encode(...)`). |
| Ładowanie modelu trwa bardzo długo | To normalne przy pierwszym uruchomieniu (kilka minut). Jeśli trwa > 10 minut – zgłoś się do prowadzącego. |
| Colab „zapomniał" wszystko po przerwie | Sesja wygasa po bezczynności. Uruchom komórki od Kroku 1 jeszcze raz. |

---

**Dziękujemy za udział w warsztatach! 🎓**
Notatnik możesz rozwijać po warsztatach – wystarczy zapisać go na swoim Dysku Google (*Plik → Zapisz kopię na Dysku*).
