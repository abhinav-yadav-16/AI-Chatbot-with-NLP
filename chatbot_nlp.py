

import math
import re
import sys
from collections import Counter, defaultdict

# ---------------- Knowledge Base ----------------
KB = [
    ("hello", "Hi there! I'm your NLP chatbot. How can I help you today?"),
    ("hi", "Hello! How can I assist you?"),
    ("what is your name", "I'm ChatPy — a simple NLP chatbot built for demo."),
    ("who made you", "You asked for an AI chatbot — so let's say you made me (with help!)."),
    ("what can you do", "I can answer simple FAQs, chit-chat, and retrieve the best matching answer from my knowledge base."),
    ("how are you", "I'm a program, so I don't have feelings, but I'm ready to help!"),
    ("tell me a joke", "Why did the programmer quit his job? Because he didn't get arrays (a raise)."),
    ("thank you", "You're welcome!"),
    ("bye", "Goodbye — have a great day!"),
    ("how to learn machine learning", "Start with Python, linear algebra, statistics, then practice with projects using scikit-learn or TensorFlow.")
]

# ---------------- Tokenizer Setup ----------------
USE_SPACY = False
USE_NLTK = False

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    USE_SPACY = True

    def spacy_tokenize(text):
        doc = nlp_spacy(text)
        return [tok.lemma_.lower() for tok in doc if not tok.is_punct and not tok.is_space]

    tokenize = spacy_tokenize
except Exception:
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        from nltk import word_tokenize

        lemmatizer = WordNetLemmatizer()
        nltk.data.find('tokenizers/punkt')

        USE_NLTK = True

        def nltk_tokenize(text):
            toks = word_tokenize(text.lower())
            toks = [re.sub(r'\W+', '', t) for t in toks if re.sub(r'\W+', '', t)]
            return [lemmatizer.lemmatize(t) for t in toks]

        tokenize = nltk_tokenize
    except Exception:
        def simple_tokenize(text):
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            return text.split()

        tokenize = simple_tokenize

# ---------------- TF-IDF Index ----------------
def build_index(kb):
    docs_tokens = [tokenize(q) for q, a in kb]
    vocab = {}
    for tokens in docs_tokens:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    vectors = []
    idf = defaultdict(int)
    for tokens in docs_tokens:
        tf = Counter(tokens)
        vectors.append(tf)
        for t in set(tokens):
            idf[t] += 1

    N = len(docs_tokens)
    tfidf_vectors = []
    for tf in vectors:
        vec = [0.0] * len(vocab)
        for t, count in tf.items():
            vec[vocab[t]] = (1 + math.log(count)) if count > 0 else 0.0
        tfidf_vectors.append(vec)

    for vec in tfidf_vectors:
        for term, idx in vocab.items():
            df = idf[term]
            if df > 0:
                vec[idx] *= math.log((1 + N) / (1 + df)) + 1.0
    return vocab, tfidf_vectors


vocab, kb_vectors = build_index(KB)

# ---------------- Utils ----------------
def vectorize_query(query, vocab):
    tokens = tokenize(query)
    tf = Counter(tokens)
    vec = [0.0] * len(vocab)
    for t, count in tf.items():
        if t in vocab:
            vec[vocab[t]] = (1 + math.log(count)) if count > 0 else 0.0
    return vec, tokens


def cosine_sim(a, b):
    num = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return num / (math.sqrt(na) * math.sqrt(nb))


def find_best_answer(query, kb, kb_vectors, vocab, threshold=0.15):
    qvec, tokens = vectorize_query(query, vocab)
    best_idx, best_score = None, 0.0
    for i, dvec in enumerate(kb_vectors):
        score = cosine_sim(qvec, dvec)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score >= threshold:
        return kb[best_idx][1], best_score

    lowq = query.lower()
    if any(g in lowq for g in ["hi", "hello", "hey"]):
        return "Hello! What can I do for you?", 0.0
    if "joke" in lowq:
        return "Why do programmers prefer dark mode? Because light attracts bugs.", 0.0
    if "bye" in lowq:
        return "Goodbye! Feel free to chat again.", 0.0

    return "Sorry, I don't know the answer to that yet.", 0.0


# ---------------- Chat Loop ----------------
def interactive_chat():
    print("ChatPy — simple NLP chatbot. Type 'exit' or 'quit' to stop.")
    print("Using:", "spaCy" if USE_SPACY else ("NLTK" if USE_NLTK else "SimpleTokenizer"))
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            print("Bot: Goodbye!")
            break
        ans, score = find_best_answer(query, KB, kb_vectors, vocab)
        print(f"Bot: {ans}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        samples = [
            "Hello",
            "Who made you?",
            "tell me a joke",
            "How to learn machine learning?",
            "What's your name?",
            "Goodbye"
        ]
        for s in samples:
            ans, score = find_best_answer(s, KB, kb_vectors, vocab)
            print("Q:", s)
            print("A:", ans, f"(score={score:.3f})\n")
    else:
        interactive_chat()
