"""Synthetic retrieval QA with a dial for retriever quality.

Each chunk is a few filler sentences plus one fact about an invented entity.
The question asks for the fact of exactly one chunk. Retriever scores are
`signal * is_gold + noise * N(0, 1)`: turn `noise` up and the retriever gets
worse, which is what we need to see how the method degrades.

Entity and value names are made-up words, so a model can't answer from
pretraining; it has to read the right chunk.
"""
import numpy as np

SYLLABLES = ["ba", "ko", "ri", "zu", "me", "la", "to", "vi", "su", "ne",
             "da", "pho", "qua", "tra", "gle", "mir", "sol", "dun", "kel", "wor"]
ATTRIBUTES = ["capital", "founder", "currency", "anthem", "river", "mascot", "language", "emblem"]
FILLER = [
    "{e} lies far from the main trade roads.",
    "Most of {e} is farmland, with a few small markets.",
    "Travelers describe {e} as quiet in the winter months.",
    "The oldest records of {e} were kept by local monks.",
    "{e} was surveyed twice during the last century.",
    "Rainfall in {e} is higher than in the neighbouring valleys.",
    "A short railway once connected {e} to the coast.",
    "The population of {e} has stayed roughly the same for decades.",
]


def _word(rng, n=3):
    return "".join(rng.choice(SYLLABLES, size=n)).capitalize()


def make_example(rng, idx, k=8, filler=4, noise=1.0, signal=2.0):
    entities = []
    while len(entities) < k:
        w = _word(rng)
        if w not in entities:
            entities.append(w)
    attr = ATTRIBUTES[rng.randint(len(ATTRIBUTES))]
    gold_i = rng.randint(k)

    chunks, answer = [], None
    for i, ent in enumerate(entities):
        value = _word(rng, 2)
        sentences = [FILLER[j].format(e=ent) for j in rng.choice(len(FILLER), size=filler, replace=False)]
        sentences.insert(rng.randint(filler + 1), f"The {attr} of {ent} is {value}.")
        is_gold = i == gold_i
        if is_gold:
            answer = value
        chunks.append({
            "text": f"{ent}. " + " ".join(sentences),
            "score": float(signal * is_gold + noise * rng.randn()),
            "gold": is_gold,
        })

    return {
        "id": f"syn-{idx}",
        "question": f"What is the {attr} of {entities[gold_i]}?",
        "answers": [answer],
        "chunks": chunks,
    }


def make_dataset(n, k=8, filler=4, noise=1.0, signal=2.0, seed=0):
    rng = np.random.RandomState(seed)
    return [make_example(rng, i, k, filler, noise, signal) for i in range(n)]
