import spacy

nlp = spacy.load("en_core_web_sm")

# Common "function word" POS to drop (keeps content POS like NOUN/VERB/ADJ/ADV/PROPN/NUM)
DROP_POS = {
    "ADP",  # prepositions
    "AUX",  # auxiliaries/modals
    "CCONJ",  # coordinating conjunctions
    "SCONJ",  # subordinating conjunctions
    "DET",  # determiners
    "PART",  # particles (to, not, etc.)
    "PRON",  # pronouns
    "INTJ",  # interjections (often discourse-y)
}


def manipulate_sentence(text: str) -> str:
    doc = nlp(text)
    kept = []
    for tok in doc:
        if tok.is_punct:
            continue
        if tok.is_space:
            continue

        # spaCy stopwords cover many discourse markers (e.g., "therefore", "however") depending on model/version
        if tok.is_stop:
            continue

        # drop function-word POS
        if tok.pos_ in DROP_POS:
            continue

        # optional: drop pure symbols
        if (
            tok.is_currency
            or tok.like_num is False
            and tok.is_alpha is False
            and tok.text.strip() == ""
        ):
            continue

        # keep original order; match the paper’s examples by lowercasing
        kept.append(tok.text.lower())

    return " ".join(kept)
