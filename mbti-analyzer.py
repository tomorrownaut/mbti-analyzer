import spacy
import numpy as np
from spacy import displacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from numpy import dot
from numpy.linalg import norm


cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))

nlp = spacy.load("en_core_web_md")
val = input("Enter your document: ")
doc = nlp(val)


print("\n----------------------------\nGrammar parsing:")
for token in doc:
    print(token.orth_, token.pos_, token.pos, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])




# gather all known words, take only the lowercased versions
#allWords = list({w for w in nlp.vocab if w.has_vector and w.orth_.islower() and w.lower_ != searchWordString})
# sort by similarity to the result
#allWords.sort(key=lambda w: cosine(w.vector, searchWord.vector))


matcher = Matcher(nlp.vocab)
tiPattern = [{"LOWER": "i"}, {"LOWER": "think"}]
matcher.add("Ti", [tiPattern])
tePattern = [{"LOWER": "they"}, {"LOWER": "think"}]
matcher.add("Te", [tePattern])
fiPattern = [{"LOWER": "i"}, {"LOWER": "feel"}]
matcher.add("Fi", [fiPattern])
fePattern = [{"LOWER": "they"}, {"LOWER": "feel"}]
matcher.add("Fe", [fePattern])
matches = matcher(doc)
print("\n----------------------------\nPattern matching:")
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(f"Pattern id: {string_id}, span: {start}, {end}, pattern text: {span.text}")




searchWordString = input("\nEnter your word-association search word: ")
searchWord = nlp.vocab[searchWordString]
computed_similarities = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower and word.text != searchWordString:
            if word.is_alpha:
                if not word.is_punct:
                    doc = nlp(searchWordString)
                    if doc[0].pos == 95:
                        continue
                    else:
                        similarity = cosine(searchWord.vector, word.vector)
                        computed_similarities.append((word, similarity))




computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
#allWords.reverse()
print(f"\n----------------------------\nTop 10 closest results for {searchWordString}:")
print([w[0].text for w in computed_similarities[:10]])
#for word in allWords[:10]:
#    print(word.orth_)
