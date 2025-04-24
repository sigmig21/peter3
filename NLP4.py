from nltk.stem import WordNetLemmatizer
import re

import nltk
nltk.download('wordnet')
# NLTK Lemmatizer (Existing Code)
wnl = WordNetLemmatizer()

def lemmatize_words_nltk(word_list):
    """Lemmatize words using NLTK's WordNetLemmatizer."""
    print("\n--- Using NLTK WordNetLemmatizer ---")
    for word in word_list:
        # NLTK lemmatizer can take a POS tag, but for simplicity, we'll use the default (noun)
        # A more advanced version would require POS tagging first.
        print(f"{word} ---> {wnl.lemmatize(word)}")

# --- Simple Scratch Lemmatizer (From Scratch Concept) ---

def simple_scratch_lemmatizer(word_list):
    """
    A very basic, rule-based lemmatizer implemented from scratch.
    This is a simplified illustration and not a comprehensive lemmatizer.
    It primarily handles simple plural forms.
    """
    print("\n--- Using Simple Scratch Lemmatizer ---")
    lemmatized_words = []
    for word in word_list:
        original_word = word
        lemmatized_word = word.lower() # Start with lowercase

        # Basic rules for plurals (very simplified)
        if lemmatized_word.endswith('ies'):
            lemmatized_word = lemmatized_word[:-3] + 'y'
        elif lemmatized_word.endswith('es'):
            # Handle cases like 'boxes' -> 'box', but not 'goes' -> 'go' (requires more complex rules)
            if len(lemmatized_word) > 3 and lemmatized_word[-3] in 'sxz' or lemmatized_word[-4:-2] in ['sh', 'ch']:
                 lemmatized_word = lemmatized_word[:-2]
            # Simple rule for words ending in 'e' + 's' like 'houses' -> 'house'
            elif lemmatized_word.endswith('ses'):
                 lemmatized_word = lemmatized_word[:-1] # Remove just the 's'
            else:
                 pass # More complex cases not handled
        elif lemmatized_word.endswith('s') and not lemmatized_word.endswith('ss'):
            # Simple rule for most words ending in 's'
            lemmatized_word = lemmatized_word[:-1]

        # Add more rules here for other cases (e.g., irregular verbs, adjectives)
        # This is just a basic illustration.

        lemmatized_words.append((original_word, lemmatized_word))
        print(f"{original_word} ---> {lemmatized_word}")

# Sample input words
sample_words = ["running", "flies", "cats", "dogs", "better", "best", "corpora", "boxes", "houses", "goes"]

# Lemmatize the sample words using NLTK
lemmatize_words_nltk(sample_words)

# Lemmatize the sample words using the simple scratch lemmatizer
simple_scratch_lemmatizer(sample_words)

# --- Conceptual Explanation ---
# This assignment focuses on **lemmatization**, another important text normalization technique in Natural Language Processing (NLP).
# **Lemmatization:** This is the process of reducing words to their base or dictionary form, known as a "lemma". Unlike stemming, which often just chops off endings and might result in non-words, lemmatization aims to find the true root word based on its meaning and part of speech. For example, "running", "runs", and "ran" would all be lemmatized to "run".
#
# The assignment contrasts two approaches to lemmatization:
# 1. **NLTK's `WordNetLemmatizer`:** This lemmatizer uses the WordNet lexical database. WordNet is a large database of English words linked together by their semantic relationships. The `WordNetLemmatizer` uses this database to look up the lemma of a word. For accurate lemmatization, it often requires the Part-of-Speech (POS) tag of the word (e.g., whether "run" is used as a verb or a noun), as the lemma can depend on the word's grammatical role. The example in the code simplifies this by using the default POS (which is typically noun), but a more robust implementation would involve POS tagging first.
# 2. **Simple Scratch Lemmatizer (From Scratch Concept):** This is a very basic, rule-based lemmatizer implemented manually. It applies predefined rules, primarily focused on handling simple plural forms (e.g., removing 's', 'es', 'ies'). This approach is illustrative of the concept but is highly limited. It cannot handle irregular forms (like "ran" -> "run" or "better" -> "good") and would require a vast and complex set of rules to be effective for a wide range of words and linguistic phenomena.
#
# The assignment demonstrates both methods on sample words, including plurals and some irregular forms. The output shows how the NLTK lemmatizer, leveraging WordNet, can find the correct lemmas for many words, while the simple scratch lemmatizer is limited to the rules it implements. This comparison highlights the difference between a knowledge-based approach (NLTK + WordNet) and a simple rule-based approach, and underscores why lemmatization is generally preferred over stemming when linguistic accuracy is important. Lemmatization is valuable in applications where understanding the meaning of words is critical, such as text analysis, information retrieval, and machine translation.

# --- Potential Viva Questions ---
# What is lemmatization, and how does it differ from stemming? Provide examples.
# Expected Response: Lemmatization reduces words to their dictionary or base form (lemma), which is a valid word (e.g., "running", "runs", "ran" -> "run"). Stemming reduces words to a root form by chopping off endings, which may not be a valid word (e.g., "running" -> "runn"). Lemmatization is more linguistically accurate.
# Why is lemmatization generally considered more linguistically accurate than stemming?
# Expected Response: Lemmatization considers the meaning and part of speech of a word to find its true base form, often using a lexicon or dictionary. Stemming is a heuristic process that simply applies rules to chop off endings, without considering the word's meaning or grammatical role.
# How does NLTK's `WordNetLemmatizer` likely work (conceptually, referencing WordNet)?
# Expected Response: It uses the WordNet lexical database, which links words based on semantic relationships. The lemmatizer looks up the word in WordNet and, ideally with the correct POS tag, finds its corresponding lemma.
# Explain the rules implemented in the `simple_scratch_lemmatizer`. What are its shortcomings?
# Expected Response: The simple scratch lemmatizer uses basic rules primarily for handling regular plurals (removing 's', 'es', 'ies'). Its shortcomings include not handling irregular forms (like "ran", "better"), not considering context or POS, and requiring extensive manual rule creation for broader coverage.
# Why is knowing the Part-of-Speech of a word important for accurate lemmatization?
# Expected Response: Some words have different lemmas depending on their part of speech (e.g., "run" as a verb vs. "run" as a noun). Knowing the POS helps the lemmatizer select the correct base form from potential options.
# In what scenarios would you prefer using lemmatization over stemming, and vice versa?
# Expected Response: Prefer lemmatization when linguistic accuracy and meaning are important (e.g., text analysis, machine translation). Prefer stemming when speed and reducing vocabulary size are the primary goals, and the resulting non-words are acceptable (e.g., information retrieval, some text mining tasks).
# How would you handle irregular verbs (e.g., "ran" -> "run") in a scratch lemmatizer?
# Expected Response: Handling irregular forms in a scratch lemmatizer would require creating a lookup table or dictionary mapping irregular forms to their lemmas (e.g., {"ran": "run", "went": "go", "better": "good"}). This list would need to be manually compiled and maintained.
