import re

def simple_rule_based_pos_tagger(sentence):
    """
    A very basic rule-based POS tagger.
    This is a simplified example and not a comprehensive tagger.
    """
    # Convert to lowercase for simpler matching, but keep original for output
    words = sentence.split()
    tagged_words = []

    # Define some very basic rules
    # This is a highly incomplete set of rules
    rules = {
        r'\b(?:a|an|the)\b': 'DT',  # Determiner
        r'\b(?:is|am|are|was|were|be|been|being)\b': 'VB', # Verb (Be forms)
        r'\b(?:run|running|walk|walking|eat|eating)\b': 'VB', # Verb (some common verbs)
        r'\b(?:quick|brown|lazy)\b': 'JJ', # Adjective (some common adjectives)
        r'\b(?:fox|dog|cat|house)\b': 'NN', # Noun (some common nouns)
        r'\b(?:over|under|on|in|at)\b': 'IN', # Preposition
        r'\b(?:and|but|or)\b': 'CC', # Coordinating Conjunction
        r'\b(?:quickly|slowly)\b': 'RB', # Adverb
    }

    for word in words:
        # Remove punctuation for simpler matching, but keep original word
        cleaned_word = re.sub(r'[^\w\s]', '', word).lower()
        tag = 'NN' # Default tag: Noun (simple fallback)

        # Apply rules
        for pattern, pos_tag in rules.items():
            if re.match(pattern, cleaned_word):
                tag = pos_tag
                break # Use the first matching rule

        # Simple rule for proper nouns (starts with capital letter)
        if word and word[0].isupper() and tag == 'NN': # Added check for empty word
             tag = 'NNP' # Proper Noun

        tagged_words.append((word, tag))

    return tagged_words

# Sample sentence for tagging
sample_sentence = "The quick brown fox jumps over the lazy dog."

# Perform basic rule-based POS tagging
pos_tags = simple_rule_based_pos_tagger(sample_sentence)

# Print the original sentence and the POS tagged words
print(f"Original Sentence: {sample_sentence}")
print("\nPOS Tags (Basic Rule-Based):")
for word, tag in pos_tags:
    print(f"{word}: {tag}")

# Example with another sentence
another_sentence = "John is running quickly."
pos_tags_another = simple_rule_based_pos_tagger(another_sentence)

print(f"\nOriginal Sentence: {another_sentence}")
print("\nPOS Tags (Basic Rule-Based):")
for word, tag in pos_tags_another:
    print(f"{word}: {tag}")

# --- Conceptual Explanation ---
# This assignment introduces Part-of-Speech (POS) tagging, which is a fundamental task in Natural Language Processing (NLP).
# **Part-of-Speech (POS) Tagging:** This is the process of assigning a grammatical category, such as noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, etc., to each word in a sentence. The tag indicates the role of the word in the sentence and its relationship with other words. POS tagging is crucial for many downstream NLP tasks like parsing, named entity recognition, and machine translation.
#
# This assignment implements a very basic **rule-based POS tagger** from scratch.
# **Rule-Based POS Tagging:** This approach relies on a set of hand-written rules based on linguistic knowledge. These rules often involve patterns in word endings, prefixes, or surrounding words. For example, a rule might state that words ending in "-ing" are likely verbs (VBG - verb, gerund or present participle) or nouns (NN - noun, singular or mass). The tagger applies these rules sequentially or based on priority to assign tags.
#
# The tagger in this assignment uses a predefined set of regular expression rules to identify patterns in words and assign tags accordingly (e.g., identifying determiners like "the", "a", "an").
# It also includes a simple rule for identifying **proper nouns** based on capitalization.
# **Proper Nouns (NNP):** These are nouns that name a specific person, place, organization, or thing (e.g., "John", "London", "Google"). In English, proper nouns are typically capitalized. The simple rule here checks if a word starts with a capital letter and assigns it the 'NNP' tag if it's initially tagged as a common noun ('NN').
#
# The assignment demonstrates the tagging process on sample sentences and prints the words along with their assigned tags.
# It is important to note that this is a highly simplified example. Real-world POS taggers are much more complex and accurate, often using statistical methods (like Hidden Markov Models or Maximum Entropy models) or machine learning approaches (like Conditional Random Fields or deep learning models) trained on large annotated corpora. These methods learn patterns from data rather than relying solely on hand-crafted rules, making them more robust and capable of handling ambiguities.

# --- Potential Viva Questions ---
# What is Part-of-Speech (POS) tagging, and what is its purpose in NLP?
# Expected Response: POS tagging is the process of assigning a grammatical category (like noun, verb, adjective) to each word in a text. Its purpose is to understand the grammatical structure of a sentence, which is essential for many downstream NLP tasks.
# Explain how the rule-based tagger in this assignment works.
# Expected Response: This rule-based tagger uses a predefined set of hand-written rules, often based on patterns like word endings, prefixes, or specific words (like "the" for determiner). It iterates through words and applies the first matching rule to assign a tag. It also has a simple rule for proper nouns based on capitalization.
# What are the limitations of a simple rule-based POS tagger?
# Expected Response: Simple rule-based taggers struggle with ambiguity (words that can have multiple POS tags depending on context), require extensive manual rule creation and maintenance, and may not generalize well to new data or variations in language use.
# How do more advanced POS tagging methods (like HMMs, CRFs, or deep learning models) differ from this rule-based approach?
# Expected Response: Advanced methods are typically statistical or machine learning-based. They learn patterns and probabilities from large annotated datasets rather than relying on hand-written rules. This allows them to handle ambiguity better and generalize more effectively to unseen text.
# What are some applications of POS tagging?
# Expected Response: Applications include named entity recognition, sentiment analysis, machine translation, information extraction, text-to-speech systems, and syntactic parsing.
# How does punctuation affect this rule-based tagger?
# Expected Response: The current simple tagger removes punctuation before applying rules, which means punctuation itself is not tagged and might affect the identification of words if punctuation is attached to them. A more robust tagger would handle punctuation as separate tokens.
# Could you add more rules to improve this tagger? Give an example.
# Expected Response: Yes, for example, you could add rules for common verb endings (e.g., words ending in "-ed" are often past tense verbs - VBD), plural nouns (e.g., words ending in "-s" are often plural nouns - NNS), or specific function words (e.g., "to" is often a particle - TO, or preposition - IN).
