from nltk.stem.snowball import SnowballStemmer # type: ignore


class language:
    def __init__(self, pre, suf):
        self.prefixes = pre
        self.suffixes = suf


class customStemmer:
    def __init__(self, lang):
        self.language = lang

    def stem(self, word):
        for prefix in self.language.prefixes:
            if word.startswith(prefix):
                word = word[len(prefix) :]
                break

        for suffix in self.language.suffixes:
            if word.endswith(suffix):
                word = word[: -len(suffix)]
                break

        if hasattr(self.language, 'char_replacements'):
             for original, replacement in self.language.char_replacements.items():
                 word = word.replace(original, replacement)


        return word


german_prefixes = ["ge", "be", "ver"]
german_suffixes = [ "en", "er", "ung", "heit", "keit", "isch", "lich", "ig", "e", "s", "n", "t",]
german_char_replacements = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}
german = language(german_prefixes, german_suffixes)
german.char_replacements = german_char_replacements


english_prefixes = ["re", "un"] # Example English prefixes
english_suffixes = ["ed", "ing", "ly", "s"] # Example English suffixes
english = language(english_prefixes, english_suffixes)


stemmer_german_snowball = SnowballStemmer("german")
stemmer_german_custom = customStemmer(german)

stemmer_english_snowball = SnowballStemmer("english")
stemmer_english_custom = customStemmer(english)


list_of_german_words = ["laufen", "gelaufen", "schönheit", "häuser", "freundschaft"]
list_of_english_words = ["running", "jumped", "happily", "unbelievable", "restarted"]

print("German Stemming:")
list_of_german_root_words_custom = [stemmer_german_custom.stem(word) for word in list_of_german_words]
list_of_german_root_words_snowball = [stemmer_german_snowball.stem(word) for word in list_of_german_words]

print("Custom Stemmer:", list_of_german_root_words_custom)
print("Snowball Stemmer:", list_of_german_root_words_snowball)

print("\nEnglish Stemming:")
list_of_english_root_words_custom = [stemmer_english_custom.stem(word) for word in list_of_english_words]
list_of_english_root_words_snowball = [stemmer_english_snowball.stem(word) for word in list_of_english_words]

print("Custom Stemmer:", list_of_english_root_words_custom)
print("Snowball Stemmer:", list_of_english_root_words_snowball)

# --- Conceptual Explanation ---
# This assignment explores the concept of stemming, a text normalization technique used in Natural Language Processing (NLP).
# Stemming reduces words to their root or base form, known as a "stem". The goal is to group together words that have similar meanings but different endings (e.g., "running", "runs", "ran" might all be reduced to "run").
# It's a heuristic process that often simply chops off the ends of words, which means the resulting stem may not be a linguistically correct word.
# The assignment demonstrates two approaches to stemming:
# 1. Simple Rule-Based Custom Stemmer (Implemented from Scratch): This approach involves defining a set of rules based on common prefixes and suffixes in a language. The `customStemmer` class applies these predefined rules to strip prefixes and suffixes from words. This method is straightforward but limited by the completeness and accuracy of the defined rules. It might fail on irregular forms or produce non-words.
# 2. Snowball Stemmer (from NLTK library): The Snowball stemmer is a more sophisticated algorithmic stemmer. It uses a set of rules and a series of steps to perform stemming for various languages. It's generally more effective than simple rule-based stemmers but can still produce non-words.
# The assignment compares the output of the custom stemmer with the NLTK Snowball stemmer for sample German and English words. This comparison highlights the differences in their approaches and the complexities involved in accurately reducing words to their stems across different languages and word forms. Stemming is often used as a preprocessing step in NLP tasks like information retrieval and text mining to reduce the vocabulary size and improve performance.

# --- Potential Viva Questions ---
# What is stemming, and why is it important in Natural Language Processing?
# What is the difference between stemming and lemmatization? (This is a common related concept)
# Explain the approach used in the `customStemmer` class. What are its limitations?
# How does the Snowball Stemmer work conceptually (you don't need to know the exact algorithm, but the idea)?
# Why might the output of a custom stemmer differ from a library stemmer like Snowball?
# In what NLP tasks would stemming be a useful preprocessing step?
# What are the potential drawbacks of using stemming?
# Expected Response: Stemming can produce non-words or stems that are not linguistically correct. It can also conflate words with different meanings if their stems are the same (e.g., "universal" and "university"). This can sometimes lead to a loss of meaning or reduced accuracy in certain NLP tasks.
