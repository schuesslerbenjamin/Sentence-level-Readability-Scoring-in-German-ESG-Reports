import nltk
import string
import re


def get_word_count(sentence: str) -> int:
    """
    Calculate the number of words in one sentence.
    """

    return len(sentence.split())


def get_average_words_per_sentence(text: str, language = "english") -> float:
    """
    Calculate the average number of words per sentence in a text of multiple sentences.
    Unit: words
    """
    
    sentences = nltk.sent_tokenize(text, language=language) # split the text into sentences

    return sum(get_word_count(sentence) for sentence in sentences) / len(sentences)


def text_to_tokens(text: str, language: str = 'english', stem: bool = False, filter_stopwords: bool = False) -> list:
    """
    Convert a text into tokens.
    """

    tokens = text.lower().split() # lowercase the query and split into tokens

    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens] # Remove any punctuation characters
    tokens = [token for token in tokens if token] # remove empty tokens

    if stem:
        # Stem using snowball stemmer from nltk
        tokens = [nltk.SnowballStemmer(language=language, ignore_stopwords=True).stem(token) for token in tokens]

    if filter_stopwords:
        # only download nltk stopwords if missing
        try:
            tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words(language)]
        except:
            nltk.download('stopwords', quiet=True)
            tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words(language)]

    return tokens


def get_average_word_length(text: str, verbose: bool = False) -> float:
    """
    Calculate the average word length.
    Unit: characters
    """

    tokens = text_to_tokens(text)

    if verbose:
        print(tokens)

    return sum(len(token) for token in tokens) / len(tokens)


def count_words_longer_than(text: str, x: int = 6) -> int:
    """
    Calculate the number of words longer than x characters.
    """

    tokens = text_to_tokens(text)
    return len([token for token in tokens if len(token) > x])


def count_sentences_longer_than(text: str, threshold: int = 20, language: str = "english") -> int:
    """
    Calculate the number of sentences longer than threshold words.
    """

    sentences = nltk.sent_tokenize(text, language=language) # split the text into sentences

    return len([sentence for sentence in sentences if get_word_count(sentence) > threshold])


def count_words_with_more_than_x_syllables(sentence: str, x: int = 3, language: str = "english") -> int:
    """
    Count the number of words with more than x syllables in a sentence.
    """

    tokens = text_to_tokens(sentence)
    
    syllable_tokenizer = nltk.tokenize.SyllableTokenizer(lang = "en") # syllable tokenizer only works for English, but as we only want the number of syllables, the results are good enough for German

    syllables_per_word = [len(syllable_tokenizer.tokenize(token)) for token in tokens]

    polysyllabic_words = [syllables for syllables in syllables_per_word if syllables > x] # words with more than x syllables

    return len(polysyllabic_words)


def proportion_of_polysyllabic_words(text: str, x: int = 3, language: str = "english") -> float:
        """
        Calculate the proportion of polysyllabic words (more than x syllables) in a text.
        """

        tokens = text_to_tokens(text, language = language)
        total_words = len(tokens)
        polysyllabic_words = count_words_with_more_than_x_syllables(text, x, language)

        return polysyllabic_words / total_words if total_words > 0 else 0


def split_sentence_into_clauses(sentence: str) -> list:
    """
    Split a sentence into clauses using the split as in Kercher (2013) that use Harrison and Bakker (1998, p. 132-133) but without using brackets as splitting characters.
    """
    
    # Replace splitting characters (Komma, Doppelpunkt, Strichpunkt, Gedankenstrich) with dummy
    temp_sentence = re.sub(r'([,:;–—])', r'\1|||', sentence)
    
    # Split sentence at dummy
    parts = temp_sentence.split('|||')

    # Check if clauses have enough words
    clauses = []
    current_clause = ""

    for part in parts:
        # Remove leading and trailing blanks
        part = part.strip()

        # Count words split by blanks
        word_count = len(part.split())

        if word_count >= 4:
            # Add current clause to list
            if current_clause:
                clauses.append(current_clause.strip())
                current_clause = ""
            clauses.append(part)
        else:
            # Add short clause to current clause
            current_clause += (" " + part)

    # Add last clause
    if current_clause:
        clauses.append(current_clause.strip())

    return clauses


def hohenheimer_readability_index_politics(text: str, language: str = "german") -> float:
    """
    Based on Kercher (2013, p. 384)

    Satz- und Satzteilparameter
        Durchschnittliche Satzteillänge (in Silben) S1 = (x - 12,37) * (100 / (24,12 - 12,37)) 
        Anteil der Satzteile mit mehr als 6 Wörtern S2 = (x - 41,77) * (100 / (67,42 - 41,77)) 
        Durchschnittliche Satzlänge (in Silben) S3 = (x - 22,10) * (100 / (52,59 - 22,10)) 
        Anteil der Sätze mit mehr als 16 Wörtern S4 = (x - 21,90) * (100 / (64,67 - 21,90)) 

        Satzkomplexität = (S1 + S2 + S3 + S4) / 4

    Wortparameter
        Durchschnittliche Wortlänge (in Silben) W1 = (x - 1,936) * (100 / (2,339 – 1,936)) 
        Anteil der Wörter mit mehr als 3 Silben W2 = (x - 10,75) * (100 / (21,21 - 10,75))

        Wortkomplexität = (W1 + W2) / 2
    
    HKPS = (Satzkomplexität + Wortkomplexität) / 2
    
    """


    sentences = nltk.sent_tokenize(text, language=language) # split the text into sentences

    total_clauses = 0 # total number of clauses in the whole text

    total_clause_length_in_syllables = 0 # total number of syllables in the whole text

    clauses_longer_than_6_words = 0 # total number of clauses longer than 6 words in the whole text

    sentences_longer_than_16_words = 0 # total number of sentences longer than 16 words in the whole text

    text_length_in_syllables = 0 # total number of syllables in the whole text
    text_length_in_words = 0 # total number of words in the whole text

    words_longer_than_3_syllables = 0 # total number of words longer than 3 syllables in the whole text
    

    for sentence in sentences:
        clauses = split_sentence_into_clauses(sentence) # use , ; : – — as splitting characters
        total_clauses += len(clauses)

        sentence_length_in_words = 0

        for clause in clauses:

            tokens = text_to_tokens(clause)

            if len(tokens) > 6:
                clauses_longer_than_6_words += 1
            
            sentence_length_in_words += len(tokens)
            text_length_in_words += len(tokens)

            syllable_tokenizer = nltk.tokenize.SyllableTokenizer(lang = "en") # syllable tokenizer is only implemented for English, but as we only want the number of syllables, the results are good enough for German
            
            syllables_per_word = [len(syllable_tokenizer.tokenize(token)) for token in tokens]

            total_clause_length_in_syllables += sum(syllables_per_word)
            text_length_in_syllables += sum(syllables_per_word)

            words_longer_than_3_syllables += len([syllables for syllables in syllables_per_word if syllables > 3])

        if sentence_length_in_words > 16:
            sentences_longer_than_16_words += 1


    # Calculate s1
    average_clause_length = total_clause_length_in_syllables / total_clauses
    s1 = (average_clause_length - 12.37) * (100 / (24.12 - 12.37)) 
    
    # Calculate s2
    proportion_of_clauses_longer_than_6_words = (clauses_longer_than_6_words / total_clauses) * 100 # times 100 because the formula uses percentages
    s2 = (proportion_of_clauses_longer_than_6_words - 41.77) * (100 / (67.42 - 41.77))

    # calculate s3
    average_sentence_length = total_clause_length_in_syllables / len(sentences)
    s3 = (average_sentence_length - 22.10) * (100 / (52.59 - 22.10)) 

    # calculate s4
    proportion_of_sentences_longer_than_16_words = (sentences_longer_than_16_words / len(sentences)) * 100 # times 100 because the formula uses percentages
    s4 = (proportion_of_sentences_longer_than_16_words - 21.90) * (100 / (64.67 - 21.90)) 

    # calculate w1
    average_word_length = text_length_in_syllables / text_length_in_words
    w1 = (average_word_length - 1.936) * (100 / (2.339 - 1.936))

    # calculate w2
    proportion_of_words_longer_than_3_syllables = (words_longer_than_3_syllables / text_length_in_words) * 100 # times 100 because the formula uses percentages
    w2 = (proportion_of_words_longer_than_3_syllables - 10.75) * (100 / (21.21 - 10.75))

    # calculate sentence complexity
    sentence_complexity = (s1 + s2 + s3 + s4) / 4

    # calculate word complexity
    word_complexity = (w1 + w2) / 2

    # calculate HKPS
    hohenheimer_readability_index = (sentence_complexity + word_complexity) / 2

    return hohenheimer_readability_index


def lix_score(text: str, language: str = "english") -> float:
    """
    Calculate the LIX readability index.
    Based on Björnsson (1968)
    LIX = (words / sentences) + (long words * 100) / words
    where long words are words with more than 6 characters.

    The number of sentences is defined as the number of periods, colons, or capital first letters.
    """

    sentences = nltk.sent_tokenize(text, language=language) # split the text into sentences

    words = 0
    long_words = 0

    for sentence in sentences:
        words += len(text_to_tokens(sentence))

        long_words += count_words_longer_than(sentence, 6)

    return (words / len(sentences)) + (long_words * 100) / words
