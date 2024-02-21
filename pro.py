from itertools import groupby
from best_profanity import predict

def remove_repeated_letters(word):
    # Remove repeated letters, e.g., "clouudd" becomes "cloud"
    return ''.join(ch for ch, _ in groupby(word))

def process_sentence(sentence):
    # Split the sentence into words
    words = sentence.split()

    processed_words = [remove_repeated_letters(word) for word in words]

    # Join the processed words back into a sentence
    processed_sentence = ' '.join(processed_words)

    return processed_sentence

def has_profanity(sentence):
    # Add custom profanity words for Tagalog
    tagalog_profanity = ["putangina", "gago", "tangina", "bobo", "inutil", "tanga", "tanginamo", "pakyu", "puta", "pokpok", "bilat", "tite"]

    # Combine custom profanity words with the existing profanity check
    all_profanity_words = tagalog_profanity + predict.get_all_profanity_words()

    # Check for profanity using the combined list of words
    return predict([sentence], profanity_words=all_profanity_words)[0] > 0.5
