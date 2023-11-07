# english_word_list_generator.py
import nltk

# Ensure that the NLTK package is installed and the required resources are downloaded
nltk.download('words')

def generate_english_word_list(filename='english_words.txt'):
    from nltk.corpus import words
    word_list = words.words()
    with open(filename, 'w') as file:
        for word in word_list:
            file.write(word + '\n')

# Call the function to generate the word list
generate_english_word_list()