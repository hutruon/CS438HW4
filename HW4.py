import re
import math
from collections import defaultdict

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line)

def split_data(lines, split_ratio=0.8):
    split_index = int(split_ratio * len(lines))
    return lines[:split_index], lines[split_index:]

def preprocess_line(line):
    line = re.sub(r"[^\w\s]", '', line)
    line = re.sub(r"\s+", ' ', line).strip()
    return line.lower()

def heuristic_split(s, word_set):
    for end in range(len(s), 0, -1):
        if s[:end].lower() in word_set:
            return [s[:end]] + heuristic_split(s[end:], word_set)
    return [s] if s else []

def preprocess_message(message, word_set):
    message = preprocess_line(message)
    words = message.split()
    processed_words = []
    for word in words:
        processed_words.extend(heuristic_split(word, word_set))
    return processed_words

def tokenize_messages(lines, word_set):
    tokens = defaultdict(lambda: {'ham': 0, 'spam': 0})
    total_ham = total_spam = 0

    for line in lines:
        label, text = line.split("\t")
        words = set(preprocess_message(text, word_set))
        for word in words:
            tokens[word][label] += 1
            if label == 'ham':
                total_ham += 1
            else:
                total_spam += 1

    return tokens, total_ham, total_spam

def calculate_spamminess(probs, threshold=1.0):
    spamminess_scores = {}
    for word, prob in probs.items():
        if prob['ham'] == 0:
            spamminess_scores[word] = float('inf')
        else:
            spamminess = prob['spam'] / prob['ham']
            if spamminess > threshold:
                spamminess_scores[word] = spamminess
    return spamminess_scores

def print_top_spam_words(spamminess_scores, top_n=20):
    top_words = sorted(spamminess_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("\nTop words indicative of spam:")
    for word, score in top_words:
        print(f"{word}: {score}")

def calculate_probabilities(tokens, total_ham, total_spam):
    probs = defaultdict(lambda: {'ham': 0.5, 'spam': 0.5})
    for token, counts in tokens.items():
        probs[token]['ham'] = math.log((counts['ham'] + 1) / (total_ham + 2))
        probs[token]['spam'] = math.log((counts['spam'] + 1) / (total_spam + 2))
    return probs

def classify(message, probs, word_set):
    words = set(preprocess_message(message, word_set))
    spam_log_prob = ham_log_prob = 0.0
    for word in words:
        if word in probs:
            spam_log_prob += probs[word]['spam']
            ham_log_prob += probs[word]['ham']
    return 'spam' if spam_log_prob > ham_log_prob else 'ham'

# Load the English words from file
with open('english_words.txt', 'r', encoding='utf-8') as f:
    english_words = set(f.read().splitlines())

# Read the data from the file
lines = read_file('SMSSpamCollection.txt')

# Split the data into training and test sets
training_data, test_data = split_data(lines)

# Write the training and test data to separate files
write_to_file('training_data.txt', training_data)
write_to_file('test_data.txt', test_data)

# Tokenize the messages and calculate word frequencies
tokens, total_ham, total_spam = tokenize_messages(training_data, english_words)

# Calculate the probability of each word being in a spam or ham message using logs
probs = calculate_probabilities(tokens, total_ham, total_spam)

# Evaluate the model using the test data
true_positives = true_negatives = false_positives = false_negatives = 0

for line in test_data:
    true_label, text = line.split("\t")
    predicted_label = classify(text, probs, english_words)

    if true_label == 'spam' and predicted_label == 'spam':
        true_positives += 1
    elif true_label == 'ham' and predicted_label == 'ham':
        true_negatives += 1
    elif true_label == 'spam' and predicted_label == 'ham':
        false_negatives += 1
    elif true_label == 'ham' and predicted_label == 'spam':
        false_positives += 1
# Calculate the metrics for spam
spam_precision = true_positives / (true_positives + false_positives)
spam_recall = true_positives / (true_positives + false_negatives)

# Calculate the metrics for ham
ham_precision = true_negatives / (true_negatives + false_negatives)
ham_recall = true_negatives / (true_negatives + false_positives)

# Calculate the F-Scores for spam and ham
spam_f_score = 2 * (spam_precision * spam_recall) / (spam_precision + spam_recall)
ham_f_score = 2 * (ham_precision * ham_recall) / (ham_precision + ham_recall)

# Calculate the overall accuracy
accuracy = (spam_recall + ham_recall) / 2

# Print out the metrics
print(f"Spam Precision: {spam_precision}")
print(f"Spam Recall: {spam_recall}")
print(f"Ham Precision: {ham_precision}")
print(f"Ham Recall: {ham_recall}")
print(f"Spam F-Score: {spam_f_score}")
print(f"Ham F-Score: {ham_f_score}")
print(f"Accuracy: {accuracy}")

# Calculate and print the top spam-indicative words
spamminess_scores = calculate_spamminess(probs)
print_top_spam_words(spamminess_scores)
