import re

def preprocess_text(text):
    amharic_pattern = re.compile(r'[ሀ-ፐ]+')
    non_amharic_pattern = re.compile(r'[^ሀ-ፐ\s]+')  # Matches anything that's not Amharic characters or whitespace
    # Remove non-Amharic characters and symbols
    text = re.sub(non_amharic_pattern, '', text) 
    # Replace multiple spaces with a single space
    # text = re.sub(r'\s+', ' ', text)
    
    return text

input_file = 'am.txt'
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

cleaned_text = preprocess_text(text)
output_file = 'new.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(cleaned_text)


