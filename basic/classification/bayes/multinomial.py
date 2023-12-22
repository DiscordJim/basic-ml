import regex as re
from math import log


SPAM = [
    'hello I am the nigerian prince, I would like to invite you to supply me with money to fill my coffers and I shall give you the money back afterwards.',
    'this is microsoft support, you will be sentenced to 3000 years in prison if you do not repair your system. call now.',
    'please subscribe to the free Oven tasting competition, entry is $20',
    'hello We are doing a giftcard giveaway :) you simple have to submit $10 and we shall send you the money back',
    'please send me a steam giftcard and i will refund you your money.',
    'there is a large giftcard giveaway taking place this evening, please attend if you can!',
    'hello this is microsoft support, please send money'
]

REAL = [
    'Hello friend, how are you?',
    'It has been too long man! How are you doing? Very excited to hear about your new business proposal to make money',
    'Ever since my wife moved to nigeria it has not been the same, man.'
    'How are you bro?',
    'What do you smell like?',
    'I feel like something smells fishy about this whole deal going down between the Jimmy and the Trumpets.',
    'That is absolutely insane man! I got a new car yesterday and I am so stoked about it, it has been absolutely amazing.',
    'No way man, did you know that there is a new life hack available for lighting barbeques?',
    'I really want to visit Cancun for the break, but I am simply not sure if I have enough time.'
]

def clean_text(string: str) -> str:
    # Uncase the text
    string = string.lower()
    
    # Remove the punctuation
    string = re.sub('\.|\,|\!|\?', '', string)
    return string

def create_mapping(total_words: set, message_set: list) -> dict:
    mappings: dict = {}
    word_count = 0
    for message in message_set:
        # Clean the message
        message = clean_text(message)
        
        # Split using space based tokenization
        words = message.split(' ')
        for word in words:
            word_count += 1
            if word not in total_words:
                total_words.append(word)
            if word not in mappings.keys():
                mappings[word] = 0
            mappings[word] += 1
    return mappings, word_count

word_list = []

spam_words, spam_total = create_mapping(word_list, SPAM)
real_words, real_total = create_mapping(word_list, REAL)

for key in spam_words.keys():
    spam_words[key] /= spam_total
for key in real_words.keys():
    real_words[key] /= real_total
    
# print(spam_words)
    
# Probability of normal / spam messages
p_r = len(REAL) / (len(REAL) + len(SPAM))
p_s = len(SPAM) / (len(REAL) + len(SPAM))
    

SAMPLE = 'wanna join my giftcard giveaway for a chance of visiting the nigerian prince?'
words = clean_text(SAMPLE).split(' ')

probability_fake = log(p_s)
probability_real = log(p_r)
for i in words:
    if i in spam_words.keys():
        probability_fake += log(spam_words[i])
    else:
        probability_fake += log(1 / spam_total)
    if i in real_words.keys():
        probability_real += log(real_words[i])
    else:
        probability_real += log(1 / real_total)

if probability_real > probability_fake:
    print(f'This message is probably real.')
else:
    print(f'This message is probably fake.')
# print(probability_fake,probability_real)
# print(f'The probability this message is real is {probability_real:.3f}. The probability it is fake is {probability_fake:.4f}.')
