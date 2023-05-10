import argparse
import math
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--random-seed', type=int, default=1,
                    help='set random seed for replicability; 0 implies truly random seed')
parser.add_argument('--source', type=str, required=True)
parser.add_argument('-N', type=int, default=3)
parser.add_argument('--start_symbol', type=str, default='<S>')
parser.add_argument('--end_symbol', type=str, default='</S>')
parser.add_argument('--text', type=str, default='')
parser.add_argument('--split', type=float, default=1.0)
parser.add_argument('--file', type=str)

# This file contains the code for 7.1 and 7.2

def ngrams_from_text(line, N, start_symbol, end_symbol):
    # Define the inital prefix, containing the start symbols
    prefix = [start_symbol for _ in range(N-1)] 
    # Loop through each symbol of the line
    for i in range(len(line)):
        # Yield the prefix and the current symbol
        yield prefix, line[i]
        prefix = prefix[1:] + [line[i]]
    # Yield the last prefix and the end symbol
    yield prefix, end_symbol


def get_base_node(model, prefix):
    node = model
    for c in prefix:
        if c not in node:
            node[c] = {}
        node = node[c]
    return node


def add_ngram_to_model(model, prefix, last):
    base = get_base_node(model, prefix)
    if last not in base:
        base[last] = 0
    base[last] += 1


def generate(model, start):
    start = list(start)
    while True:
        base = get_base_node(model, start)
        chars, counts = zip(*base.items())
        # Compute the probability of each symbol
        probabilities = [count / sum(counts) for count in counts]  # Compute probabilities
        char = random.choices(population=chars, weights=probabilities, k=1)[0]
        start = start[1:] + [char]
        yield char, probabilities[chars.index(char)]  # Yield symbol and its probability


def estimate_model(args):
    model = {}
    with open(args.source, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines_read = 0
        # Define number of lines to be read from split
        split = int(len(lines) * args.split)
        # Loop through each line of the file
        for line in lines:
            # Remove the trailing newline
            line = line.strip()
            # Get the ngrams from the line
            ngram_source = ngrams_from_text(line, args.N, args.start_symbol, args.end_symbol)
            for ngram in ngram_source:
                add_ngram_to_model(model, ngram[0], ngram[1])
            # Increment the number of lines read and check if there have been read more lines that specified for training
            lines_read += 1
            if lines_read > split:
                break    
    return model

# Calculate the cross-entropy and perplexity of the model
def calculate_cross_entropy(model, text, N, start_symbol):
    # Define the initial prefix, containing the start symbols
    prefix = [start_symbol for _ in range(N - 1)]
    log_sum = 0.0
    symbols = 0
    # Loop through each symbol of the text
    for symbol in text:
        # Get the base node of the prefix
        base = get_base_node(model, prefix)
        # If the symbol is in the base node
        if symbol in base:
            # Compute the probability of each symbol
            probabilities = [count / sum(base.values()) for count in base.values()]
            # find the index of the symbol in the base node
            symbol_index = list(base.keys()).index(symbol)
            # Add the log probability of the symbol to the log sum
            log_sum += math.log(probabilities[symbol_index])
            # Increment the number of symbols
            symbols += 1
        # Update the prefix
        prefix = prefix[1:] + [symbol]
    # If there are symbols
    if symbols > 0:
        # Compute the cross-entropy and perplexity
        cross_entropy = -log_sum / symbols
        perplexity = 2 ** cross_entropy
        return cross_entropy, perplexity
    else:
        return None, None


if __name__ == '__main__':
    args = parser.parse_args()
    #Define start symbol
    start = [args.start_symbol for _ in range(args.N - 1)]
    model = estimate_model(args)
    # Input text
    text = ""
    # Set text to the source file
    with open(args.source, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    # If there was a split specified, split the text into the specified amount for train/test    
    if args.split < 1.0:
        split = int(len(text) * args.split)
        text = text[:split]
    # If there was a text specified, set the text to the specified text    
    if args.text != "":
        text = args.text.strip()
    # If there was a file specified, set the text to the text in the file    
    if args.file != None:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()  

    generated_text = ''
    print("Generated text: ")
    # Generate text
    for symbol, probability in generate(model, start):
        if symbol == args.end_symbol:
            break
        generated_text += symbol
    print(generated_text)
    # Calculate cross-entropy and perplexity
    cross_entropy, perplexity = calculate_cross_entropy(model, text, args.N, args.start_symbol)
    if cross_entropy is not None and perplexity is not None:
        print("Cross-Entropy:", cross_entropy)
        print("Perplexity:", perplexity)
