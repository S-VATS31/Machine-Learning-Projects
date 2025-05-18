# Encode raw text into raw bytes
raw_bytes = raw_text.encode("utf-8")
# Encode bytes into tokens using mapping
tokens = list(map(int, raw_bytes))

# Print lengths
print(f"Raw Text Length:   {len(raw_text)}") # Number of characters
print(f"Raw Bytes Length:  {len(raw_bytes)}") # Number of UTF-8 bytes; NOTE: 1 character can be encoded into 1-4 bytes
print(f"Raw Tokens Length: {len(tokens)}") # Number of Tokens (1 byte per token)

def count_occurrences(tokens):
    """
    Counts the number of occurrences for each adjacent pair

    Args:
        tokens (list): List of unicode code points based on characters from raw_text.

    Returns:
        counter (dict): Dictionary with adjacent pair code points and number of occurrences.
            Keys (tuple): Unicode points of adjacent pairs.
            Values (int): Number of occurrences
    """
    counter = {} # Initalize counter
    for adj_pair in zip(tokens, tokens[1:]): # Creates adjacent pairing. Example: ('a', 'b'), ('b','c'), ('c','d')
        if adj_pair in counter:
            counter[adj_pair] += 1 # Add 1 to the number of occurrences
        else:
          counter[adj_pair] = 1 # Set all occurrences to 1, and add if found again.
    return counter

# Call function
pairs = count_occurrences(tokens)

# Loop through counter
for k, v in sorted(pairs.items(), key=lambda item: item[1], reverse=True): # Prints from largest frequency to lowest frequency
    print(f"Adjacent Pair: {k} | Frequency: {v}")

def merge_tokens(tokens, target_pair, new_token):
    """
    Iteratively merge tokens with the largest frequency.

    Args:
        tokens (list): List of unicode code points based on characters from raw_text.
        target_pair (tuple): The adjacent pair being merged. Example: ('a', 'b') -> ('c').

    Returns:
        merged_tokens (list): List of tokens successfully merged.
    """
    merged_tokens = [] # Initialize list
    i = 0
    # Iterate through all tokens
    while i < len(tokens):
        # Ensure merging is eligible with no errors
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == target_pair:
            merged_tokens.append(new_token) # Append merged token to new list
            i += 2 # Skip to the next adjacent pair (2 tokens)
        else:
            merged_tokens.append(tokens[i]) # Unmerged tokens are preserved
            i += 1 # Skip one single token
    return merged_tokens

num_merges = 1500 # Number of merges occurring
original_tokens = list(tokens) # Backup copy of original tokens

merges = {}
for i in range(num_merges):
    pairs = count_occurrences(tokens) # Counts number of occurrences per adjacent pair

    if not pairs:
        break # Exit early if no pairs are left to merge

    # Select the most frequent pair to merge
    max_pair = max(pairs, key=pairs.get)
    max_freq = pairs[max_pair] # Get the frequency of the most frequent pair

    new_token = 1378 + i # New token ID
    print(f"{max_pair} with frequency {max_freq} has been merged into {new_token}")

    # Perform merge and update tokens
    tokens = merge_tokens(tokens, max_pair, new_token)
    merges[max_pair] = new_token
