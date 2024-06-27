#%% 
import re
import copy
from collections import Counter
import pandas as pd
from tokenizers import (
                        decoders,
                        models,
                        pre_tokenizers,
                        processors,
                        trainers,
                        Tokenizer,
                       )
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

def setup_tokenizer(mode, corpus_filename, special_tokens, vocab_size):
    """
    Example usage: 
    train_tokenizer(mode='whitespace', fix_vocab_bool=False, corpus_filename='S50_T10_T95_corpus', train_vocab_size=200, type='gpt') # train from corpus

    """
    corpus = f'{corpus_filename}.txt'
    if mode == 'bpe':
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer() # lowercase and strip accents
        tokenizer.model = models.BPE()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                special_tokens=special_tokens) #  max_length= # refers to max length of 1 token
        tokenizer.train([corpus], trainer=trainer)
    elif mode == 'whitespace':
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = trainers.WordLevelTrainer(show_progress=True,
                                special_tokens=special_tokens,
                                vocab_size=vocab_size)
        tokenizer.train([corpus], trainer=trainer)
    else:
        raise ValueError('Tokenizer mode not recognized.')

    return tokenizer

def train_tokenizer(mode, fix_vocab_bool, corpus_filename, train_vocab_size, type):
    '''Build a tokenizer trained on our custom dataset (based on BPE used in GPT-2)

    Args:
    fix_vocab_bool: whether to define the entire vocab based on unique words
    training_vocab_size: if training, define the max vocab size 
    type: 'bert', 'gpt', affects post process

    Usage:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file='tokenizer.json', # You can load from the tokenizer file, alternatively
        pad_token='[PAD]',
        sos_token='[SOS]',
        eos_token='[EOS]',
        sep_token='[SEP]',
        cls_token='[CLS]',
        mask_token='[MASK]',
        unk_token='[UNK]',
        # do not need to explicitly map custom tokens, e.g., EVENT, ACTOR, LOC
    )

    encoding = tokenizer.encode('Let's test this tokenizer.') # encode
    decoding = tokenizer.decode(encoding) # decode

    print(f'Encoding: {encoding}')
    print(f'Decoding: {decoding}')
    '''
    if type == 'gpt':  # Autoregressive tokenizer
        special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[MASK]', '[UNK]']
    elif type == 'bert':  # Bert Tokenizer
        special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[SEP]', '[CLS]', '[MASK]', '[UNK]']
    else:
        raise ValueError('Tokenizer type not recognized.')

    corpus = f'{corpus_filename}.txt'

    if fix_vocab_bool:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer() # lowercase and strip accents
        tokenizer.model = models.BPE()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()

        df_unique_vocab = pd.read_csv('dataset_unique_vocab_set_dota.csv') # Fixed vocabulary
        unique_vocab_list = df_unique_vocab['unique_vocab'].to_list()
        unique_vocab_list = [x for x in unique_vocab_list if type(x) != float] # remove nan

        fixed_vocab = special_tokens + unique_vocab_list
        tokenizer.add_tokens(fixed_vocab)

        print('Vocab list: ', unique_vocab_list)
        print()
        print('Training tokenizer from fixed vocab from unique words from dataset_unique_vocab_set_dota.csv')
        print()
    else:
        tokenizer = setup_tokenizer(mode, corpus_filename, special_tokens, train_vocab_size)      

        print()
        print('Training tokenizer from corpus')
        print()

    cls_token_id = tokenizer.token_to_id('[CLS]')
    sep_token_id = tokenizer.token_to_id('[SEP]')
    pad_token_id = tokenizer.token_to_id('[PAD]')
    sos_token_id = tokenizer.token_to_id('[SOS]')
    eos_token_id = tokenizer.token_to_id('[EOS]')
    unk_token_id = tokenizer.token_to_id('[UNK]')
    mask_token_id = tokenizer.token_to_id('[MASK]')

    print('Special token indices: ')
    print(f'- PAD {pad_token_id}')
    print(f'- SOS {sos_token_id}')
    print(f'- EOS {eos_token_id}')
    print(f'- MASK {mask_token_id}')
    print(f'- UNK {unk_token_id}')
    print(f'- SEP {sep_token_id}')
    print(f'- CLS {cls_token_id}')


    if type == 'bert':  # Bert Tokenizer
        tokenizer.post_processor = processors.TemplateProcessing( # add cls and sep tokens at end and sep sentences
                                                                single=f'[CLS]:0 $A:0 [SEP]:0',
                                                                pair=f'[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1',
                                                                special_tokens=[('[CLS]', cls_token_id), ('[SEP]', sep_token_id)],
                                                                )
    elif type == 'gpt':  # Autoregressive tokenizer
        tokenizer.post_processor = processors.TemplateProcessing(
                                                                single=f'[SOS] $A [EOS]',
                                                                pair=f'[SOS] $A [EOS] $B [EOS]',
                                                                special_tokens=[('[SOS]', sos_token_id), ('[EOS]', eos_token_id)],
                                                            )
    else:
        raise ValueError('Tokenizer type not recognized.')

    # calculate maximum token length for captions
    max_tok_len = 0
    min_tok_len = 10
    max_caption = ''
    min_caption = ''

    token_lens = []
    with open(corpus, 'r') as f:
        for line in f:
            encoding = tokenizer.encode(line)
            token_lens.append(len(encoding.ids))
            if len(encoding.ids) > max_tok_len:
                max_tok_len = len(encoding.ids)
                max_caption = line
       
            if len(encoding.ids) < min_tok_len:
                min_tok_len = len(encoding.ids)
                min_caption = line
    
    # For corpus go through and save each unique sentence and token length
    unique_sentences = set()
    with open(corpus, 'r') as f:
        for line in f:
            unique_sentences.add(line)

    unique_sentences = list(unique_sentences)
    sentence_tok_lens_dict = {}
    for sentence in unique_sentences:
        encoding = tokenizer.encode(sentence)
        sentence_tok_lens_dict[sentence] = len(encoding.ids)
    
    # Sort dictionary by lowest token length to highest
    sentence_tok_lens_dict = dict(sorted(sentence_tok_lens_dict.items(), key=lambda item: item[1]))

    # print token length stats
    print()
    tok_savename = f'tokenizer_dota_A_{str(len(tokenizer.get_vocab()))}_v8.json'
    print()
    print(f'Training tokenizer from corpus and saving to: {tok_savename}')
    print(f'Mean token length: {sum(token_lens)/len(token_lens)}')
    print(f'Maximum token length: {max_tok_len} for caption: {max_caption}')
    print(f'Minimum token length: {min_tok_len} for caption: {min_caption}')
    print('Tokenizer vocab size: ', len(tokenizer.get_vocab()))
    print()
    print('Unique sentences and token lengths:')
    for key in sentence_tok_lens_dict:
        print(f'{key}: {sentence_tok_lens_dict[key]}')

    tokenizer.save(tok_savename)

def unique_phrases_in_dataset(corpus_filepath):  # corpus_filepath e.g., 'dataset_corpus_dota.txt'
    sentence_counts = {}  # Initialize a dictionary to store the sentence counts

    # Process each line in the file
    with open(corpus_filepath, 'r') as file:  # Open the file and read the lines
        for line in file:
            sentence = line.strip().lower()  # Strip leading and trailing whitespace and convert to lowercase for consistency
            if sentence in sentence_counts:  # Increment the count for this sentence in the dictionary
                sentence_counts[sentence] += 1
            else:
                sentence_counts[sentence] = 1

    # Calculate the number of unique sentences
    num_unique_sentences = len(sentence_counts)

    # Sort the sentences by their counts in descending order for better readability
    sorted_sentence_counts = sorted(sentence_counts.items(), key=lambda x: x[1], reverse=True)

    print(f'Number of unique sentences: {num_unique_sentences}')
    print(sorted_sentence_counts)


def get_tokenizer_details(tokenizer, corpus):
    # cls_token_id = tokenizer.cls_token_id
    # sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    print('Special token indices: ')
    print(f'- PAD {pad_token_id}')
    print(f'- BOS {bos_token_id}')
    print(f'- EOS {eos_token_id}')
    # print(f'- SEP {sep_token_id}')
    # print(f'- CLS {cls_token_id}')
    print(f'- MASK {mask_token_id}')
    print(f'- UNK {unk_token_id}')

    # calculate maximum token length for captions
    max_tok_len = 0
    min_tok_len = 10
    max_caption = ''
    min_caption = ''

    token_lens = []
    with open(corpus, 'r') as f:
        for line in f:
            encoding = tokenizer.encode(line)
            print(f'{encoding=}') #*debug
            decoded = tokenizer.decode(encoding) # decode to check if the tokenization is correct
            if decoded != line:
                print('Decoded: ', decoded)
                print('Original: ', line)
                raise ValueError('Decoded tokenization does not match original text.')

            token_lens.append(len(encoding))
            if len(encoding) > max_tok_len:
                max_tok_len = len(encoding)
                max_caption = line
       
            if len(encoding) < min_tok_len:
                min_tok_len = len(encoding)
                min_caption = line

    # print token length stats
    print()
    print(f'Mean token length: {sum(token_lens)/len(token_lens)}')
    print(f'Maximum token length: {max_tok_len} for caption: {max_caption}')
    print(f'Minimum token length: {min_tok_len} for caption: {min_caption}')
    print('Tokenizer vocab size: ', len(tokenizer.get_vocab()))


def pretrained_details_from_file(tokenizer_filepath):
    """
    Get vocab size and tokenization details for custom dataset
    """
    tokenizer = PreTrainedTokenizerFast(
                                        tokenizer_file='{}.json'.format(tokenizer_filepath),
                                        pad_token='[PAD]',
                                        sos_token='[SOS]',
                                        eos_token='[EOS]',
                                        sep_token='[SEP]',
                                        cls_token='[CLS]',
                                        mask_token='[MASK]',
                                        unk_token='[UNK]',
                                        )
    corpus_filepath = 'dataset_corpus_dota.txt'

    get_tokenizer_details(tokenizer, corpus_filepath)


def pretrained_details_from_tokenizer(tokenizer):
    """
    Get vocab size and tokenization details for custom dataset
    """

    corpus_filepath = 'dataset_corpus_dota.txt'

    get_tokenizer_details(tokenizer, corpus_filepath)


# Function to tokenize dataset and count token frequency
def get_token_frequencies(dataset):
    token_freqs = Counter()
    for line in dataset:
        tokens = tokenizer.tokenize(line)
        token_freqs.update(tokens)
    return token_freqs

def tune_tokenizer(tokenizer, tokenizer_name, frequency_threshold=10):
    '''Reduce vocab size of pretrained tokenizer to tune to dataset vocab

    Args:
        tokenizer (_type_): tokenizer to tune
        tokenizer_name: str name of tokenizer for filename save
        frequency_threshold (int, optional): Frequency of token to be included in new vocab. Defaults to 10.
    '''
    df_unique_vocab = pd.read_csv('dataset_unique_vocab_set_dota.csv') # Fixed vocabulary
    dataset = df_unique_vocab['unique_vocab'].to_list()

    # Get token frequencies in the dataset
    token_freqs = get_token_frequencies(dataset)

    # Filter tokens that meet the frequency threshold
    new_vocab = [token for token, freq in token_freqs.items() if freq >= frequency_threshold]

    # Adding special tokens to new vocabulary
    new_vocab.extend([tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token])

    # Create new tokenizer configuration
    new_vocab_file = f'{tokenizer_name}_new_vocab.txt'
    with open(new_vocab_file, 'w') as file:
        file.write('\n'.join(new_vocab))


def get_unique_vocab_set_dota():
    # Load the text file into pandas DataFrames
    savename = 'dota'

    df = pd.read_csv('dataset_corpus_dota.txt', header=None)
    combined_series = df.stack() # Flatten the DataFrame to a Series
    
    # Remove any duplicates to get unique phrases
    unique_phrases = combined_series.drop_duplicates().reset_index(drop=True)
    unique_phrases_filtered = unique_phrases.str.lower().str.replace('.', '', regex=False) # lowercase and remove full stops
    unique_phrases_filtered = unique_phrases_filtered.replace(',', '', regex=False) # remove punctuation (commas)

    # Get unique words from the unique phrases, clean them and remove duplicates
    unique_vocab = pd.Series(' '.join(unique_phrases_filtered).split()).drop_duplicates()
    unique_vocab_list = unique_vocab.to_list()
    final_vocab_list = copy.deepcopy(unique_vocab_list)
    final_vocab_list += ['.', ','] # Add full stops and commas back in

    # Regex pattern for finding special characters (excluding apostrophes)
    special_char_pattern = "[^a-zA-Z0-9'/ -]"
    split_pattern = "(?<=\W)(?<!['/-])\B|\B(?=\W)(?!['/-])"

    for text in unique_vocab_list:
        # Check if the string contains any special characters
        special_chars = re.findall(special_char_pattern, text)
        
        if special_chars:
            word_list = re.split(split_pattern, text) # Remove special characters

            # Filter out empty strings and remove non-alphanumeric characters (excluding apostrophes) from individual words
            processed_words = [re.sub("[^a-zA-Z0-9']", '', word) for word in word_list if word]

            final_vocab_list += special_chars
            final_vocab_list += processed_words

    unique_chars = set()
    for word in final_vocab_list:
        for char in word:
            unique_chars.add(char)
    unique_chars = pd.Series(list(unique_chars)).drop_duplicates()

    final_vocab_list = pd.Series(final_vocab_list).drop_duplicates()

    print(f'Length of unique_phrases: {len(unique_phrases)}, unique_vocab: {len(unique_vocab)}, unique_chars: {len(unique_chars)}')

    max_length = max(len(unique_phrases), len(final_vocab_list), len(unique_chars))
    unique_phrases = unique_phrases.reindex(range(max_length))
    final_vocab_list = final_vocab_list.reindex(range(max_length))
    unique_chars = unique_chars.reindex(range(max_length))

    # Combine unique phrases and words into a DataFrame
    final_df = pd.DataFrame({
                                'unique_phrases': unique_phrases,
                                'unique_vocab': final_vocab_list,
                                'unique_chars': unique_chars
                            })

    # final_df = final_df.dropna()

    # Save the final DataFrame to a new CSV file
    output_csv_path = f'dataset_unique_vocab_set_{savename}.csv'
    final_df.to_csv(output_csv_path, index=False)

    print(f'Step 1. Unique vocab saved to: {output_csv_path}')

    unique_phrases_list = unique_phrases.to_list()

    max_sentence_len = 0
    total_words = 0

    # for each phrase open a text file and print on new line
    for phrase in unique_phrases_list:
        word_count = len(phrase.split())
        max_sentence_len = max(max_sentence_len, word_count)
        total_words += word_count
    
    average_length = total_words / len(unique_phrases_list)
    print(f'Output: Max sentence length {max_sentence_len} Average sentence length: {average_length}')

def get_unique_vocab_set():
    # Load the CSV files into pandas DataFrames
    file_paths = [
        # 'BDD-OIA.csv', # https://twizwei.github.io/bddoia_project/
        # 'BDD-AD.csv',
        # 'BDD-X.csv'
        'DOTA.csv'
    ]

    savename = 'dota'

    dfs = [pd.read_csv(file) for file in file_paths]

    # Combine the first column from both DataFrames into a single series
    combined_series = pd.concat([df.iloc[:, 0] for df in dfs], axis=0)

    # Remove any duplicates to get unique phrases
    unique_phrases = combined_series.drop_duplicates().reset_index(drop=True)
    unique_phrases_filtered = unique_phrases.str.lower().str.replace('.', '', regex=False) # lowercase and remove full stops
    unique_phrases_filtered = unique_phrases_filtered.replace(',', '', regex=False) # remove punctuation (commas)

    # Get unique words from the unique phrases, clean them and remove duplicates
    unique_vocab = pd.Series(' '.join(unique_phrases_filtered).split()).drop_duplicates()
    unique_vocab_list = unique_vocab.to_list()
    final_vocab_list = copy.deepcopy(unique_vocab_list)
    final_vocab_list += ['.', ','] # Add full stops and commas back in

    # Regex pattern for finding special characters (excluding apostrophes)
    special_char_pattern = "[^a-zA-Z0-9'/ -]"
    split_pattern = "(?<=\W)(?<!['/-])\B|\B(?=\W)(?!['/-])"

    for text in unique_vocab_list:
        # Check if the string contains any special characters
        special_chars = re.findall(special_char_pattern, text)
        
        if special_chars:
            word_list = re.split(split_pattern, text) # Remove special characters

            # Filter out empty strings and remove non-alphanumeric characters (excluding apostrophes) from individual words
            processed_words = [re.sub("[^a-zA-Z0-9']", '', word) for word in word_list if word]

            final_vocab_list += special_chars
            final_vocab_list += processed_words

    unique_chars = set()
    for word in final_vocab_list:
        for char in word:
            unique_chars.add(char)
    unique_chars = pd.Series(list(unique_chars)).drop_duplicates()

    final_vocab_list = pd.Series(final_vocab_list).drop_duplicates()

    print(f'Length of unique_phrases: {len(unique_phrases)}, unique_vocab: {len(unique_vocab)}, unique_chars: {len(unique_chars)}')

    max_length = max(len(unique_phrases), len(final_vocab_list), len(unique_chars))
    unique_phrases = unique_phrases.reindex(range(max_length))
    final_vocab_list = final_vocab_list.reindex(range(max_length))
    unique_chars = unique_chars.reindex(range(max_length))

    # Combine unique phrases and words into a DataFrame
    final_df = pd.DataFrame({
                                'unique_phrases': unique_phrases,
                                'unique_vocab': final_vocab_list,
                                'unique_chars': unique_chars
                            })

    # final_df = final_df.dropna()

    # Save the final DataFrame to a new CSV file
    output_csv_path = f'dataset_unique_vocab_set_{savename}.csv'
    final_df.to_csv(output_csv_path, index=False)

    print(f'Step 1. Unique vocab saved to: {output_csv_path}')

    unique_phrases_list = unique_phrases.to_list()

    max_sentence_len = 0
    total_words = 0

    # for each phrase open a text file and print on new line
    output_txt_path = f'dataset_unique_phrases_{savename}.txt'
    for phrase in unique_phrases_list:
        with open(output_txt_path, 'a') as f:
            f.write(phrase + '\n')
        
        word_count = len(phrase.split())
        max_sentence_len = max(max_sentence_len, word_count)
        total_words += word_count
    
    average_length = total_words / len(unique_phrases_list)
    print(f'Step 2. Unique phrases saved to {output_txt_path}')
    print(f'Output: Max sentence length {max_sentence_len} Average sentence length: {average_length}')

def filter_bdd_x():
    # Filter BDD-X for unique phrases
    # Load the CSV file
    file_path = 'BDD-X-Annotations.csv'  # Replace with your file path
    df = pd.read_csv(file_path)

    # Filter columns that match the specified pattern
    pattern = r"Answer\.\d+(action|justification)"
    filtered_columns = df.filter(regex=pattern)

    # Flatten the DataFrame to a Series, drop NaNs, and remove duplicates
    aggregated_series = filtered_columns.stack().dropna().unique()

    # Convert to a Pandas Series for better display and handling
    unique_aggregated_series = pd.Series(aggregated_series)
    unique_aggregated_series.to_csv('BDD-X.csv', index=False)
