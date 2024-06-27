#%%
from tokenizer_tools import train_tokenizer, pretrained_details_from_file, pretrained_details_from_tokenizer, get_unique_vocab_set_dota
if __name__ == '__main__':
    train_tokenizer(mode='bpe', fix_vocab_bool=False, corpus_filename='MERGED_HCLASS_T10_65_corpus', train_vocab_size=200, type='gpt') # train from corpus
