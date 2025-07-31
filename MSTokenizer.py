import torch
import json

class GlycoBertTokenizer:
    def __init__(self, vocab_list, max_seq_length=512):
        # BERT's special tokens
        self.special_tokens = {
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'unk_token': '[UNK]',
            'mask_token': '[MASK]'
        }
        
        # List of special token symbols
        special_token_symbols = list(self.special_tokens.values())

        # Filter out special tokens from vocab_list to prevent duplicates
        vocab_list = [word for word in vocab_list if word not in special_token_symbols]

        # Create a combined list of special tokens and vocab_list
        combined_list = special_token_symbols + vocab_list

        # Create vocab and reverse vocab dictionaries
        self.vocab = {word: idx for idx, word in enumerate(combined_list)}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.max_seq_length = max_seq_length

    def tokenize(self, text):
        return text.split()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        batch_token_ids = []
        batch_attention_masks = []
    
        for text in texts:
            tokens = self.tokenize(text)
            token_ids = [self.vocab.get(token, self.vocab[self.special_tokens['unk_token']]) for token in tokens]

            # Prepend [CLS] token and append [SEP] token
            token_ids = [self.vocab[self.special_tokens['cls_token']]] + token_ids + [self.vocab[self.special_tokens['sep_token']]]

            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Padding or truncating to the max_seq_length
            if len(token_ids) < self.max_seq_length:
                padding_length = self.max_seq_length - len(token_ids)
                token_ids += [self.vocab[self.special_tokens['pad_token']]] * padding_length
                attention_mask += [0] * padding_length
            else:
                token_ids = token_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]

            batch_token_ids.append(torch.tensor(token_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))

        return {
            "token_ids": torch.stack(batch_token_ids),
            "attention_mask": torch.stack(batch_attention_masks)
        }

    def decode(self, batch_token_ids, skip_special_tokens=False):
        if batch_token_ids.dim() == 1:
            batch_token_ids = batch_token_ids.unsqueeze(0)

        decoded_texts = []
        for token_ids in batch_token_ids:
            if skip_special_tokens:
                decoded_texts.append(' '.join([self.reverse_vocab[token_id.item()] for token_id in token_ids if token_id.item() not in [self.vocab[val] for val in self.special_tokens.values()]]))
            else:
                decoded_texts.append(' '.join([self.reverse_vocab[token_id.item()] for token_id in token_ids if token_id.item() != self.vocab[self.special_tokens['pad_token']]]))

        return decoded_texts if len(decoded_texts) > 1 else decoded_texts[0]

    
    def save_vocabulary(self, path="vocab.json"):
        with open(path, 'w') as file:
            json.dump(self.vocab, file)

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    @classmethod
    def load_vocabulary(cls, path="vocab.json", max_seq_length=512):
        with open(path, 'r') as file:
            loaded_vocab = json.load(file)
        return cls(list(loaded_vocab.keys()), max_seq_length) 


