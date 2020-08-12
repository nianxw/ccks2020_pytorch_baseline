import os
import json
import logging
import pickle
import numpy as np
import torch
from torch.utils import data
from collections import namedtuple

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def csv_reader(fd, delimiter='\t'):
    """csv_reader"""
    def gen():
        """gen"""
        for i in fd:
            yield i.rstrip('\n').split(delimiter)
    return gen()


class type_pair_dataset(data.Dataset):

    def __init__(self, input_file, tokenizer, max_seq_len, label_map_path, 
                 is_train=True, shuffle=True):
        with open(label_map_path, encoding='utf8') as f:
            self.label_map = json.load(f)
        self.is_train = is_train
        self.shuffle = shuffle
        self.data = self.load_data(input_file)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

    def load_data(self, input_file):
        cache_file = input_file.replace('.txt', '_data.pkl')
        if os.path.exists(cache_file):
            logger.info("loading data from cache file: %s" % cache_file)
            return pickle.load(open(cache_file, 'rb'))
        else:
            logger.info("loading data from input file: %s" % input_file)
            with open(input_file, 'r', encoding='utf8') as f:
                reader = csv_reader(f)
                headers = next(reader)
                text_indices = [
                    index for index, h in enumerate(headers) if h != "label"
                ]

                i = 1
                examples = []
                for line in reader:
                    if i % 100000 == 0:
                        logger.info("%d examples have been loaded" % i)
                    for index, text in enumerate(line):
                        if index in text_indices:
                            line[index] = text.replace(' ', '')
                    text_a = line[2]
                    text_b = line[3]
                    text_c = line[4]
                    if self.is_train:
                        Record = namedtuple('Record', ['text_a', 'text_b', 'text_c', 'label_id', 'type_id'])
                        label_id = line[5]
                        type_id = self.label_map[line[6]]
                        example = Record(
                                text_a=text_a,
                                text_b=text_b,
                                text_c=text_c,
                                label_id=label_id,
                                type_id=type_id,)
                    else:
                        Record = namedtuple('Record', ['text_a', 'text_b', 'text_c', 'qid', 'ent_id'])
                        qid = line[0]
                        ent_id = line[7]
                        example = Record(
                                text_a=text_a,
                                text_b=text_b,
                                text_c=text_c,
                                qid=qid,
                                ent_id=ent_id,)

                    examples.append(example)
                    i += 1

                if self.shuffle:
                    np.random.shuffle(examples)
            # pickle.dump(examples, open(cache_file, 'wb'))
            return examples

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[: self.max_seq_len - 2]

        tokens = [self.cls_token] + tokens + [self.sep_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def __getitem__(self, idx):
        example = self.data[idx]
        token_a = self.tokenize(example.text_a)
        token_b = self.tokenize(example.text_b)
        token_c = self.tokenize(example.text_c)
        return example, token_a, token_b, token_c

    def __len__(self):
        return len(self.data)


def pad_seq(insts):
    return_list = []

    max_len = max(len(inst) for inst in insts)

    # input ids
    inst_data = np.array(
        [inst + list([0] * (max_len - len(inst))) for inst in insts],
    )
    return_list += [inst_data.astype("int64")]

    # input sentence type
    return_list += [np.zeros_like(inst_data).astype("int64")]

    # input position
    inst_pos = np.array([list(range(0, len(inst))) + [0] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_pos.astype("int64")]

    # input mask
    input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
    return_list += [input_mask_data.astype("float32")]

    return return_list


def collate_fn(data):
    examples, tokens_a, tokens_b, tokens_c = zip(*data)

    padded_token_query_ids, padded_text_type_query_ids, padded_position_query_ids, input_query_mask = pad_seq(tokens_a)
    padded_token_left_ids, padded_text_type_left_ids, padded_position_left_ids, input_left_mask = pad_seq(tokens_b)
    padded_token_right_ids, padded_text_type_right_ids, padded_position_right_ids, input_right_mask = pad_seq(tokens_c)

    return_list = [
        padded_token_query_ids, padded_text_type_query_ids, padded_position_query_ids, input_query_mask,
        padded_token_left_ids, padded_text_type_left_ids, padded_position_left_ids, input_left_mask,
        padded_token_right_ids, padded_text_type_right_ids, padded_position_right_ids, input_right_mask,
    ]
    # train & eval
    if 'label_id' in examples[0]._fields:
        batch_labels = [float(example.label_id) for example in examples]
        return_list += [batch_labels]
    if 'type_id' in examples[0]._fields:
        batch_types = [int(example.type_id) for example in examples]
        return_list += [batch_types]
    return_list = [torch.tensor(batch_data) for batch_data in return_list]

    # predict
    if 'qid' in examples[0]._fields:
        batch_qids = [example.qid for example in examples]
        return_list += [np.array(batch_qids)]
    if 'ent_id' in examples[0]._fields:
        batch_ent_ids = [example.ent_id for example in examples]
        return_list += [np.array(batch_ent_ids)]
    return return_list