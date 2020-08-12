import argparse
import logging
import os
import random

import numpy as np
import torch

from transformers import BertConfig, BertTokenizer

from utils import data_helper, train_helper

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # 路径

    # 1. 训练和测试数据路径
    parser.add_argument("--train_set", default='./data/generated/train.txt', type=str, help="Path to training data.")
    parser.add_argument("--dev_set", default='./data/generated/dev.txt', type=str, help="Path to dev data.")
    parser.add_argument("--test_set", default='./data/generated/dev_test.txt', type=str, help="Path to test data.")
    parser.add_argument("--label_map_path", default='./data/generated/type_label_map.json', type=str, help="label_map_path")

    # 2. 预训练模型路径
    parser.add_argument("--vocab_file", default="./pretrain/vocab.txt", type=str, help="Init vocab to resume training from.")
    parser.add_argument("--config_path", default="./pretrain/bert_config.json", type=str, help="Init config to resume training from.")
    parser.add_argument("--init_checkpoint", default="./pretrain/pytorch_model.bin", type=str, help="Init checkpoint to resume training from.")

    # 3. 保存模型
    parser.add_argument("--save_path", default="./check_points", type=str, help="Path to save checkpoints.")
    parser.add_argument("--load_path", default="./check_points/model_final.bin", type=str, help="Path to load checkpoints.")

    # 模型参数
    parser.add_argument("--hidden_size", default=768, type=int, help="model hidden size")

    # 训练和测试参数
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to perform training.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to perform evaluation on dev data set.")
    parser.add_argument("--do_predict", default=True, type=bool, help="Whether to perform evaluation on test data set.")

    parser.add_argument("--epochs", default=2, type=int, help="Number of epoches for fine-tuning.")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Total examples' number in batch for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total examples' number in batch for eval.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="Number of words of the longest seqence.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate used to train with warmup.")
    parser.add_argument("--warmup_proportion", default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--log_steps",
                        type=int,
                        default=10,
                        help="The steps interval to print loss.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=1000,
                        help="The steps interval to save model")
    parser.add_argument("--validation_steps",
                        type=int,
                        default=2000,
                        help="The steps interval to evaluate model performance.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.use_cuda:
        if args.local_rank == -1:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
    else:
        device = torch.device("cpu")
        n_gpu = 0
    logger.info("device: {}, n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    bert_tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
    bert_config = BertConfig.from_pretrained(args.config_path)

    # 获取数据
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if args.do_train:
        logger.info("loading train dataset")
        train_dataset = data_helper.type_pair_dataset(args.train_set, bert_tokenizer, 
                                                      max_seq_len=args.max_seq_len, 
                                                      label_map_path=args.label_map_path)

    if args.do_eval:
        logger.info("loading eval dataset")
        eval_dataset = data_helper.type_pair_dataset(args.dev_set, bert_tokenizer, 
                                                     max_seq_len=args.max_seq_len, 
                                                     label_map_path=args.label_map_path,
                                                     shuffle=False)

    if args.do_predict:
        logger.info("loading test dataset")
        test_dataset = data_helper.type_pair_dataset(args.test_set, bert_tokenizer,
                                                     max_seq_len=args.max_seq_len,
                                                     label_map_path=args.label_map_path,
                                                     is_train=False,
                                                     shuffle=False)

    if args.do_train:
        train_helper.train(bert_tokenizer, bert_config, args, train_dataset, eval_dataset)

    if args.do_predict:
        train_helper.test(bert_tokenizer, bert_config, args, test_dataset)


if __name__ == "__main__":
    main()
