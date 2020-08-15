import os
import logging
import time
import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

from models import type_pair_rank
from utils import data_helper
from utils import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def data_to_cuda(batch):
    return_lists = []
    for t in batch:
        if isinstance(t, torch.Tensor):
            return_lists += [t.cuda()]
        else:
            return_lists += [t]
    return return_lists


def batch_forward(batch, model, pair_loss_fc, type_loss_fc):
    query_input_ids, query_sentence_types, query_position_ids, query_masks, \
        left_input_ids, left_sentence_types, left_position_ids, left_masks, \
        right_input_ids, right_sentence_types, right_position_ids, right_masks, \
        labels, types = batch
    pair_probs, type_out, _, _ = model(query_input_ids, query_masks, query_sentence_types, query_position_ids,
                                       left_input_ids, left_masks, left_sentence_types, left_position_ids,
                                       right_input_ids, right_masks, right_sentence_types, right_position_ids)
    pair_loss, type_loss = util.loss(pair_loss_fc,
                                     type_loss_fc,
                                     pair_probs, type_out,
                                     labels, types)

    loss = pair_loss + type_loss
    acc, f1 = util.accuracy(pair_probs, labels)
    return loss, acc, f1


def train(tokenizer, config, args, train_data_set, eval_data_set=None):
    # 获取数据
    num_train_optimization_steps = int(len(train_data_set) / args.train_batch_size) * args.epochs
    if args.local_rank != -1 and args.use_cuda:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        train_data_loader = DataLoader(dataset=train_data_set,
                                       batch_size=args.train_batch_size,
                                       sampler=DistributedSampler(train_data_set),
                                       collate_fn=data_helper.collate_fn)
        if args.do_eval:
            eval_data_loader = DataLoader(dataset=eval_data_set,
                                          batch_size=args.eval_batch_size,
                                          sampler=DistributedSampler(eval_data_set),
                                          collate_fn=data_helper.collate_fn)
    else:
        train_data_loader = DataLoader(dataset=train_data_set,
                                       batch_size=args.train_batch_size,
                                       num_workers=8,
                                       collate_fn=data_helper.collate_fn)
        if args.do_eval:
            eval_data_loader = DataLoader(dataset=eval_data_set,
                                          batch_size=args.eval_batch_size,
                                          num_workers=8,
                                          collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    sentence_encoder = BertModel.from_pretrained(args.init_checkpoint, config=config)
    model = type_pair_rank.TypePairRank(sentence_encoder, args)

    if args.use_cuda:
        model.cuda()
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
            model = nn.DataParallel(model)

    if args.load_path is not None and os.path.exists(args.load_path):
        ckpt = args.load_path
        # steps = int(re.search("\d+", ckpt).group())
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            # name = name.replace("module.", "")
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        logger.info("Successfully loaded checkpoint '%s'" % ckpt)

    # prepare optimizer
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
        {'params': [p for n, p in parameters_to_optimize 
            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters_to_optimize
            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(parameters_to_optimize, lr=args.learning_rate, correct_bias=False)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()

    log_loss = 0.0
    log_acc = 0.0
    log_f1 = 0.0

    pair_loss_fc = nn.BCELoss()
    type_loss_fc = nn.CrossEntropyLoss()
    begin_time = time.time()
    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)
            loss, acc, f1 = batch_forward(batch, model, pair_loss_fc, type_loss_fc)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log_loss += loss.data.item()
            log_acc += acc
            log_f1 += f1

            if steps % args.log_steps == 0:
                end_time = time.time()
                used_time = end_time - begin_time
                logger.info(
                    "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                    "ave acc: %f, ave f1: %f speed: %f s/step" %
                        (
                            epoch, steps, num_train_optimization_steps,
                            steps, log_loss / args.log_steps, log_acc / args.log_steps,
                            log_f1 / args.log_steps, used_time / args.log_steps,
                        ),
                )
                begin_time = time.time()
                log_loss = 0.0
                log_acc = 0.0
                log_f1 = 0.0

            if steps % args.validation_steps == 0:
                if args.do_eval:
                    model.eval()

                    eval_total_loss = 0.0
                    eval_total_acc = 0.0
                    eval_total_f1 = 0.0

                    eval_steps = 0
                    eval_begin_time = time.time()
                    for batch_eval in eval_data_loader:
                        eval_steps += 1
                        if args.use_cuda:
                            batch_eval = data_to_cuda(batch_eval)
                        eval_loss, eval_acc, eval_f1 = batch_forward(batch_eval, model, pair_loss_fc, type_loss_fc)
                        eval_total_loss += eval_loss.data.item()
                        eval_total_acc += eval_acc
                        eval_total_f1 += eval_f1
                    eval_end_time = time.time()
                    logger.info("***** Running evalating *****")
                    logger.info(
                        "eval result —— epoch: %d, ave eval loss: %f, "
                        "ave eval acc: %f, ave eval f1: %f, "
                        "eval used time: %.6f " %
                        (
                            epoch,  eval_total_loss / eval_steps, eval_total_acc / eval_steps,
                            eval_total_f1 / eval_steps, eval_end_time - eval_begin_time,
                        ),
                    )
                    logger.info("*****************************")
                    model.train()
            if steps % args.save_steps == 0:
                if args.local_rank == 0 or args.local_rank == -1:
                    torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_path, 'model_%d.bin' % steps))
        if args.local_rank == 0 or args.local_rank == -1:
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_path, 'model_%d.bin' % steps))
    torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_path, 'model_final.bin'))


def test(tokenizer, config, args, dataset):
    num_test_steps = int(len(dataset) / args.eval_batch_size)

    sentence_encoder = BertModel(config=config)
    model = type_pair_rank.TypePairRank(sentence_encoder, args)
    if args.use_cuda:
        model = nn.DataParallel(model)
        model.cuda()

    ckpt = args.load_path
    state_dict = torch.load(ckpt)["state_dict"]
    own_state = model.state_dict()
    for name, param in state_dict.items():
        # name = name.replace("module.", "")
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    logger.info("Successfully loaded checkpoint '%s'" % ckpt)

    test_data_loader = DataLoader(dataset=dataset,
                                  batch_size=args.eval_batch_size,
                                  num_workers=8,
                                  collate_fn=data_helper.collate_fn)

    logger.info("***** Running predicting *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Num steps = %d", num_test_steps)

    model.eval()
    qid_total = None
    left_score_total = None
    type_prob_total = None
    ent_id_total = None
    for batch in tqdm(test_data_loader):
        if args.use_cuda:
            batch = data_to_cuda(batch)
        query_input_ids, query_sentence_types, query_position_ids, query_masks, \
            left_input_ids, left_sentence_types, left_position_ids, left_masks, \
            right_input_ids, right_sentence_types, left_position_ids, right_masks, \
            qids, ent_ids = batch
        _, type_out, left_logits, right_loguts = model(query_input_ids, query_masks, query_sentence_types, query_position_ids,
                                            left_input_ids, left_masks, left_sentence_types, left_position_ids,
                                            right_input_ids, right_masks, right_sentence_types, left_position_ids)
        left_probs = F.sigmoid(left_logits).data.view(-1).cpu().numpy()  # [batch_size]
        type_probs = F.softmax(type_out, dim=-1).data.cpu().numpy()  # [batch_size, 24]

        ent_ids = ent_ids.reshape(ent_ids.shape[0], -1)
        if ent_id_total is None:
            ent_id_total = ent_ids
        else:
            ent_id_total = np.concatenate((ent_id_total, ent_ids), axis=0)

        qids = qids.reshape(qids.shape[0], -1)
        if qid_total is None:
            qid_total = qids
        else:
            qid_total = np.concatenate((qid_total, qids), axis=0)\

        left_probs = left_probs.reshape(left_probs.shape[0], -1)
        if left_score_total is None:
            left_score_total = left_probs
        else:
            left_score_total = np.concatenate(
                (left_score_total, left_probs), axis=0,
            )

        type_probs = type_probs.reshape(type_probs.shape[0], -1)
        if type_prob_total is None:
            type_prob_total = type_probs
        else:
            type_prob_total = np.concatenate(
                (type_prob_total, type_probs), axis=0,
            )

    predict_res = {}
    predict_res['qid_total'] = qid_total
    predict_res['left_score_total'] = left_score_total
    predict_res['type_prob_total'] = type_prob_total
    predict_res['ent_id_total'] = ent_id_total
    util.predict_post_process(predict_res)