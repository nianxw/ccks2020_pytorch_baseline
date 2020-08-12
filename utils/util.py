import numpy as np
import torch
import json
import logging
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def f1_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return f1


def acc(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return np.mean((preds == labels).astype(np.float32))


def loss(pair_loss_fc, type_loss_fc, pair_probs, type_out, labels, types):
    # rank_loss
    pair_loss = pair_loss_fc(pair_probs.view(-1), labels)

    # type_loss
    type_loss = type_loss_fc(type_out, types)
    return pair_loss, type_loss


def accuracy(pair_probs, labels):
    preds = pair_probs > 0.5
    preds = preds.view(-1).type(torch.int).cpu().numpy()
    labels = labels.view(-1).type(torch.int).cpu().numpy()
    accu = acc(preds, labels)
    f1 = f1_score(preds, labels)
    return accu, f1


def predict_post_process(predict_res, is_test=False,
                         type_label_map_reverse_path='./data/generated/type_label_map_reverse.json'):
    ent_type_dic = {}
    for ent_info in tqdm(open('./data/basic_data/kb.json', 'r', encoding='utf8')):
        ent_info = json.loads(ent_info.strip())
        subject_id = ent_info['subject_id']
        subject_type = ent_info['type']
        ent_type_dic[subject_id] = subject_type

    logger.info("kb have been loaded")
    type_label_map_reverse = json.load(open(type_label_map_reverse_path, 'r', encoding="utf8"))
    qid_total = predict_res['qid_total']
    left_score_total = predict_res['left_score_total']
    type_prob_total = predict_res['type_prob_total']
    ent_id_total = predict_res['ent_id_total']
    qid_current = qid_total[0][0]
    left_score_qid = []
    type_qid = None
    ent_id_cand = []
    qid_pred = {}
    for qid, left, type_prob, ent_id in zip(
        qid_total, left_score_total,
        type_prob_total, ent_id_total,
    ):
        if qid[0] == qid_current:
            left_score_qid.append(left[0])
            type_qid = np.argmax(type_prob)
            ent_id_cand.append(ent_id[0])
        if qid[0] != qid_current:
            pred_type = type_label_map_reverse[str(type_qid)]
            score = []
            for i in range(len(ent_id_cand)):
                if ent_id_cand[i] == 'NIL':
                    score.append(left_score_qid[i] * 0.5 + 0.4)
                elif pred_type in ent_type_dic[ent_id_cand[i]]:
                    score.append(left_score_qid[i] * 0.5 + 0.5)
                else:
                    score.append(left_score_qid[i] * 0.5)
            pred_ent = ent_id_cand[score.index(max(score))]
            if pred_ent == 'NIL':
                pred_ent = 'NIL_' + pred_type
            qid_pred[qid_current] = pred_ent
            left_score_qid = [left[0]]
            qid_current = qid[0]
            ent_id_cand = [ent_id[0]]
            type_qid = np.argmax(type_prob)
    qid_pred[qid_current] = pred_ent

    if is_test:
        outfile_path = "./data/generated/test_pred.json"
        basic_data_path = "./data/basic_data/test.json"
    else:
        outfile_path = "./data/generated/eval_pred.json"
        basic_data_path = "./data/basic_data/dev.json"
    outfile = open(outfile_path, 'w', encoding="utf8")

    qid = 1
    for line in open(basic_data_path, 'r', encoding='utf8'):
        line_json = json.loads(line.strip())
        mention_data = line_json.get('mention_data')
        mention_data_pred = []
        for item in mention_data:
            kb_id = qid_pred[str(qid)]
            item['kb_id'] = kb_id
            mention_data_pred.append(item)
            qid += 1
        line_json['mention_data'] = mention_data_pred
        outfile.write(json.dumps(line_json, ensure_ascii=False))
        outfile.write('\n')
    logger.info("predict finished")
