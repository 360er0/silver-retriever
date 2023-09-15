import argparse

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_passages(dataset_name, passages_split, passage_prefix, passage_field_separator, use_title):
    passages = load_dataset(dataset_name, name='passages', split=passages_split)
    passages = pd.DataFrame(passages)
    if not use_title:
        passages['title'] = ''

    passages['passage'] = passages.apply(
        lambda r: passage_prefix + r['title'] + passage_field_separator + r['text'], axis=1,
    )
    return passages


def load_pairs(dataset_name, pairs_split, question_prefix):
    pairs = load_dataset(dataset_name, name='pairs', split=pairs_split)
    pairs = pd.DataFrame(pairs)

    pairs = pairs.loc[
        (pairs['passage_id'] != '') & (pairs['relevant'] == True),
        ['question_id', 'question', 'passage_id'],
    ].drop_duplicates()
    pairs['score'] = 1.0

    questions = pairs.drop_duplicates('question_id')[['question_id', 'question']]
    questions['question'] = questions['question'].apply(lambda x: question_prefix + x)

    return pairs, questions


def find_top_passages(emb_questions, emb_passages, normalize_embeddings, top_k):
    if normalize_embeddings:
        emb_questions = normalize(emb_questions)
        emb_passages = normalize(emb_passages)
    similarity = emb_questions @ emb_passages.T

    preds = []
    for qid, sim in tqdm(enumerate(similarity)):
        ids = np.argpartition(sim, -top_k)[-top_k:]

        for score, pid in zip(sim[ids], ids):
            preds.append({
                'question_id': questions['question_id'].values[qid],
                'passage_id': passages['id'].values[pid],
                'score': score,
            })

    preds = pd.DataFrame(preds)
    preds['score'] -= preds['score'].min() - 1e-3  # convert to non-negative scores

    return preds


def acc_at_k(row, k):
    scores = sorted(zip(row['score_true'], row['score_pred']), reverse=True)[:k]
    return any([(t > 0) and (p > 0) for p, t in scores])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates a sentence-transformer model in document retrieval task.')

    # model arguments
    parser.add_argument('--model_name', type=str, default='ipipan/silver-retriever-base-v1',
                        help='a name of the huggingface model which will be used during the evaluation')
    parser.add_argument('--normalize_embeddings', type=str2bool, default=True,
                        help='normalize the model embeddings before retrieval')
    parser.add_argument('--passage_field_separator', type=str, default='</s>',
                        help='a string used to concatenate passage title and passage text')
    parser.add_argument('--question_prefix', type=str, default='Pytanie: ',
                        help='a string added to the question before encoding')
    parser.add_argument('--passage_prefix', type=str, default='',
                        help='a string added to the passage before encoding')

    # dataset arguments
    parser.add_argument('--dataset_name', type=str, default='piotr-rybak/allegro-faq',
                        help='a name of the huggingface dataset which will be used during the evaluation')
    parser.add_argument('--passages_split', type=str, default='test',
                        help='a passages split used to create index, usually there is only one split')
    parser.add_argument('--pairs_split', type=str, default='test',
                        help='a question-passage pairs split')
    parser.add_argument('--use_title', type=str2bool, default=True,
                        help='use the passage titles for encoding the passage')

    # other arguments
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of retrieved passages')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of examples in a single batch during the encoding')

    args = parser.parse_args()
    is_gpu_available = torch.cuda.is_available()

    encoder = SentenceTransformer(args.model_name)
    if is_gpu_available:
        encoder = encoder.half()

    passages = load_passages(args.dataset_name, args.passages_split, args.passage_prefix, args.passage_field_separator, args.use_title)
    pairs, questions = load_pairs(args.dataset_name, args.pairs_split, args.question_prefix)

    emb_passages = encoder.encode(
        passages['passage'].values,
        batch_size=args.batch_size,
        device='cuda' if is_gpu_available else 'cpu',
        show_progress_bar=True,
        normalize_embeddings=False,  # we normalize embeddings later
    ).astype(np.float32)

    emb_questions = encoder.encode(
        questions['question'].values,
        batch_size=args.batch_size,
        device='cuda' if is_gpu_available else 'cpu',
        show_progress_bar=True,
        normalize_embeddings=False,  # we normalize embeddings later
    ).astype(np.float32)

    preds = find_top_passages(emb_questions, emb_passages, args.normalize_embeddings, args.top_k)
    scores = pairs \
        .merge(preds, on=['question_id', 'passage_id'], how='outer', suffixes=('_true', '_pred')) \
        .fillna(0.0) \
        .groupby('question_id').agg(list) \
        .reset_index()

    acc = scores.apply(lambda r: acc_at_k(r, args.top_k), axis=1).mean()
    ndcg = scores.apply(lambda r: ndcg_score([r['score_true']], [r['score_pred']], k=args.top_k), axis=1).mean()

    print(f'Accuracy@{args.top_k}: {acc:.4f}')
    print(f'NDCG@{args.top_k}: {ndcg:.4f}')
