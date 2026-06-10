#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import logging
from tqdm import tqdm
from ontology import Ontology

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='../../deepgozero-main/data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
def main(data_root, ont):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    # test_data_file = f'{data_root}/test_data_with_text.pkl'
    test_data_file = '../data/mf/predictions_deepgozero_zero_10_with_text.pkl'
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])

    test_df = pd.read_pickle(test_data_file)

    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    prot_ac_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_ac_index[row.accessions[0]] = i

    dsets = {'test': test_df}

    # BLAST Similarity (Diamond)
    for dset in ('test',):
        diamond_scores_file = f'{data_root}/{ont}/{dset}_diamond.res'
        diamond_scores = {}
        with open(diamond_scores_file) as f:
            for line in f:
                it = line.strip().split()
                if it[0] == it[1]:  # Ignore same proteins (for train)
                    continue
                if it[0] not in diamond_scores:
                    diamond_scores[it[0]] = {}
                diamond_scores[it[0]][it[1]] = float(it[2])

        diam_preds = []
        print(f'Diamond preds for {dset}')
        df = dsets[dset]
        for i, row in tqdm(enumerate(df.itertuples())):
            annots = {}
            prop_annots = {}
            prot_id = row.proteins
            # DiamondScore
            if prot_id in diamond_scores:
                sim_prots = diamond_scores[prot_id]
                allgos = set()
                total_score = 0.0
                for p_id, score in sim_prots.items():
                    allgos |= annotations[prot_index[p_id]]
                    total_score += score
                allgos = list(sorted(allgos))
                sim = np.zeros(len(allgos), dtype=np.float32)
                for j, go_id in enumerate(allgos):
                    s = 0.0
                    for p_id, score in sim_prots.items():
                        if go_id in annotations[prot_index[p_id]]:
                            s += score
                    sim[j] = s / total_score
                for go_id, score in zip(allgos, sim):
                    annots[go_id] = score

                prop_annots = annots.copy()
                for go_id, score in annots.items():
                    for sup_go in go_rels.get_ancestors(go_id):
                        if sup_go in prop_annots:
                            prop_annots[sup_go] = max(prop_annots[sup_go], score)
                        else:
                            prop_annots[sup_go] = score
            diam_preds.append(prop_annots)

        df['diam_preds'] = diam_preds
        filename = f'../data/mf/{dset}_data_diam_with_text.pkl'
        df.to_pickle(filename)


if __name__ == '__main__':
    main()
