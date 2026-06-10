import pandas as pd

def get_test_data_with_text():
    data_path = '../../deepgozero-main/data/'
    test_pred = pd.read_pickle(data_path + 'mf/predictions_deepgozero_zero_10_with_text.pkl')
    # test_pred = pd.read_pickle('../data/mf/predictions_deepgozero_zero_10_with_text.pkl')

    test_text = pd.read_pickle(data_path + 'mf/test_data_with_text.pkl')
    test_pred['uniprot_text'] = test_text['uniprot_text']
    test_pred.to_pickle('../data/mf/predictions_deepgozero_zero_10_with_text.pkl')

get_test_data_with_text()