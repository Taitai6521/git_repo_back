import pandas as pd
import numpy as np

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold



from load_data import load_train_data, load_test_data




logger = getLogger(__name__)



DIR = 'result_tmp/'
if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)


    logger.info('start')

    df = load_train_data()



    　
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    logger.info('train end')

    df = load_test_data()

    x_test = df[use_cols].sort_values('id')



    logger.debug('test data load end {}'.format(x_test.shape))
    logger.info('test data load end {}'.format(x_test.shape))

    pred_test = clf.predict_proba(x_test)



    def_submit = pd.read_csv(SAMPLE_SUBMIT_FILE)
    df['target'] = pred_test
    df_submit.to_csv(DIR + 'submit.csv')

    for train_idx, valid_idx in cv.split(x_train, y_train)



    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
