import pandas as pd


def __load_data(path, drop):
    data = pd.read_csv(path).drop(drop, axis=1)
    data['Comment'] = __process_comments(data['Comment'])

    return data


def __process_comments(comments):
    return comments.str.strip().str.strip('"').str.replace('_', ' ').str.decode('unicode-escape')


def get_train_data():
    data = __load_data('./data/train.csv', ['Date'])
    data = data.append(__load_data('./data/test_with_solutions.csv', ['Date', 'Usage']), ignore_index=True)

    return data['Comment'].values, data['Insult'].values


def get_test_data():
    data = __load_data('./data/impermium_verification_labels.csv', ['id', 'Date', 'Usage'])

    return data['Comment'].values, data['Insult'].values
