import data
from models import LogisticRegression
from sklearn.metrics import roc_auc_score as auc_score


def build_model():
    print "Reading Data"
    comments, labels = data.get_train_data()
    test_comments, test_labels = data.get_test_data()

    print "Building Logistic Regression Pipeline"
    lr_pipeline = LogisticRegression.build_stacked_model()

    print "Training Logistic Regression"
    lr_pipeline.fit(comments, labels)

    print "Predicting Logistic Regression"
    predictions = lr_pipeline.predict_proba(test_comments)

    print auc_score(test_labels, predictions[:, 1])

if __name__ == "__main__":
    build_model()
