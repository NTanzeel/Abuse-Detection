import data
from models import LogisticRegression as lr


def build_model():
    print "Reading Data"
    comments, labels = data.get_train_data()
    print comments
    test_comments, test_labels = data.get_test_data()

    print "Building Logistic Regression Pipeline"
    lr_pipeline = lr.build_stacked_model()

    print "Training Logistic Regression"
    lr_pipeline.fit(comments, labels)

    print "Predicting Logistic Regression"
    predictions = lr_pipeline.predict(test_comments)

    print predictions

if __name__ == "__main__":
    build_model()
