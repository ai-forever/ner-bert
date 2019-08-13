import sys
import warnings
from modules.data import bert_data_clf
from modules.models.classifiers import BERTBiLSTMAttnClassifier
from modules.train.train_clf import NerLearner


warnings.filterwarnings("ignore")
sys.path.append("../")


def main():
    train_df_path = "/home/ubuntu/censor/train2.csv"
    valid_df_path = "/home/ubuntu/censor/dev2.csv"
    test_df_path = "/home/ubuntu/censor/test.csv"
    num_epochs = 100


    data = bert_data_clf.LearnDataClass.create(
        train_df_path=train_df_path,
        valid_df_path=valid_df_path,
        idx2cls_path="/home/ubuntu/censor/idx2cls.txt",
        clear_cache=False,
        batch_size=64
    )

    model = BERTBiLSTMAttnClassifier.create(len(data.train_ds.cls2idx), hidden_dim=768)
    learner = NerLearner(
        model, data, "/home/ubuntu/censor/cls.cpt4", t_total=num_epochs * len(data.train_dl))
    learner.fit(epochs=num_epochs)
    

if __name__ == "__main__":
    main()
