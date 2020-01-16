from modules.train.learner import Learner


if __name__ == "__main__":
    learn = Learner.create(
        tensorboard_dir="logs/ob",
        model_name="bert-base-cased",
        model_type="cls",
        train_df_path="/data/aaemeljanov/author_identification/train.csv",
        valid_df_path="/data/aaemeljanov/author_identification/valid.csv",
        dictionaries_path="dicts.pkl",
        batch_size=4,
        update_freq=8
    )
    learn.learn()
