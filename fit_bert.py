from modules.train.learner import Learner


if __name__ == "__main__":
    learn = Learner.create(
        tensorboard_dir="logs/obp",
        model_name="bert-base-cased",
        model_type="cls",
        train_df_path="/data/aaemeljanov/author_identification/train.csv",
        valid_df_path="/data/aaemeljanov/author_identification/valid.csv",
        test_df_path="/data/aaemeljanov/author_identification/test.csv",
        dictionaries_path="dicts.pkl",
        batch_size=4,
        update_freq=8,
        weight_decay=0.95,
        lr=2e-5,
        warmup=0.1,
        b1=0.9,
        b2=0.999,
        epochs=5
    )
    learn.learn()
