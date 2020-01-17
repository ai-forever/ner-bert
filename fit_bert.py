from modules.train.learner import Learner


if __name__ == "__main__":
    learn = Learner.create(
        tensorboard_dir="logs/obp2",
        model_name="bert-base-cased",
        model_type="cls",
        train_df_path="train.csv",
        valid_df_path="valid.csv",
        test_df_path="test.csv",
        dictionaries_path="dicts.pkl",
        batch_size=4,
        update_freq=8,
        weight_decay=0.05,
        lr=2e-5,
        warmup=0.1,
        b1=0.9,
        b2=0.999,
        epochs=10,
        checkpoint_dir="cobp2"
    )
    learn.learn()
