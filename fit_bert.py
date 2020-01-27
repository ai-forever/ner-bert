from modules.train.learner import Learner


if __name__ == "__main__":
    learn = Learner.create(
        tensorboard_dir="logs/k0",
        model_name="bert-base-multilingual-cased",
        model_type="cls",
        train_df_path="train.csv",
        valid_df_path="valid.csv",
        dictionaries_path="dicts.pkl",
        batch_size=4,
        update_freq=8,
        weight_decay=0.01,
        lr=2e-5,
        warmup=0.1,
        b1=0.8,
        b2=0.999,
        epochs=10,
        checkpoint_dir="k0",
        max_grad_norm=1,
        model_args=dict(hidden_dropout_prob=0.1)
    )
    print(learn.learn())
