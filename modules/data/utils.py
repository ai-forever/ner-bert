import pickle


def save_pkl(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, "rb") as file:
        res = pickle.load(file)
    return res


def collate_tokens(values, pad_idx=0):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # print([v.shape for v in values])
    size = max(v.size(0) for v in values)
    # print(size)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][:len(v)])
    return res
