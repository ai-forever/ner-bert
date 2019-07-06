import os


def get_file_names(path):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tokens'):
                res.append(os.path.join(root, os.path.splitext(file)[0]))
    return res
