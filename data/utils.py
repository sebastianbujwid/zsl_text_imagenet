def read_class_names_list(file):
    with open(file, 'r') as f:
        return f.read().splitlines()
