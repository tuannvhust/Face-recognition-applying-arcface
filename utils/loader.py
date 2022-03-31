
def get_test_list(test_list_path):
    with open(test_list_path,'r') as file:
        pairs = file.readlines()
    data_list = []
    for pair in pairs:
        split = pair.split()
        if split[0] not in data_list:
            data_list.append(split[0])
        if split[1] not in data_list:
            data_list.append(split[1])
    return data_list