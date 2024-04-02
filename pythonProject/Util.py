def read_file(path):
    with open(path, 'r') as f:
        file = f.read()
    return file

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def str_to_dict(string):
    string = string.replace(",", ".")
    l = string.split()
    l = [x.strip() for x in l]

    num_attributes = 0
    temp = l[num_attributes]
    att_names =[]
    while is_float(temp):
        num_attributes+=1
        att_names.append(f'att{num_attributes}')
        temp = l[num_attributes]
    att_names.append("result")

    data = {x:[] for x in att_names}
    # print(num_attributes)
    for i in range(0, len(l), num_attributes+1):
        for j in range(0, num_attributes):
            data[att_names[j]].append(float(l[i+j]))
        data[att_names[-1]].append(l[i+num_attributes])



    return data

def file_to_dict(path, check=False):
    f = read_file(path)
    dic = str_to_dict(f)
    if check:
        print(f'{path}:')
        dataset_info(dic)
        print()
    return dic

def dataset_info(dic):
    for k, v in list(dic.items())[:-1]: print(f'{k}: {len(v)} values, ranging from {min(v)} to {max(v)}')
    s = set(dic["result"])
    print(f'result: {len(s)} unique values {s}')

def get_observation(dataset, index):
    return [dataset[x][index] for x in dataset]

def get_dataset_size(dataset):
    return len(dataset["result"])