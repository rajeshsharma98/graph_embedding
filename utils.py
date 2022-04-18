import json

def adj_list_reader(file):
    adj_list = {};
    # Opening JSON file
    with open(file) as json_file:
        adj_list = json.load(json_file)
    return adj_list
