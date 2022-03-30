import yaml
import os

def get_yaml_data(yaml_file):
    f = open(yaml_file, 'r', encoding='utf-8')
    fdata = f.read()
    f.close()
    d = yaml.load(fdata)
    return d 

def gen_yaml_file(d, fn):
    f = open(fn, 'w', encoding='utf-8')
    yaml.dump(d, f)
    f.close()
