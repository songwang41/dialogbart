import pickle
import fire
def print_pickle(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    print(data)
    with open(input_file+".txt", 'w') as f:
        f.write(str(data)+'\n')

if __name__=="__main__":
    fire.Fire(print_pickle)
    
