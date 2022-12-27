from utilities import load_from_file, activate
from network import JordanNetwork

if __name__ == "__main__":
    '''Network currently has only fibonacci sequence weights trained, so if you would like to predict some other 
    sequence, you must train this network first '''
    print("Pick one of sequences\n")
    seq_type = int(input("1 - Fibonacci sequence\n"
                         "2 - Periodical sequence\n"
                         "3 - Power function with base 1/2\n"
                         "4 - Function х!*0,000001\n")
                   )
    while True:
        jordan_network = JordanNetwork(1e-2, 1e-6, 1000000, 4, load_from_file(seq_type), activate)
        print("Choose option:\n"
              "1 - Learning mode\n"
              "2 - Work mode\n"
              "3 - Load weights\n"
              "4 - Switch sequence\n"
              "0 - Exit program\n")
        choice = int(input()
                     )
        match choice:
            case 0:
                break
            case 1:
                print(jordan_network.matrix)
                jordan_network.train_network()
                jordan_network.save_weight_matrices(seq_type)
            case 2:
                jordan_network.predict_next_seq_val()
            case 3:
                jordan_network.load_weight_matrices(seq_type)
                jordan_network.predict_next_seq_val()
            case 4:
                print(
                    "1 - Fibonacci sequence\n"
                    "2 - Periodical sequence\n"
                    "3 - Power function with base 1/2\n"
                    "4 - Function х!*0,000001\n"
                )
                seq_type = int(input())
            case _:
                print('Invalid input value')
