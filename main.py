from getdata import GetDataSet
from clients import Client, ClientsGroup
import torch.nn.functional as F
from model.Net import Net
import torch
import argparse
from tqdm import tqdm
import argparse
import os
from torch import optim
import numpy as np
from copy import deepcopy



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

parser.add_argument('-nc', '--num_of_clients', type=int, default=3, help='numer of the clients')

parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                    help='C fraction, 0 means 1 client, 1 means total clients')

parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')

parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')

parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')

parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="需要训练的数据集")

parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')

parser.add_argument('-ncomm', '--num_comm', type=int, default=10, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')

parser.add_argument('-sigma', '--sigma_value', type=int, default=0.1, help='the value of sigma for optimatization')



                

def calculate_M(net):
    shap=0
    for name, param in net.named_parameters():
        w=param.data.view(-1)
        shap+=w.shape[0]
        # print(w.shape)
    return shap

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__


    test_txt = open("test_accuracy.txt", mode="a")

    test_mkdir(args['save_path'])

    dev = torch.device("cpu")

    sigma=args['sigma_value']

    net = None


    if args['model_name'] == 'mnist_cnn':
        net = Net()
    else:
        pass


    loss_func = F.cross_entropy




    num_in_comm = int(max(args['num_of_clients'], 1))#int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}

    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    m=calculate_M(net)

    lambda_k =torch.zeros(2*m*(args['num_of_clients']-1))
    
    myClients = ClientsGroup('MNIST', args['num_of_clients'], dev,net,global_parameters,loss_func,args['epoch'],args['batchsize'],m,args['learning_rate'],sigma)
    # testDataLoader = myClients.testDataLoader




    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        y_k_plus_1_list = []
        lambda_k_plus_1_list = []
        ACC_list=[]
        for client in tqdm(clients_in_comm):
            
            if i==0:
                y_k = myClients.init_y_k[client]
            y_k_plus_1, lambda_k_plus_1 , ACC  =myClients.clients_set[client].localModelUpdate(lambda_k,y_k)
            y_k_plus_1_list.append(y_k_plus_1)
            lambda_k_plus_1_list.append(lambda_k_plus_1)
            ACC_list.append(ACC)

        avg_y_k_plus_1 = deepcopy(y_k_plus_1_list[0])
        avg_lambda_k_plus_1 = deepcopy(lambda_k_plus_1_list[0])
        avg_acc= deepcopy(ACC_list[0])
        n_models=len(lambda_k_plus_1_list)#args['num_of_clients']
        for i in range(1, n_models):
            avg_acc += ACC_list[i]
            for key in range(len(avg_y_k_plus_1)):
                avg_y_k_plus_1[key] += y_k_plus_1_list[i][key]
                avg_lambda_k_plus_1[key] += lambda_k_plus_1_list[i][key]
                
        for key in range(len(avg_y_k_plus_1)):
            avg_y_k_plus_1[key] /= n_models
            avg_lambda_k_plus_1[key] /= n_models
        avg_acc /=n_models
        lambda_k =  avg_lambda_k_plus_1
        y_k =  avg_y_k_plus_1

        print('average accuracy: ' + str(float(avg_acc)) + "\n")
    test_txt.close()

