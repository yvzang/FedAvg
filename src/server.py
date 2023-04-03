import torch
from Module import cifar10
from client import Client
from torch.utils.data import DataLoader
import random
from threading import Thread, Lock
from seal import *
import queue
import psutil


class Server():
    def __init__(self):
        self.init_module()
        self.module_path = "module.pth"
    
    def init_module(self):
        '''初始化模型'''
        #self.module = self.__to_cuda__(cifar10())

    def __to_cuda__(self, module):
        if(torch.cuda.is_available()):
            return module.cuda()
        
    def __train_init__(self, client_num):
        '''确定参与方数量，初始化模型参数'''
        self.sealer = Seal()
        #设置全局模型参数
        module = cifar10()
        lock = Lock()
        self.clients = [Client(i, module.state_dict(), self.sealer, lock) for i in range(client_num)]
        

    def train(self, test_epoch, end_rate, train_rate, client_num):
        '''训练模型参数'''
        self.__train_init__(client_num)
        #进行epoch轮迭代
        choiced_clients_num = int(train_rate * len(self.clients))
        epoch = 0
        total_epoch = 0
        while(True):
            #挑选train_rate * len(clients)个参与方
            choiced_clients = random.sample(self.clients, choiced_clients_num)

            params_queue = queue.Queue(maxsize=len(self.clients))
            thread_list = []
            for single_client in choiced_clients:
                #同步训练，获得局部模型参数
                thread = Thread(target=single_client.update_parameters, args=(params_queue, 100, 256), daemon=True, name="{}".format(single_client.client_id))
                thread_list.append(thread)
                thread.start()
            #等待参与方训练完成
            for t in thread_list:
                t.join()
            #加权平均
            total_parameters_added = self.__parameter_addition__(params_queue)
            self.set_clients_parameters(self.clients, total_parameters_added, len(choiced_clients))
            #判断是否结束训练
            epoch += 1
            total_epoch += 1
            print("第{}轮训练：".format(total_epoch))
            #查看内存使用率
            info = psutil.virtual_memory()
            print(info.percent)
            if(epoch % test_epoch == 0):
                epoch = 0
                current_rate = self.test()
                if current_rate >= end_rate:
                    break

    def test(self):
        choiced_client = random.sample(self.clients, 1)
        return choiced_client[0].test()

    def save(self, save_path):
        choiced_client = random.sample(self.clients, 1)
        return choiced_client[0].save(save_path)

    
    def __parameter_divi__(self, parameters, n):
        '''计算参数的平均值: para / n'''
        if(isinstance(parameters, cipher_list) == False):
            raise Exception("参数类型不正确.")


    def __parameter_addition__(self, params_queue):
        '''对所有参与方的参数加和'''
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        cipher_total = params_queue.get()
        if(isinstance(cipher_total, cipher_list) == False):
            raise Exception("参数类型不正确.")
        while(params_queue.empty() == False):
            item = params_queue.get()
            temp_poiter = self.sealer.add(cipher_total, item)
            self.sealer.delete_cipher_memory(cipher_total)
            self.sealer.delete_cipher_memory(item)
            cipher_total = temp_poiter
        return cipher_total
        

    def set_clients_parameters(self, clients, para, divide_num):
        '''将参数para传递给参与方列表clients'''
        for client in clients:
            client.set_gradients(para, divide_num)
