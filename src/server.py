import torch
from Module import cifar10
from client import Client
from torch.utils.data import DataLoader
import random
from threading import Thread, Lock
from transformer import Transformer
from torch.utils.tensorboard import SummaryWriter
import queue
import psutil


class Server():
    def __init__(self):
        self.module_path = "module.pth"
        self.module = cifar10()
        self.clients = []
        self.trans = Transformer()
        self.writer = SummaryWriter(log_dir="./runs/fedprox")

    def __to_cuda__(self, module):
        if(torch.cuda.is_available()):
            return module.cuda()
        
    def add_clients(self, clients):
        lock = Lock()
        for c in clients:
            c.client_init(self.module.state_dict(), lock, self.writer)
            self.clients.append(c)
        

    def print_percent(self, percent):
        taltol_length = 100
        shap_num = int(percent * taltol_length)
        line_num = taltol_length - shap_num
        _format_shap = "#" * shap_num
        _format_shap = _format_shap + "%" + str(percent.item() * 100)
        _formate_line = "-" * line_num
        print(_format_shap + _formate_line)
        

    def train(self, test_epoch, end_rate, train_rate, epoch, batch_size):
        '''训练模型参数'''
        if(len(self.clients) == 0):
            raise RuntimeError("客户端未连接")
        #进行epoch轮迭代
        choiced_clients_num = int(train_rate * len(self.clients))
        total_epoch = 0
        while(True):
            #挑选train_rate * len(clients)个参与方
            choiced_clients = random.sample(self.clients, choiced_clients_num)
            
            params_queue = queue.Queue(maxsize=len(self.clients))
            thread_list = []
            for single_client in choiced_clients:
                #同步训练，获得局部模型参数
                thread = Thread(target=single_client.update_parameters, args=(params_queue, epoch, batch_size), daemon=True, name="{}".format(single_client.client_id))
                thread_list.append(thread)
                thread.start()
            #等待参与方训练完成
            for t in thread_list:
                t.join()
            #加权平均
            global_params = self.__parameter_weight_divition__(params_queue)
            self.set_clients_parameters(self.clients, global_params)
            #判断是否结束训练
            total_epoch += 1
            print("第{}轮训练：".format(total_epoch))
            #查看内存使用率
            info = psutil.virtual_memory()
            print("内存使用率:" + str(info.percent))
            if(total_epoch % test_epoch == 0):
                current_rate = self.test()
                self.print_percent(current_rate)
                self.writer.add_scalars(main_tag="accuracy", tag_scalar_dict={"without_kl": current_rate}, global_step=total_epoch)
                if current_rate >= end_rate:
                    break

    def test(self):
        choiced_client = random.sample(self.clients, 1)
        return choiced_client[0].test()

    def save(self, save_path):
        choiced_client = random.sample(self.clients, 1)
        return choiced_client[0].save(save_path)


    def __parameter_weight_divition__(self, params_queue):
        '''对所有参与方的参数加和'''
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        params_list = []
        kl_list = []
        #先取出所有元素
        while(params_queue.empty() == False):
            params_dics = params_queue.get()
            #先检验参数类型
            if(isinstance(params_dics, dict)is not True):
                raise Exception("参数类型不正确")
            params_list.append(params_dics["params"])
            kl_list.append(params_dics["kl_div"])
        if(len(params_list) != len(kl_list)):
            raise Exception("参数列表与KL散度列表不匹配")
        kl_ten = torch.Tensor(kl_list)
        kl_taltol = kl_ten.sum()
        #熵权法算出所占比例
        kl_percent =((-kl_ten + 1) / (len(kl_list) - kl_taltol))
        kl_percent.max()
        print(kl_percent)
        #加权平均
        result_ten = torch.zeros(1).float()
        for i in range(len(params_list)):
            result_ten = result_ten + torch.Tensor(params_list[i]) * kl_percent[i]
        return result_ten.tolist()
        

    def set_clients_parameters(self, clients, para):
        '''将参数para传递给参与方列表clients'''
        for client in clients:
            client.set_parameters_from_list(para)
