import torch
from Module import cifar10
from client import Client
from torch.utils.data import DataLoader
from torch.nn.modules import CrossEntropyLoss
import random
from threading import Thread, Lock
from transformer import Transformer
from copy import deepcopy
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import queue
import psutil


class Server():
    def __init__(self):
        self.module_path = "module.pth"
        self.module = self.__to_cuda__(cifar10())
        self.testbatchsize = 200
        self.testdataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=False,
                                    transform=torchvision.transforms.ToTensor(), download=True)
        self.testdadaloader = DataLoader(self.testdataset, self.testbatchsize, True)
        self.loss_fn = self.__to_cuda__(CrossEntropyLoss())
        self.clients = []
        self.trans = Transformer()
        self.writer = SummaryWriter(log_dir="./runs/fedprox")
        self.learning_rate = 0.1
        self.norm_rate = 0.05
        self.agg_rate = 1
        self.optim = torch.optim.SGD(self.module.parameters(), self.learning_rate)

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
        _format_shap = _format_shap + "%{:.4f}" + str(percent.item() * 100)
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
            global_params = self.__parameter_weight_divition__(params_queue, choiced_clients_num)
            loss, accuracy = self.__get_loss__()
            print("loss: {}".format(loss.item()) + ", accuracy: {}".format(accuracy.item()))
            self.writer.add_scalars(main_tag="loss", tag_scalar_dict={"p=4,q=1": loss}, global_step=total_epoch)
            self.set_clients_parameters(self.clients, global_params)
            #判断是否结束训练
            total_epoch += 1
            print("第{}轮训练：".format(total_epoch))
            if(total_epoch % test_epoch == 0):
                self.print_percent(accuracy)
                self.writer.add_scalars(main_tag="accuracy", tag_scalar_dict={"p=4,q=1": accuracy}, global_step=total_epoch)
                if accuracy >= end_rate:
                    break


    def save(self, save_path):
        choiced_client = random.sample(self.clients, 1)
        return choiced_client[0].save(save_path)
    
    def __get_loss__(self):
        loss = self.__to_cuda__(torch.zeros(1))
        accuracy = self.__to_cuda__(torch.zeros(1))
        for image, label in self.testdadaloader:
            image = self.__to_cuda__(image)
            label = self.__to_cuda__(label)
            output = self.module(image)
            #计算损失
            curr_loss = self.loss_fn(output, label)
            loss += curr_loss
            #计算准确率
            accu_list = output.argmax(1)
            accu_list = (accu_list == label).sum()
            accuracy += accu_list.float()
        mean_accu = accuracy / (len(self.testdadaloader) * self.testbatchsize)
        return loss, mean_accu
    
    def calculate_score(self, pseudo_grad):
        with torch.set_grad_enabled(False):
            old_weight = deepcopy(self.module.state_dict())
            old_loss, _ = self.__get_loss__()
            #计算更新后的损失
            with torch.set_grad_enabled(True):
                self.optim.zero_grad()
                norm_tatol = self.__to_cuda__(torch.zeros(1))
                for(param_name, param) in self.module.named_parameters():
                    param.grad = pseudo_grad[param_name]
                    ten_grad = pseudo_grad[param_name].reshape([-1])
                    norm_tatol += ten_grad.norm(2, dim=0)
                self.optim.step()

            new_loss, _ = self.__get_loss__()
            score = old_loss - new_loss + self.norm_rate * (1 / norm_tatol)
            self.module.load_state_dict(old_weight)
            return score.item()


    def __parameter_weight_divition__(self, params_queue, divide_num):
        '''对所有参与方的参数加和'''
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        with torch.set_grad_enabled(True):
            #先取出所有梯度
            params_list = []
            norm_list = []
            while(params_queue.empty() == False):
                param = params_queue.get()["params"]
                params_list.append(param)
                norm_list.append(self.calculate_score(param))
            norm_params_list = list(zip(norm_list, params_list))
            #筛选
            norm_params_list.sort(key=lambda np: np[0], reverse=True)
            aggregation_list = deepcopy(norm_params_list[:int(len(norm_params_list) * self.agg_rate)])
            if(len(aggregation_list) == 0):
                raise Exception("score-parameters对为空")
            total_score = sum([score for score, _ in aggregation_list])
            #聚合
            self.optim.zero_grad()
            for(param_name, param) in self.module.named_parameters():
                param.grad = torch.zeros_like(param)
                for score, pseudo_grad in aggregation_list:
                    param.grad = param.grad + (score / total_score) * pseudo_grad[param_name]
            self.optim.step()
            return self.module.state_dict()

    def set_clients_parameters(self, clients, para):
        '''将参数para传递给参与方列表clients'''
        for client in clients:
            client.set_parameters(para)
