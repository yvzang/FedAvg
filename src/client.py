from Module import cifar10
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules import CrossEntropyLoss
from transformer import Transformer
import torch
import torchvision
from seal import *


class Client():
    def __init__(self, id, params, sealer, lock):
        self.dataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=True,
                                                    transform=torchvision.transforms.ToTensor(), download=True)
        self.local_module = self.__to_cuda__(cifar10())
        self.lock = lock
        self.set_parameters(params)
        self.trans = Transformer()
        self.module_length = len(self.trans.para_to_list(self.local_module.state_dict(), self.local_module))
        self.__loss_fn = self.__to_cuda__(CrossEntropyLoss())
        self.learning_rate = 0.01
        self.__optim = torch.optim.SGD(self.local_module.parameters(), lr=self.learning_rate)
        self.sealer = sealer
        self.client_id = id
        self.pg = 5.0e-4
        self.ng = -5.0e-4

    def __to_cuda__(self, module):
        if(torch.cuda.is_available()):
            return module.cuda()
        
    def set_parameters(self, parameters):
        self.local_module.load_state_dict(parameters)
        

    def set_gradients(self, para, divide_num):
        '''设置参与方本地模型参数,'''
        if isinstance(para, cipher_list) == False:
            raise Exception("模型参数类型不正确.")
        #先解密
        self.lock.acquire()
        parameters_list = self.sealer.decrypt(para, self.module_length)
        self.lock.release()
        #转为tensor
        parameters_tensor = torch.tensor(parameters_list)
        #计算平均
        parameters_tensor = parameters_tensor / divide_num
        #设置grad
        self.trans.list_to_grad(parameters_tensor, self.local_module.parameters(), self.local_module)
        #更新梯度
        self.__optim.step()


    def set_gradient_rate(self, rate):
        self.pg = self.pg * rate
        self.ng = self.ng * rate

        
    def get_gradients(self):
        with torch.set_grad_enabled(True):
            result_lst = []
            for value in self.local_module.parameters():
                ten_grad_peer_para = value.grad.data
                pos_bool = (ten_grad_peer_para > 0).float()
                neg_bool = (ten_grad_peer_para < 0).float()
                pos_grad = pos_bool * ten_grad_peer_para
                neg_grad = neg_bool * ten_grad_peer_para
                sum_fn_list = list(range(len(list(ten_grad_peer_para.shape))))
                pg = pos_grad.sum(sum_fn_list) / pos_bool.sum(sum_fn_list)
                ng = neg_grad.sum(sum_fn_list) / neg_bool.sum(sum_fn_list)
                ten_grad_peer_para = (pos_bool * pg.item()) + (neg_bool * ng.item())
                
                ten_grad_peer_para = ten_grad_peer_para.reshape([-1])
                ten_grad_peer_para = ten_grad_peer_para.cpu()
                lst = ten_grad_peer_para.numpy().tolist()
                result_lst = result_lst + lst
            return result_lst



    def update_parameters(self, params_queue, epoch, mini_batch):
        train_batchs = DataLoader(self.dataset, mini_batch, True)
        
        #epoch轮迭代
        self.local_module.eval()
        print("参与方{}开始训练..".format(self.client_id))
        self.__curr_loss = 0
        #一次迭代
        image, label = train_batchs.__iter__().__next__()
        #计算输出
        image = self.__to_cuda__(image)
        label = self.__to_cuda__(label)
        output = self.local_module(image)
        #计算损失
        self.__curr_loss = self.__loss_fn(output, label)
        print("第{}个参与方迭代, 损失值：{}".format(self.client_id, self.__curr_loss))
        #初始化梯度参数
        self.__optim.zero_grad()
        #反向传播
        self.__curr_loss.backward()
        gradient_list = self.get_gradients()
        self.lock.acquire()
        params_queue.put(self.sealer.encrypt(gradient_list), block=True)
        self.lock.release()
        return

    def test(self, module_path=None):
        mini_batch = 200
        #如果提供了保存模型路径应该从路径读取
        if module_path != None:
            self.local_module.load_state_dict(torch.load(module_path))
        dataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
        data_batch = DataLoader(dataset, mini_batch, True)

        with torch.no_grad():
            total_loss = 0
            for image, label in data_batch:
                image = self.__to_cuda__(image)
                label = self.__to_cuda__(label)
                output = self.local_module(image)
                
                loss_list = output.argmax(1)
                loss_list = (loss_list == label).sum()
                total_loss += loss_list.float()
            mean_loss = total_loss / (len(data_batch) * mini_batch)
            print("平均正确率为{}".format(mean_loss))
            return mean_loss

    def save(self, save_path):
        torch.save(self.local_module.state_dict(), save_path)
        return True