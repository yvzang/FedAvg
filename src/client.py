from Module import cifar10
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules import CrossEntropyLoss
from transformer import Transformer
from torch.utils.data import WeightedRandomSampler
from copy import deepcopy
import torch
import gc
import torchvision
from scipy import stats


class Client():
    def __init__(self, id):
        self.dataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=True,
                                                    transform=torchvision.transforms.ToTensor(), download=True)
        self.local_module = self.__to_cuda__(cifar10())
        self.trans = Transformer()
        self.module_length = len(self.trans.para_to_list(self.local_module.state_dict(), self.local_module))
        self.__loss_fn = self.__to_cuda__(CrossEntropyLoss())
        self.learning_rate = 0.01
        self.__optim = torch.optim.SGD(self.local_module.parameters(), lr=self.learning_rate)
        self.client_id = id
        self.taltol_epoch = 0
        self.weights = [1 for image, label in self.dataset]

    def client_init(self, params, lock, writer):
        self.set_parameters(params)
        self.lock = lock
        self.writer = writer

    def __to_cuda__(self, module):
        if(torch.cuda.is_available()):
            return module.cuda()
        
    def set_weight(self, bios = None):
        if(bios is not None):
            self.weights = [5 if label in bios else 1 for image, label in self.dataset]
        
    def set_parameters(self, parameters):
        self.local_module.load_state_dict(parameters)

    def get_parameters(self):
        return self.local_module.state_dict()

    def set_parameters_from_list(self, params_lst):
        '''设置参与方本地模型参数,'''
        if isinstance(params_lst, list) == False:
            raise Exception("模型参数类型不正确.")
        params_ten = self.trans.list_to_para(params_lst, self.local_module)
        self.set_parameters(params_ten)
    
    def print_percent(self, percent):
        taltol_length = 40
        shap_num = int(percent * 40)
        line_num = taltol_length - shap_num
        _format_shap = "#" * shap_num
        _formate_line = "-" * line_num
        print(_format_shap + _formate_line)


    def update_parameters(self, params_queue, epoch, mini_batch):
        
        self.local_module.eval()

        sampler = WeightedRandomSampler(self.weights, self.dataset.__len__(), True)
        print("参与方{}开始训练..".format(self.client_id))
        old_weights = deepcopy(self.local_module.state_dict())
        curr_loss = 0
        total_loss = 0
        total_epoch = 0
        for ep in range(epoch):
            train_batchs = DataLoader(self.dataset, mini_batch, False, sampler)
            for image, label in train_batchs:
                #计算输出
                image = self.__to_cuda__(image)
                label = self.__to_cuda__(label)
                output = self.local_module(image)
                #计算损失
                curr_loss = self.__loss_fn(output, label)
                total_loss += curr_loss.item()
                #初始化梯度参数
                self.__optim.zero_grad()
                #反向传播
                curr_loss.backward()
                #梯度更新
                self.__optim.step()
                total_epoch = total_epoch + 1
        new_weight = self.local_module.state_dict()
        with torch.no_grad():
            pseudo_grad = {param_name : old_weights[param_name].data - new_weight[param_name] for param_name in new_weight.keys()}
            #记录曲线
            self.writer.add_scalars(main_tag="loss", tag_scalar_dict={"without_kl_".format(self.client_id): curr_loss / total_epoch}, global_step=self.taltol_epoch)
            self.taltol_epoch = self.taltol_epoch + 1
            print("第{}个参与方迭代, 损失值：{}".format(self.client_id, total_loss / total_epoch))
            self.lock.acquire()
            params_queue.put({"params":pseudo_grad}, block=True)
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