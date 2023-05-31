import server
from client import Client
import torch
from Module import cifar10

client_num = 2
if __name__ == "__main__":
    ser = server.Server()
    ser.add_clients([Client(i) for i in range(4)])
    ser.train(5, 0.8, 1, 1, 200)
    ser.test()