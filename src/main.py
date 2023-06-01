import server
from client import Client
import torch
from Module import cifar10

client_num = 2
if __name__ == "__main__":
    ser = server.Server()
    client1 = Client(0)
    client1.set_weight([0, 1], 2)
    client2 = Client(1)
    client2.set_weight([2, 3], 5)
    ser.add_clients([client1, client2])
    ser.train(5, 0.8, 1, 1, 256)
    ser.test()