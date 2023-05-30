import server
from client import Client
import torch
from Module import cifar10

client_num = 2
if __name__ == "__main__":
    ser = server.Server()
    client1 = Client(0)
    client2 = Client(1)
    client3 = Client(2)
    ser.add_clients([client1, client2, client3])
    ser.train(5, 0.8, 1, 1, 200)
    ser.test()