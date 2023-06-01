import server
from client import Client
import torch
from Module import cifar10
from copy import deepcopy


if __name__ == "__main__":
    ser = server.Server()
    weight_template = {i:1 for i in range(10)}
    clients = []
    clients.append(Client(1))
    clients.append(Client(2))
    clients.append(Client(3))
    clients.append(Client(4))
    clients.append(Client(5))

    client6 = Client(6)
    w6 = deepcopy(weight_template)
    w6[0] = 2
    w6[1] = 2
    client6.set_weight(w6)
    clients.append(client6)

    client7 = Client(7)
    w7 = deepcopy(w6)
    w7[7] = 3
    w7[4] = 3
    client7.set_weight(w7)
    clients.append(client7)

    client8 = Client(8)
    w8 = deepcopy(weight_template)
    w8[5] = 4
    w8[9] = 5
    w8[3] = 3
    client8.set_weight(w8)
    clients.append(client8)

    client9 = Client(9)
    w9 = deepcopy(weight_template)
    w9[4] = 6
    w9[0] = 5
    client9.set_weight(w9)
    clients.append(client9)

    client10 = Client(10)
    w10 = deepcopy(w8)
    w10[0] = 0
    w10[7] = 0
    client10.set_weight(w10)
    clients.append(client10)

    ser.add_clients(clients)
    ser.train(5, 0.8, 1, 3, 256)
    ser.test()