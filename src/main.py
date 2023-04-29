import server

if __name__ == "__main__":
    ser = server.Server()
    ser.init_module()
    ser.train(5, 0.8, 1, 2)
    ser.test()