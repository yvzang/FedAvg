import server

if __name__ == "__main__":
    ser = server.Server()
    ser.init_module()
    ser.train(5, 0.8, 1, 10, 200, 4)
    ser.test()