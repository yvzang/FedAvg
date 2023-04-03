from ctypes import *


class DataBuffer(Structure):
    _fields_ = [
        ('SealType', POINTER(c_char)),
        ('Length', c_ulonglong)
    ]

class Keys(Structure):
    _fields_ = [
        ('PublicKey',DataBuffer),
        ('SecretKey', DataBuffer),
        ('RelinKey', DataBuffer)
    ]

class py_plain_list(Structure):
    _fields_ = [
        ('data', POINTER(c_double)),
        ('length', c_int)
    ]

class cipher_list(Structure):
    _fields_ = [
        ('cipher_data', POINTER(DataBuffer)),
        ('length', c_int)
    ]

class Seal():
    
    def __init__(self):
        self.__dll_module_functions__()
        self.parameters = self.__get_parameters__()
        self.keys = self.__get_keys__()


    def __dll_module_functions__(self):
        self.__dll = windll.LoadLibrary(r"C:\Users\10270\Documents\cppproject\seal_dll\x64\Release\seal_dll.dll")
        #初始化get_parameter_seriazation函数
        self.__get_parameters_fn = self.__dll.get_parameter_seriazation
        self.__get_parameters_fn.restype = DataBuffer
        #初始化get_keys函数
        self.__get_keys_fn = self.__dll.get_keys
        self.__get_keys_fn.argtype = DataBuffer
        self.__get_keys_fn.restype = Keys
        #初始化encryption函数
        self.__encryption_fn = self.__dll.encryption
        self.__encryption_fn.argtypes = (py_plain_list, DataBuffer, DataBuffer)
        self.__encryption_fn.restype = cipher_list
        #初始化decryption函数
        self.__decryption_fn = self.__dll.decryption
        self.__decryption_fn.argtypes = (cipher_list, DataBuffer, DataBuffer)
        self.__decryption_fn.restype = py_plain_list
        #初始化addition函数
        self.__addition_fn = self.__dll.addition
        self.__addition_fn.argtypes = (cipher_list, cipher_list, DataBuffer, DataBuffer)
        self.__addition_fn.restype = cipher_list
        #初始化mutiplication函数
        self.__mutiplication_fn = self.__dll.mutiplication
        self.__mutiplication_fn.argtypes = (cipher_list, cipher_list, DataBuffer, DataBuffer)
        self.__mutiplication_fn.restype = cipher_list
        #初始化delete_cipher_list函数
        self.__delete_cipher_list_fn = self.__dll.delete_cipher_list
        self.__delete_cipher_list_fn.argtype = cipher_list
        #初始化delete_py_plain_list函数
        self.__delete_py_plain_list_fn = self.__dll.delete_py_plain_list
        self.__delete_py_plain_list_fn.argtype = py_plain_list

    def __get_parameters__(self):
        return self.__get_parameters_fn()

    def __get_keys__(self):
        return self.__get_keys_fn(self.parameters)

    def encrypt(self, py_plain):
        '''加密
        参数：
        py_plain: 需要加密的python列表
        返回：
        cipher_list类的密文'''
        plain = py_plain_list((c_double * len(py_plain))(*py_plain), len(py_plain))
        return self.__encryption_fn(plain, self.parameters, self.keys.PublicKey)

    def decrypt(self, cipher, require_length):
        '''解密
        参数：
        cipher:一个cipher_list类的密文
        require_length: 被加密的原始明文的长度
        返回：
        python列表明文'''
        plain_struct = self.__decryption_fn(cipher, self.parameters, self.keys.SecretKey)
        py_list = plain_struct.data[:require_length]
        self.delete_plain_memory(byref(plain_struct))
        return py_list

    def add(self, cipher_l, cipher_r):
        return self.__addition_fn(cipher_l, cipher_r, self.parameters, self.keys.RelinKey)

    def mutiple(self, cipher_l, cipher_r):
        return self.__mutiplication_fn(cipher_l, cipher_r, self.parameters, self.keys.RelinKey)

    def delete_cipher_memory(self, cipher):
        self.__delete_cipher_list_fn(cipher)

    def delete_plain_memory(self, palin):
        self.__delete_py_plain_list_fn(palin)