#pragma once

#ifdef TEST_DLL
#define TEST_DLL_API extern "C" _declspec(dllexport)
#else
#define TEST_DLL_API extern "C" _declspec(dllimport)
#endif


using seal_type = char;
using plain_type = double;
struct data_buffer {
	seal_type* data;
	unsigned long long length;
};


using ser_public_key_t = data_buffer;
using ser_secret_key_t = data_buffer;
using ser_relin_key_t = data_buffer;
using ser_params_t = data_buffer;
using ser_cipher_t = data_buffer;

struct keys {
	ser_public_key_t public_key;
	ser_secret_key_t secret_key;
	ser_relin_key_t relin_key;
};
using ser_keys = keys;

/*这是python api传递给c++的结构，包含plain list的指针和元素数量*/
struct py_plain_list {
	plain_type* data;
	int length;
};

/*这是要返回给python api的结构，包含指向每个字段的指针和包含的字段数量*/
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};


/*获得序列化parameters参数*/
ser_params_t get_parameter_seriazation();

/*获得密钥
参数：
ser_params_t: 序列化parameters
return:
包括公钥、私钥和reline key的结构体*/
ser_keys get_keys(ser_params_t);

/*加密
参数：需要加密的明文py_plain_list
struct py_plain_list {
	plain_type* data;
	int length;
};
序列化参数ser_params_t、序列化公钥ser_public_key_t
返回:
序列化密文cipher_list
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};*/
cipher_list encryption(py_plain_list, ser_params_t, ser_public_key_t);
/*解密
参数：被序列化的密文对象
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};
序列化的参数ser_params_t(data_buffer)、序列化的密钥ser_secret_key_t(data_buffer)
返回：python可识别的类型py_plain_list
struct py_plain_list {
	double* data;
	int length;
};
*/
py_plain_list decryption(cipher_list, ser_params_t, ser_secret_key_t);
/*加法同态
参数：被序列化的密文对象cipher_list
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};
被序列化的参数ser_params_t、被序列化的ser_relin_key_t
返回：
经过加法的序列化密文cipher_list
*/
cipher_list addition(cipher_list, cipher_list, ser_params_t, ser_relin_key_t);
/*乘法同态
参数：被序列化的两个密文对象cipher_list
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};
被序列化的参数ser_params_t、被序列化的ser_relin_key_t
返回：
经过乘法的序列化密文cipher_list*/
cipher_list mutiplication(cipher_list, cipher_list, ser_params_t, ser_relin_key_t);
/*删除由Pain_Interface类创建出来的cipher_list的存储空间
* 参数：
* cipher_list:
struct cipher_list { 
	ser_cipher_t* cipher_data;	->delete
	int length;
};
struct ser_cipher_t {
	seal_type* data;	->delete
	unsigned long long length;
};*/
void delete_cipher_list(cipher_list&);
/*删除由Cipher_Interface类创建出来的存储空间
参数：
pyYplain_list:
struct py_plain_list {
	plain_type* data;	->delete
	int length;
};
*/
void delete_py_plain_list(py_plain_list&);