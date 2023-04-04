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

/*����python api���ݸ�c++�Ľṹ������plain list��ָ���Ԫ������*/
struct py_plain_list {
	plain_type* data;
	int length;
};

/*����Ҫ���ظ�python api�Ľṹ������ָ��ÿ���ֶε�ָ��Ͱ������ֶ�����*/
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};


/*������л�parameters����*/
ser_params_t get_parameter_seriazation();

/*�����Կ
������
ser_params_t: ���л�parameters
return:
������Կ��˽Կ��reline key�Ľṹ��*/
ser_keys get_keys(ser_params_t);

/*����
��������Ҫ���ܵ�����py_plain_list
struct py_plain_list {
	plain_type* data;
	int length;
};
���л�����ser_params_t�����л���Կser_public_key_t
����:
���л�����cipher_list
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};*/
cipher_list encryption(py_plain_list, ser_params_t, ser_public_key_t);
/*����
�����������л������Ķ���
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};
���л��Ĳ���ser_params_t(data_buffer)�����л�����Կser_secret_key_t(data_buffer)
���أ�python��ʶ�������py_plain_list
struct py_plain_list {
	double* data;
	int length;
};
*/
py_plain_list decryption(cipher_list, ser_params_t, ser_secret_key_t);
/*�ӷ�̬ͬ
�����������л������Ķ���cipher_list
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};
�����л��Ĳ���ser_params_t�������л���ser_relin_key_t
���أ�
�����ӷ������л�����cipher_list
*/
cipher_list addition(cipher_list, cipher_list, ser_params_t, ser_relin_key_t);
/*�˷�̬ͬ
�����������л����������Ķ���cipher_list
struct cipher_list {
	ser_cipher_t* cipher_data;
	int length;
};
struct ser_cipher_t {
	seal_type* data;
	unsigned long long length;
};
�����л��Ĳ���ser_params_t�������л���ser_relin_key_t
���أ�
�����˷������л�����cipher_list*/
cipher_list mutiplication(cipher_list, cipher_list, ser_params_t, ser_relin_key_t);
/*ɾ����Pain_Interface�ഴ��������cipher_list�Ĵ洢�ռ�
* ������
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
/*ɾ����Cipher_Interface�ഴ�������Ĵ洢�ռ�
������
pyYplain_list:
struct py_plain_list {
	plain_type* data;	->delete
	int length;
};
*/
void delete_py_plain_list(py_plain_list&);