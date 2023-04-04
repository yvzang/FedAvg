#pragma once
#include"Test_Dll.h"


/*��vector���洢����Ƭ��
����������Ϊ�����list sizeҪ��slot_count��*/
using plains_vector = std::vector<plain_type>;
using plains_list_t = std::vector<plains_vector>;

class DataInterface
{
private:
	/*�����洢����encode֮�������*/
	using encode_list_t = std::vector<seal::Plaintext>;
	/*�����洢����cryption֮�������*/
	using cipher_list_t = std::vector<seal::Ciphertext>;
	plains_list_t plains_list{};
	encode_list_t encode_list{};
	cipher_list_t cipher_lst{};

public:
	/*�����޲�����ʼ���ӿڵķ���*/
	DataInterface() {};
	/*�����ĳ�ʼ���ӿ�*/
	DataInterface(const plain_type*, const int n, const int slot_count);
	/*�����ĳ�ʼ���ӿ�*/
	DataInterface(cipher_list&, seal::SEALContext&, const size_t slot_count);
	/*��this->plains_list encode to this->encode_list*/
	void encoding(seal::CKKSEncoder&, double scale);
	/*��this->encode_list encrypt to cipher_lst*/
	void encrypting(seal::Encryptor&);
	/*תΪ����*/
	cipher_list to_cipher();
	/*��this->cipher_lst decrypt to this->encode_list*/
	void decrypting(seal::Decryptor&);
	/*��this->encode_list decode to this->plain_list*/
	void decoding(seal::CKKSEncoder&, size_t);
	/*תΪ����*/
	py_plain_list to_plain();
	const DataInterface add(const DataInterface&, seal::Evaluator&, seal::RelinKeys&) const;
	const DataInterface multiple(const DataInterface&, seal::Evaluator&, seal::RelinKeys&) const;
};

