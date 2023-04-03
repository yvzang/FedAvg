#pragma once
#include<vector>
#include"Test_Dll.h"


/*ϣ����һ���ӿڣ��ܽ�python list����ת��Ϊvector��ʾ������
�����Ϳ��Է���ض�ÿ��vector����������*/
class Plain_Interface
{
private:
	/*��vector���洢����Ƭ��
	����������Ϊ�����list sizeҪ��slot_count��*/
	using plains_vector = std::vector<plain_type>;
	using plains_list_t = std::vector<plains_vector>;
	/*�����洢����encode֮�������*/
	using encode_list_t = std::vector<seal::Plaintext>;
	/*�����洢����cryption֮�������*/
	using cipher_list_t = std::vector<seal::Ciphertext>;
	plains_list_t plains_list{};
	encode_list_t encode_list{};
	cipher_list_t cipher_lst{};
public:
	Plain_Interface() {};
	~Plain_Interface();
	/*��python list��plains_list_t�Ľӿ�*/
	Plain_Interface(const plain_type*, const int n, const int slot_count);
	/*��this->plains_list encode to this->encode_list*/
	void encoding(seal::CKKSEncoder&, double scale);
	/*��this->encode_list encrypt to cipher_lst*/
	void encrypting(seal::Encryptor&);
	/*��c++��python list�Ľӿ�
	����cipher_list structure*/
	cipher_list operator()();

	cipher_list to_cipher_list(const std::vector<seal::Ciphertext>&);
};

