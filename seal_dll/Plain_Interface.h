#pragma once
#include<vector>
#include"Test_Dll.h"


/*希望有一个接口，能将python list类型转化为vector表示的类型
这样就可以方便地对每个vector做加密运算*/
class Plain_Interface
{
private:
	/*用vector来存储明文片段
	这样做是因为输入的list size要比slot_count大*/
	using plains_vector = std::vector<plain_type>;
	using plains_list_t = std::vector<plains_vector>;
	/*用来存储经过encode之后的明文*/
	using encode_list_t = std::vector<seal::Plaintext>;
	/*用来存储经过cryption之后的密文*/
	using cipher_list_t = std::vector<seal::Ciphertext>;
	plains_list_t plains_list{};
	encode_list_t encode_list{};
	cipher_list_t cipher_lst{};
public:
	Plain_Interface() {};
	~Plain_Interface();
	/*从python list到plains_list_t的接口*/
	Plain_Interface(const plain_type*, const int n, const int slot_count);
	/*将this->plains_list encode to this->encode_list*/
	void encoding(seal::CKKSEncoder&, double scale);
	/*将this->encode_list encrypt to cipher_lst*/
	void encrypting(seal::Encryptor&);
	/*从c++到python list的接口
	返回cipher_list structure*/
	cipher_list operator()();

	cipher_list to_cipher_list(const std::vector<seal::Ciphertext>&);
};

