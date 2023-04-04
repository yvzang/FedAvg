#pragma once
#include"Test_Dll.h"


/*用vector来存储明文片段
这样做是因为输入的list size要比slot_count大*/
using plains_vector = std::vector<plain_type>;
using plains_list_t = std::vector<plains_vector>;

class DataInterface
{
private:
	/*用来存储经过encode之后的明文*/
	using encode_list_t = std::vector<seal::Plaintext>;
	/*用来存储经过cryption之后的密文*/
	using cipher_list_t = std::vector<seal::Ciphertext>;
	plains_list_t plains_list{};
	encode_list_t encode_list{};
	cipher_list_t cipher_lst{};

public:
	/*禁用无参数初始化接口的方法*/
	DataInterface() {};
	/*用明文初始化接口*/
	DataInterface(const plain_type*, const int n, const int slot_count);
	/*用密文初始化接口*/
	DataInterface(cipher_list&, seal::SEALContext&, const size_t slot_count);
	/*将this->plains_list encode to this->encode_list*/
	void encoding(seal::CKKSEncoder&, double scale);
	/*将this->encode_list encrypt to cipher_lst*/
	void encrypting(seal::Encryptor&);
	/*转为密文*/
	cipher_list to_cipher();
	/*将this->cipher_lst decrypt to this->encode_list*/
	void decrypting(seal::Decryptor&);
	/*将this->encode_list decode to this->plain_list*/
	void decoding(seal::CKKSEncoder&, size_t);
	/*转为明文*/
	py_plain_list to_plain();
	const DataInterface add(const DataInterface&, seal::Evaluator&, seal::RelinKeys&) const;
	const DataInterface multiple(const DataInterface&, seal::Evaluator&, seal::RelinKeys&) const;
};

