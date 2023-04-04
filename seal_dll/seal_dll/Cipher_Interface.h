#pragma once
#include<vector>
#include"Test_Dll.h"


using py_plain_vector = std::vector<plain_type>;
using py_plain_list_t = std::vector<py_plain_vector>;

/*´Ópython listµ½*/
class Cipher_Interface
{
private:
	using cipher_list_t = std::vector<seal::Ciphertext>;
	using plain_list_t = std::vector<seal::Plaintext>;
	cipher_list_t cipher_lst{};
	plain_list_t plain_list{};
	py_plain_list_t py_plain_lst{};
public:
	Cipher_Interface() {};
	Cipher_Interface(cipher_list &, seal::SEALContext &, const size_t slot_count);
	~Cipher_Interface();
	void decrypting(seal::Decryptor&);
	void decoding(seal::CKKSEncoder&, size_t);
	cipher_list_t& get_cipher_list();
	py_plain_list operator()();
	const Cipher_Interface add(const Cipher_Interface&, seal::Evaluator&, seal::RelinKeys&) const;
	const Cipher_Interface multiple(const Cipher_Interface&, seal::Evaluator&, seal::RelinKeys&) const;
};

