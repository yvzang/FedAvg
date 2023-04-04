#include "pch.h"
#include "Cipher_Interface.h"


Cipher_Interface::~Cipher_Interface() {

}

Cipher_Interface::Cipher_Interface(cipher_list& ciphers,seal::SEALContext& context, const size_t slot_count = 4096) {
	for (int i = 0; i < ciphers.length; i++) {
		seal::Ciphertext cipher;
		cipher.load(context, reinterpret_cast<seal::seal_byte*>(ciphers.cipher_data[i].data), static_cast<size_t>(ciphers.cipher_data[i].length));
		this->cipher_lst.push_back(cipher);
	}
}

void Cipher_Interface::decrypting(seal::Decryptor& decryptor) {
	for (auto cipher : this->cipher_lst) {
		seal::Plaintext plain;
		decryptor.decrypt(cipher, plain);
		this->plain_list.push_back(plain);
	}
}

void Cipher_Interface::decoding(seal::CKKSEncoder& encoder, size_t slot_count = 4096) {
	for (auto plain : this->plain_list) {
		py_plain_vector py_plain(slot_count, 0ULL);
		encoder.decode(plain, py_plain);
		this->py_plain_lst.push_back(py_plain);
	}
}

py_plain_list Cipher_Interface::operator()() {
	py_plain_list* result = new py_plain_list;
	size_t vector_num = this->py_plain_lst[0].size();
	size_t elemet_num = (this->py_plain_lst.size()) * vector_num;
	//¸øresult·ÖÅä¿Õ¼ä
	result->data = new plain_type[elemet_num];
	result->length = elemet_num;
	plain_type* p = result->data;
	for (int i = 0; i < this->py_plain_lst.size(); i++) {
		auto vec = this->py_plain_lst[i];
		std::copy(vec.begin(), vec.end(), p);
		p = p + vector_num;
	}
	return *result;
}

std::vector<seal::Ciphertext>& Cipher_Interface::get_cipher_list(){
	return this->cipher_lst;
}

const Cipher_Interface Cipher_Interface::add(const Cipher_Interface& cipher_inter, seal::Evaluator& evaluator, seal::RelinKeys& relin_key) const {
	Cipher_Interface result;
	for (int i = 0; i < this->cipher_lst.size(); i++) {
		seal::Ciphertext cipher;
		evaluator.add(this->cipher_lst[i], cipher_inter.cipher_lst[i], cipher);
		evaluator.relinearize_inplace(cipher, relin_key);
		result.cipher_lst.push_back(cipher);
	}
	return result;
}

const Cipher_Interface Cipher_Interface::multiple(const Cipher_Interface& cipher_inter, seal::Evaluator& evaluator, seal::RelinKeys& reline_key) const {
	Cipher_Interface result;
	for (int i = 0; i < this->cipher_lst.size(); i++) {
		seal::Ciphertext cipher;
		evaluator.multiply(this->cipher_lst[i], cipher_inter.cipher_lst[i], cipher);
		//relineaer
		evaluator.relinearize_inplace(cipher, reline_key);
		//rescale
		evaluator.rescale_to_next_inplace(cipher);
		result.cipher_lst.push_back(cipher);
	}
	return result;
}