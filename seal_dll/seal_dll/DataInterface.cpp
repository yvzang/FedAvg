#include "pch.h"
#include "DataInterface.h"

DataInterface::DataInterface(const plain_type* plain_ptr, const int n, const int slot_count) {
	using namespace std;
	const plain_type* p = plain_ptr;
	int epoch = 0;
	for (epoch = 0; epoch < int(n / slot_count); epoch++) {
		plains_vector temp_vec{};
		copy(p, p + slot_count, insert_iterator<plains_vector>(temp_vec, temp_vec.begin()));
		p = p + slot_count;
		this->plains_list.push_back(temp_vec);
	}
	//迭代完留下tail的部分
	int tail_count = n - (epoch * slot_count);
	plains_vector vec(slot_count, plain_type(0));
	for (int i = 0; i < tail_count; i++) {
		vec[i] = *(p + i);
	}
	if (tail_count > 0) plains_list.push_back(vec);
}

DataInterface::DataInterface(cipher_list& ciphers, seal::SEALContext& context, const size_t slot_count = 4096) {
	for (int i = 0; i < ciphers.length; i++) {
		seal::Ciphertext cipher;
		cipher.load(context, reinterpret_cast<seal::seal_byte*>(ciphers.cipher_data[i].data), static_cast<size_t>(ciphers.cipher_data[i].length));
		this->cipher_lst.push_back(cipher);
	}
}

void DataInterface::encoding(seal::CKKSEncoder& encoder, double scale) {
	if (this->plains_list.size() == 0) {
		throw std::exception("没有可以编码的明文向量.\n");
	}
	for (plains_vector plain_vec : this->plains_list) {
		seal::Plaintext plain;
		encoder.encode(plain_vec, scale, plain);
		this->encode_list.push_back(plain);
	}
}

void DataInterface::encrypting(seal::Encryptor& encryptor) {
	if (this->encode_list.size() == 0) {
		throw std::exception("没有可以加密的encoded向量。\n");
	}
	for (auto encode_vec : this->encode_list) {
		seal::Ciphertext cipher{};
		encryptor.encrypt(encode_vec, cipher);
		this->cipher_lst.push_back(cipher);
	}
}


cipher_list DataInterface::to_cipher() {
	if (this->cipher_lst.size() == 0) {
		throw std::exception("没有可以转化的密文向量。\n");
	}
	cipher_list* result = new cipher_list;
	result->length = this->cipher_lst.size();
	result->cipher_data = new ser_cipher_t[result->length];
	for (int i = 0; i < result->length; i++) {
		//存储序列化的容器
		auto enctypted_cipher = this->cipher_lst[i];
		std::vector<seal::seal_byte> buffer(static_cast<size_t>(enctypted_cipher.save_size()));
		auto size = enctypted_cipher.save(reinterpret_cast<seal::seal_byte*>(buffer.data()), buffer.size());
		//返回的内置对象容器
		auto& ser = result->cipher_data[i];
		auto* data = new seal_type[buffer.size()];
		ser.length = static_cast<unsigned long long>(buffer.size());
		std::copy(reinterpret_cast<seal_type*>(buffer.data()), reinterpret_cast<seal_type*>(buffer.data()) + static_cast<unsigned long long>(buffer.size()), data);
		ser.data = data;
	}
	return *result;
}

void DataInterface::decrypting(seal::Decryptor& decryptor) {
	if (this->cipher_lst.size() == 0) {
		throw std::exception("没有可以解密的密文向量。\n");
	}
	for (auto cipher : this->cipher_lst) {
		seal::Plaintext plain;
		decryptor.decrypt(cipher, plain);
		this->encode_list.push_back(plain);
	}
}

void DataInterface::decoding(seal::CKKSEncoder& encoder, size_t slot_count = 4096) {
	if (this->encode_list.size() == 0) {
		throw std::exception("没有可以解码的encoded向量。\n");
	}
	for (auto plain : this->encode_list) {
		plains_vector py_plain(slot_count, 0ULL);
		encoder.decode(plain, py_plain);
		this->plains_list.push_back(py_plain);
	}
}

py_plain_list DataInterface::to_plain() {
	if (this->plains_list.size() == 0) {
		throw std::exception("没有可以转为py_plain_list的明文向量。\n");
	}
	py_plain_list* result = new py_plain_list;
	size_t vector_num = this->plains_list[0].size();
	size_t elemet_num = (this->plains_list.size()) * vector_num;
	//给result分配空间
	result->data = new plain_type[elemet_num];
	result->length = elemet_num;
	plain_type* p = result->data;
	for (int i = 0; i < this->plains_list.size(); i++) {
		auto vec = this->plains_list[i];
		std::copy(vec.begin(), vec.end(), p);
		p = p + vector_num;
	}
	return *result;
}

const DataInterface DataInterface::add(const DataInterface& cipher_inter, seal::Evaluator& evaluator, seal::RelinKeys& relin_key) const {
	if (this->cipher_lst.size() == 0 || cipher_inter.cipher_lst.size() == 0) {
		throw std::exception(((this->cipher_lst.size() == 0 ? "左" : "右") + std::string("接口数据为空。\n")).c_str());
	}
	DataInterface result;
	for (int i = 0; i < this->cipher_lst.size(); i++) {
		seal::Ciphertext cipher;
		evaluator.add(this->cipher_lst[i], cipher_inter.cipher_lst[i], cipher);
		evaluator.relinearize_inplace(cipher, relin_key);
		result.cipher_lst.push_back(cipher);
	}
	return result;
}

const DataInterface DataInterface::multiple(const DataInterface& cipher_inter, seal::Evaluator& evaluator, seal::RelinKeys& reline_key) const {
	if (this->cipher_lst.size() == 0 || cipher_inter.cipher_lst.size() == 0) {
		throw std::exception(((this->cipher_lst.size() == 0 ? "左" : "右") + std::string("接口数据为空。\n")).c_str());
	}
	DataInterface result;
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