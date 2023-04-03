#include "pch.h"
#include "Plain_Interface.h"
#include<iostream>


Plain_Interface::~Plain_Interface() {

}


Plain_Interface::Plain_Interface(const plain_type* plain_ptr, const int n, const int slot_count = 4096) {
	using namespace std;
	const plain_type* p = plain_ptr;
	int epoch = 0;
	for(epoch = 0; epoch < int(n / slot_count); epoch++){
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
	if(tail_count > 0) plains_list.push_back(vec);
}

void Plain_Interface::encoding(seal::CKKSEncoder& encoder, double scale) {
	for (plains_vector plain_vec : this->plains_list) {
		seal::Plaintext plain;
		encoder.encode(plain_vec, scale, plain);
		this->encode_list.push_back(plain);
	}
}

void Plain_Interface::encrypting(seal::Encryptor& encryptor) {
	for (auto encode_vec : this->encode_list) {
		seal::Ciphertext cipher{};
		encryptor.encrypt(encode_vec, cipher);
		this->cipher_lst.push_back(cipher);
	}
}

cipher_list Plain_Interface::operator()() {
	cipher_list* result = new cipher_list;
	result->length = this->cipher_lst.size();
	result->cipher_data = new ser_cipher_t[result->length];
	for (int i = 0; i < result->length; i++) {
		//存储序列化的容器
		auto enctypted_cipher = this->cipher_lst[i];
		std::vector<seal::seal_byte> buffer(static_cast<size_t>(enctypted_cipher.save_size()));
		auto size = enctypted_cipher.save(reinterpret_cast<seal::seal_byte*>(buffer.data()), buffer.size());
		//返回的内置对象容器
		auto & ser = result->cipher_data[i];
		auto* data = new seal_type[buffer.size()];
		ser.length = static_cast<unsigned long long>(buffer.size());
		std::copy(reinterpret_cast<seal_type*>(buffer.data()), reinterpret_cast<seal_type*>(buffer.data()) + static_cast<unsigned long long>(buffer.size()), data);
		ser.data = data;
	}
	return *result;
}

cipher_list Plain_Interface::to_cipher_list(const std::vector<seal::Ciphertext>& ciphertext_vector) {
	Plain_Interface temp = *this;
	temp.cipher_lst = ciphertext_vector;
	return temp();
}