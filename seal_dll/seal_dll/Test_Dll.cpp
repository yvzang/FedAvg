#define TEST_DLL

#include "pch.h"
#include<iostream>
#include<sstream>
#include<vector>
#include<math.h>
#include<iterator>
#include"DataInterface.h"
#include "Test_Dll.h"


template <class S>
data_buffer seriazation2struct(const S & ser) {
	using namespace seal;
	//存储序列化的容器
	std::vector<seal_byte> buffer(static_cast<size_t>(ser.save_size()));
	auto size = ser.save(reinterpret_cast<seal_byte*>(buffer.data()), buffer.size());
	//返回的内置对象容器
	data_buffer* data = new data_buffer{ new seal_type[buffer.size()], static_cast<unsigned long long>(size)};
	std::copy(reinterpret_cast<seal_type*>(buffer.data()), reinterpret_cast<seal_type*>(buffer.data()) + static_cast<unsigned long long>(size), data->data);
	return *data;
}

/*将序列化对象加载为类类型
类类型必须包含load()方法*/
template <class T>
T& seriazation2class(T & class_type, data_buffer ser) {
	using namespace seal;
	class_type.load(reinterpret_cast<seal_byte*>(ser.data), ser.length);
	return class_type;
}

template <class T>
T& seriazation2class_c(T& class_type, const seal::SEALContext& context, data_buffer ser) {
	using namespace seal;
	class_type.load(context, reinterpret_cast<seal_byte*>(ser.data), ser.length);
	return class_type;
}


ser_params_t get_parameter_seriazation() {
	using namespace seal;
	EncryptionParameters params(scheme_type::ckks);
	size_t poly_modulus_degree = 8192;
	params.set_poly_modulus_degree(poly_modulus_degree);
	params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

	return seriazation2struct(params);
}


ser_keys get_keys(ser_params_t parameters_ptr) {
	using namespace seal;
	using namespace std;
	EncryptionParameters params;
	params.load(reinterpret_cast<seal_byte*>(parameters_ptr.data), parameters_ptr.length);
	SEALContext context(params);
	KeyGenerator generator(context);
	Serializable<PublicKey>pbky = generator.create_public_key();
	SecretKey skky = generator.secret_key();
	auto rlky = generator.create_relin_keys();
	
	ser_public_key_t data_pbky = seriazation2struct(pbky);
	ser_secret_key_t data_skky = seriazation2struct(skky);
	ser_relin_key_t data_rlky = seriazation2struct(rlky);
	keys* pb_and_sec_keys = new keys{ data_pbky, data_skky, data_rlky };
	return *pb_and_sec_keys;

}


cipher_list encryption(py_plain_list plains, ser_params_t ser_params, ser_public_key_t ser_public_key) {
	using namespace seal;
	using namespace std;
	//还原参数
	EncryptionParameters params;
	seriazation2class(params, ser_params);
	//创建context
	SEALContext context(params);
	//创建encoder
	double scale = pow(2.0, 40);
	CKKSEncoder encoder(context);

	DataInterface plain_inter(plains.data, plains.length, encoder.slot_count());
	//转化为Plaintext
	plain_inter.encoding(encoder, scale);
	//还原公钥，加密
	PublicKey pbky; 
	seriazation2class_c(pbky, context, ser_public_key);
	Encryptor encryptor(context, pbky);
	plain_inter.encrypting(encryptor);

	return plain_inter.to_cipher();
}

py_plain_list decryption(cipher_list ser_cipher, ser_params_t ser_params, ser_secret_key_t ser_sercret_key) {
	using namespace seal;
	//还原参数
	EncryptionParameters params;
	seriazation2class<EncryptionParameters>(params, ser_params);
	//创建context
	SEALContext context(params);
	//创建encoder
	CKKSEncoder encoder(context);
	DataInterface cipher_inter(ser_cipher, context, encoder.slot_count());
	//还原私钥，解密
	SecretKey skky;
	seriazation2class_c(skky, context, ser_sercret_key);
	Decryptor decryptor(context, skky);
	Plaintext plain;
	cipher_inter.decrypting(decryptor);
	//decode
	cipher_inter.decoding(encoder, encoder.slot_count());
	//vector to plain list
	return cipher_inter.to_plain();
}

cipher_list addition(cipher_list cipher_l, cipher_list cipher_r, ser_params_t ser_params, ser_relin_key_t ser_relin_key) {
	using namespace seal;
	using namespace std;
	//还原参数
	EncryptionParameters params;
	seriazation2class<EncryptionParameters>(params, ser_params);
	//创建context
	SEALContext context(params);
	//创建evaluator
	Evaluator evaluator(context);
	//创建relin_key
	RelinKeys relin_key;
	seriazation2class_c<RelinKeys>(relin_key, context, ser_relin_key);
	//add
	DataInterface inter_l(cipher_l, context, 4096);
	DataInterface inter_r(cipher_r, context, 4096);
	DataInterface inter_result = inter_l.add(inter_r, evaluator, relin_key);
	return inter_result.to_cipher();
}

cipher_list mutiplication(cipher_list cipher_l, cipher_list cipher_r, ser_params_t ser_params, ser_relin_key_t ser_relin_key) {
	using namespace seal;
	using namespace std;
	//还原参数
	EncryptionParameters params;
	seriazation2class<EncryptionParameters>(params, ser_params);
	//创建context
	SEALContext context(params);
	//创建evaluator
	Evaluator valuator(context);
	//relin_key
	RelinKeys relin_key;
	seriazation2class_c<RelinKeys>(relin_key, context, ser_relin_key);
	//multiple
	DataInterface cipher_inter_l(cipher_l, context, 4096);
	DataInterface cipher_inter_r(cipher_r, context, 4096);
	DataInterface inter_result = cipher_inter_l.multiple(cipher_inter_r, valuator, relin_key);
	return inter_result.to_cipher();
}

void delete_cipher_list(cipher_list& ciphers) {
	for (int i = 0; i < ciphers.length; i++) {
		delete[] ciphers.cipher_data[i].data;
	}
	delete[] ciphers.cipher_data;
	delete& ciphers;
}

void delete_py_plain_list(py_plain_list& plains) {
	delete[] plains.data;
	delete& plains;
}