#pragma once

#include "Ensemble.h"

class ABagging : public Ensemble
{
private:
	static const std::string parallel_train_executable_name;
	static const std::string parallel_predict_executable_name;

	const std::string& get_parallel_train_executable_name();
	const std::string& get_parallel_predict_executable_name();
	const std::string get_model_save_file_name(const int model_num);
	const std::string get_model_type_save_file_name(const int model_num);
	const std::string get_model_accuracy_save_file_name(const int model_num);
	const std::string get_train_command();

public:
	ABagging(
		const arma::mat& train,
		const arma::Row<size_t>& train_labels,
		const int num_classifiers,
		const int num_classes = 2,
		const int num_processes = -1
	);
	~ABagging();
};
