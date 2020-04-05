#pragma once

#include "Ensemble.h"

class Stacking : public Ensemble
{
private:
	static const std::string parallel_train_executable_name;
	static const std::string parallel_predict_executable_name;

	const std::string& get_parallel_train_executable_name();
	const std::string& get_parallel_predict_executable_name();
	const std::string get_model_save_file_name(const int model_num);
	const std::string get_meta_classifier_save_file_name();
	const std::string get_train_command();

public:
	Stacking(
		const arma::mat& train,
		const arma::Row<size_t>& train_labels,
		const int num_classifiers = 4,
		const int num_classes = 2
	);
	~Stacking();
};
