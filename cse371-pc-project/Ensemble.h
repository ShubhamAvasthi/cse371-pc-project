#pragma once

#include "mlpack/core.hpp"
#include <string>

class Ensemble
{
private:
	static int object_id_cnt;

	const int object_id;
	const int num_classes;

	virtual const std::string& get_parallel_train_executable_name() = 0;   // Abstract function
	virtual const std::string& get_parallel_predict_executable_name() = 0; // Abstract function

	const std::string get_test_file_name();
	const std::string get_predictions_file_name();
	const std::string get_predict_command();

protected:
	static const std::string data_files_extension;
	
	const int num_classifiers;

	const std::string get_object_data_file_prefix();
	const std::string get_training_dataset_file_name();
	const std::string get_training_labels_file_name();
	const std::string get_train_command_prefix();
	void train(
		const arma::mat& train_dataset,
		const arma::Row<size_t>& train_labels,
		const std::string& train_command
	);
	
	Ensemble(const int _num_classifiers, const int _num_classes);

public:
	void predict(const arma::mat& X, arma::Row<size_t>& ensemble_predictions);
	double get_score(const arma::mat& test, const arma::Row<size_t>& labels);
};
