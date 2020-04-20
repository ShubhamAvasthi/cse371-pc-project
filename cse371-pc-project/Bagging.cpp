#include "Bagging.h"

const std::string Bagging::parallel_train_executable_name = "BaggingParallelTrain.exe";
const std::string Bagging::parallel_predict_executable_name = "BaggingParallelPredict.exe";

const std::string& Bagging::get_parallel_train_executable_name()
{
	return parallel_train_executable_name;
}

const std::string& Bagging::get_parallel_predict_executable_name()
{
	return parallel_predict_executable_name;
}

const std::string Bagging::get_model_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_" + std::to_string(model_num) + data_files_extension;
}

const std::string Bagging::get_train_command(const int sample_size, const int minimum_leaf_size)
{
	return get_train_command_prefix() + ' ' + std::to_string(sample_size) + ' ' + std::to_string(minimum_leaf_size);
}

Bagging::Bagging(
	const arma::mat& train_dataset,
	const arma::Row<size_t>& train_labels,
	const int num_classifiers,
	const int sample_size,
	const int num_classes,
	const int minimum_leaf_size,
	const int num_processes
) : Ensemble(num_classifiers, num_classes, num_processes)
{
	train(train_dataset, train_labels, get_train_command(sample_size, minimum_leaf_size));
}

Bagging::~Bagging()
{
	for (int i = 0; i < num_classifiers; i++)
		std::remove(get_model_save_file_name(i).c_str());
}
