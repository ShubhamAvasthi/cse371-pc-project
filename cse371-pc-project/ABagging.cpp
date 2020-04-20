#include "ABagging.h"

const std::string ABagging::parallel_train_executable_name = "ABaggingParallelTrain.exe";
const std::string ABagging::parallel_predict_executable_name = "ABaggingParallelPredict.exe";

const std::string& ABagging::get_parallel_train_executable_name()
{
	return parallel_train_executable_name;
}

const std::string& ABagging::get_parallel_predict_executable_name()
{
	return parallel_predict_executable_name;
}

const std::string ABagging::get_model_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_" + std::to_string(model_num) + data_files_extension;
}

const std::string ABagging::get_model_type_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_Type_" + std::to_string(model_num) + data_files_extension;
}

const std::string ABagging::get_model_accuracy_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_Accuracy_" + std::to_string(model_num) + data_files_extension;
}

const std::string ABagging::get_train_command()
{
	return get_train_command_prefix();
}

ABagging::ABagging(
	const arma::mat& train_dataset,
	const arma::Row<size_t>& train_labels,
	const int num_classifiers,
	const int num_classes,
	const int num_processes
) : Ensemble(num_classifiers, num_classes, num_processes)
{
	train(train_dataset, train_labels, get_train_command());
}

ABagging::~ABagging()
{
	for (int i = 0; i < num_classifiers; i++)
	{
		std::remove(get_model_save_file_name(i).c_str());
		std::remove(get_model_type_save_file_name(i).c_str());
		std::remove(get_model_accuracy_save_file_name(i).c_str());
	}
}
