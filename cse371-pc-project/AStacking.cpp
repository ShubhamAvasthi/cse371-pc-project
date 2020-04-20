#include "AStacking.h"

const std::string AStacking::parallel_train_executable_name = "AStackingParallelTrain.exe";
const std::string AStacking::parallel_predict_executable_name = "AStackingParallelPredict.exe";

const std::string& AStacking::get_parallel_train_executable_name()
{
	return parallel_train_executable_name;
}

const std::string& AStacking::get_parallel_predict_executable_name()
{
	return parallel_predict_executable_name;
}

const std::string AStacking::get_model_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_" + std::to_string(model_num) + data_files_extension;
}

const std::string AStacking::get_model_type_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_Type_" + std::to_string(model_num) + data_files_extension;
}

const std::string AStacking::get_meta_classifier_save_file_name()
{
	return get_object_data_file_prefix() + "_Meta_Classifier" + data_files_extension;
}

const std::string AStacking::get_train_command()
{
	return get_train_command_prefix();
}

AStacking::AStacking(
	const arma::mat& train_dataset,
	const arma::Row<size_t>& train_labels,
	const int num_classifiers,
	const int num_classes,
	const int num_processes
) : Ensemble(num_classifiers, num_classes, num_processes)
{
	train(train_dataset, train_labels, get_train_command());
}

AStacking::~AStacking()
{
	for (int i = 0; i < num_classifiers; i++)
	{
		std::remove(get_model_save_file_name(i).c_str());
		std::remove(get_model_type_save_file_name(i).c_str());
	}
	std::remove(get_meta_classifier_save_file_name().c_str());
}
