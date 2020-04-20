#include "Stacking.h"

const std::string Stacking::parallel_train_executable_name = "StackingParallelTrain.exe";
const std::string Stacking::parallel_predict_executable_name = "StackingParallelPredict.exe";

const std::string& Stacking::get_parallel_train_executable_name()
{
	return parallel_train_executable_name;
}

const std::string& Stacking::get_parallel_predict_executable_name()
{
	return parallel_predict_executable_name;
}

const std::string Stacking::get_model_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_" + std::to_string(model_num) + data_files_extension;
}

const std::string Stacking::get_meta_classifier_save_file_name()
{
	return get_object_data_file_prefix() + "_Meta_Classifier" + data_files_extension;
}

const std::string Stacking::get_train_command()
{
	return get_train_command_prefix();
}

Stacking::Stacking(
	const arma::mat& train_dataset,
	const arma::Row<size_t>& train_labels,
	const int num_classifiers,
	const int num_classes,
	const int num_processes
) : Ensemble(num_classifiers, num_classes, num_processes)
{
	train(train_dataset, train_labels, get_train_command());
}

Stacking::~Stacking()
{
	for (int i = 0; i < num_classifiers; i++)
		std::remove(get_model_save_file_name(i).c_str());
	std::remove(get_meta_classifier_save_file_name().c_str());
}
