#include "RandomSubspace.h"

const std::string RandomSubspace::parallel_train_executable_name = "RandomSubspaceParallelTrain.exe";
const std::string RandomSubspace::parallel_predict_executable_name = "RandomSubspaceParallelPredict.exe";

const std::string& RandomSubspace::get_parallel_train_executable_name()
{
	return parallel_train_executable_name;
}

const std::string& RandomSubspace::get_parallel_predict_executable_name()
{
	return parallel_predict_executable_name;
}

const std::string RandomSubspace::get_model_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_" + std::to_string(model_num) + data_files_extension;
}

const std::string RandomSubspace::get_model_sampled_features_save_file_name(const int model_num)
{
	return get_object_data_file_prefix() + "_Model_Sampled_Features_" + std::to_string(model_num) + data_files_extension;
}

const std::string RandomSubspace::get_train_command(const int feature_sample_size, const int minimum_leaf_size)
{
	return get_train_command_prefix() + ' ' + std::to_string(feature_sample_size) + ' ' + std::to_string(minimum_leaf_size);
}

RandomSubspace::RandomSubspace(
	const arma::mat& train_dataset,
	const arma::Row<size_t>& train_labels,
	const int num_classifiers,
	const int feature_sample_size,
	const int num_classes,
	const int minimum_leaf_size
) : Ensemble(num_classifiers, num_classes)
{
	train(train_dataset, train_labels, get_train_command(feature_sample_size, minimum_leaf_size));
}

RandomSubspace::~RandomSubspace()
{
	for (int i = 0; i < num_classifiers; i++)
	{
		std::remove(get_model_save_file_name(i).c_str());
		std::remove(get_model_sampled_features_save_file_name(i).c_str());
	}
}
