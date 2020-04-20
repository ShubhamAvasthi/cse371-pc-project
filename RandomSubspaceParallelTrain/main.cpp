#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include <random>
#include <string>

const std::string data_files_extension = ".bin";
const std::string get_object_data_file_prefix(const int object_id);
const std::string get_training_dataset_file_name(const int object_id);
const std::string get_training_labels_file_name(const int object_id);
const std::string get_model_save_file_name(const int object_id, const int classifier_num);
const std::string get_model_sampled_features_save_file_name(const int object_id, const int classifier_num);

int main(int argc, char **argv)
{
	// Initialize the MPI environment
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const int num_classifiers = std::stoi(argv[1]);
	const int num_classes = std::stoi(argv[2]);
	const int ensemble_object_id = std::stoi(argv[3]);
	const int feature_sample_size = std::stoi(argv[4]);
	const int minimum_leaf_size = std::stoi(argv[5]);

	arma::mat train;
	train.load(get_training_dataset_file_name(ensemble_object_id));
	arma::Row<size_t> train_labels;
	train_labels.load(get_training_labels_file_name(ensemble_object_id));

	std::default_random_engine generator(world_rank);

	const std::uniform_int_distribution<> distribution(0, train.n_rows - 1); // Training examples are present columnwise

	for (int classifier_num = world_rank; classifier_num < num_classifiers; classifier_num += world_size)
	{
		arma::Col<size_t> sampled_features(feature_sample_size);
		arma::mat sampled_train(feature_sample_size, train.n_cols);
		for (int i = 0; i < feature_sample_size; i++)
		{
			int index = distribution(generator);
			sampled_train.row(i) = train.row(index);
			sampled_features(i) = index;
		}

		mlpack::tree::DecisionTree<> decision_tree(sampled_train, train_labels, num_classes, minimum_leaf_size);
		mlpack::data::Save(get_model_save_file_name(ensemble_object_id, classifier_num), "model", decision_tree);
		sampled_features.save(get_model_sampled_features_save_file_name(ensemble_object_id, classifier_num));
	}

	// Finalize the MPI environment.
	MPI_Finalize();
}

const std::string get_object_data_file_prefix(const int object_id)
{
	return "Ensemble_Object" + std::to_string(object_id);
}

const std::string get_training_dataset_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Training_Dataset" + data_files_extension;
}

const std::string get_training_labels_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Training_Labels" + data_files_extension;
}

const std::string get_model_save_file_name(const int object_id, const int classifier_num)
{
	return get_object_data_file_prefix(object_id) + "_Model_" + std::to_string(classifier_num) + data_files_extension;
}

const std::string get_model_sampled_features_save_file_name(const int object_id, const int classifier_num)
{
	return get_object_data_file_prefix(object_id) + "_Model_Sampled_Features_" + std::to_string(classifier_num) + data_files_extension;
}
