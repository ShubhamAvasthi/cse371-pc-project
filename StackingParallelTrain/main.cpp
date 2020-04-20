#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
#include <string>

const std::string data_files_extension = ".bin";
const std::string get_object_data_file_prefix(const int object_id);
const std::string get_training_dataset_file_name(const int object_id);
const std::string get_training_labels_file_name(const int object_id);
const std::string get_model_save_file_name(const int object_id, const int classifier_num);
const std::string get_meta_classifier_save_file_name(const int object_id);

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

	arma::mat train;
	train.load(get_training_dataset_file_name(ensemble_object_id));
	arma::Row<size_t> train_labels;
	train_labels.load(get_training_labels_file_name(ensemble_object_id));

	int *sub_predictions = nullptr;
	int predictions_size;
	int seg_size;

	for (int classifier_num = world_rank; classifier_num < num_classifiers; classifier_num += world_size)
	{
		arma::Row<size_t> predictions;
		if (classifier_num & 1)
		{
			mlpack::tree::DecisionTree<> decision_tree(train, train_labels, num_classes, 5);
			mlpack::data::Save(get_model_save_file_name(ensemble_object_id, classifier_num), "model", decision_tree);

			decision_tree.Classify(train, predictions);

			if (classifier_num == world_rank)
			{
				predictions_size = predictions.n_elem;
				seg_size = predictions_size * ((num_classifiers + world_size - 1) / world_size);
				sub_predictions = new int[seg_size];
			}
		}
		else
		{
			mlpack::regression::SoftmaxRegression softmax_regressor(train, train_labels, num_classes);
			mlpack::data::Save(get_model_save_file_name(ensemble_object_id, classifier_num), "model", softmax_regressor);

			softmax_regressor.Classify(train, predictions);

			if (classifier_num == world_rank)
			{
				predictions_size = predictions.n_elem;
				seg_size = predictions_size * ((num_classifiers + world_size - 1) / world_size);
				sub_predictions = new int[seg_size];
			}
		}

		for (int i = 0; i < predictions_size; i++)
			sub_predictions[predictions_size * (classifier_num / world_size) + i] = predictions[i];
	}

	int *all_predictions = nullptr;

	if (world_rank == 0)
		all_predictions = new int[world_size * seg_size];

	MPI_Gather(sub_predictions, seg_size, MPI_INT, all_predictions, seg_size, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0)
	{
		arma::mat meta_train(num_classifiers, predictions_size);

		for (int i = 0; i < predictions_size; i++)
			for (int j = 0; j < num_classifiers; j++)
				meta_train(j, i) = all_predictions[seg_size * (j % world_size) + predictions_size * (j / world_size) + i];

		mlpack::regression::SoftmaxRegression meta_classifier(meta_train, train_labels, num_classes);
		mlpack::data::Save(get_meta_classifier_save_file_name(ensemble_object_id), "model", meta_classifier);
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

const std::string get_meta_classifier_save_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Meta_Classifier" + data_files_extension;
}
