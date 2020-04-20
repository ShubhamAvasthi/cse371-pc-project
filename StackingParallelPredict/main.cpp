#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
#include <string>

const std::string data_files_extension = ".bin";
const std::string get_object_data_file_prefix(const int object_id);
const std::string get_test_file_name(const int object_id);
const std::string get_predictions_file_name(const int object_id);
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
	const int ensemble_object_id = std::stoi(argv[2]);

	arma::mat X;
	X.load(get_test_file_name(ensemble_object_id));

	int *sub_predictions = nullptr;
	int predictions_size;
	int seg_size;

	for (int classifier_num = world_rank; classifier_num < num_classifiers; classifier_num += world_size)
	{
		arma::Row<size_t> predictions;
		if (classifier_num & 1)
		{
			mlpack::tree::DecisionTree<> decision_tree;
			mlpack::data::Load(get_model_save_file_name(ensemble_object_id, classifier_num), "model", decision_tree);

			decision_tree.Classify(X, predictions);

			if (classifier_num == world_rank)
			{
				predictions_size = predictions.n_elem;
				seg_size = predictions_size * ((num_classifiers + world_size - 1) / world_size);
				sub_predictions = new int[seg_size];
			}
		}
		else
		{
			mlpack::regression::SoftmaxRegression softmax_regressor;
			mlpack::data::Load(get_model_save_file_name(ensemble_object_id, classifier_num), "model", softmax_regressor);

			softmax_regressor.Classify(X, predictions);

			if (classifier_num == world_rank)
			{
				predictions_size = predictions.n_elem;
				seg_size = predictions_size * ((num_classifiers + world_size - 1) / world_size);
				sub_predictions = new int[seg_size];
			}
		}

		for (int i = 0; i < predictions.n_elem; i++)
			sub_predictions[predictions_size * (classifier_num / world_size) + i] = predictions[i];
	}

	int *all_predictions = nullptr;

	if (world_rank == 0)
		all_predictions = new int[world_size * seg_size];

	MPI_Gather(sub_predictions, seg_size, MPI_INT, all_predictions, seg_size, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0)
	{
		arma::Row<size_t> ensemble_predictions;
		
		arma::mat meta_X(num_classifiers, predictions_size);

		for (int i = 0; i < predictions_size; i++)
			for (int j = 0; j < num_classifiers; j++)
				meta_X(j, i) = all_predictions[seg_size * (j % world_size) + predictions_size * (j / world_size) + i];

		mlpack::regression::SoftmaxRegression meta_classifier;
		mlpack::data::Load(get_meta_classifier_save_file_name(ensemble_object_id), "model", meta_classifier);

		meta_classifier.Classify(meta_X, ensemble_predictions);

		ensemble_predictions.save(get_predictions_file_name(ensemble_object_id).c_str());
		delete all_predictions;
	}

	delete sub_predictions;

	// Finalize the MPI environment.
	MPI_Finalize();
}

const std::string get_object_data_file_prefix(const int object_id)
{
	return "Ensemble_Object" + std::to_string(object_id);
}

const std::string get_test_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Test" + data_files_extension;
}

const std::string get_predictions_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Predictions" + data_files_extension;
}

const std::string get_model_save_file_name(const int object_id, const int classifier_num)
{
	return get_object_data_file_prefix(object_id) + "_Model_" + std::to_string(classifier_num) + data_files_extension;
}

const std::string get_meta_classifier_save_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Meta_Classifier" + data_files_extension;
}
