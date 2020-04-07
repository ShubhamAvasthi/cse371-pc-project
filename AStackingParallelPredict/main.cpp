#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
#include <string>

const std::string data_files_extension = ".bin";
const std::string get_object_data_file_prefix(const int object_id);
const std::string get_test_file_name(const int object_id);
const std::string get_predictions_file_name(const int object_id);
const std::string get_model_save_file_name(const int object_id, const int world_rank);
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

	const int ensemble_object_id = std::stoi(argv[1]);

	arma::mat X;
	X.load(get_test_file_name(ensemble_object_id));

	arma::Row<size_t> predictions;
	mlpack::tree::DecisionTree<> decision_tree;
	mlpack::regression::SoftmaxRegression softmax_regressor;

	try
	{
		mlpack::data::Load(get_model_save_file_name(ensemble_object_id, world_rank), "decision_tree_model", decision_tree, true);
		decision_tree.Classify(X, predictions);
	}
	catch (...)
	{
		mlpack::data::Load(get_model_save_file_name(ensemble_object_id, world_rank), "softmax_regressor_model", softmax_regressor, true);
		softmax_regressor.Classify(X, predictions);
	}

	int *sub_predictions = new int[predictions.n_elem];
	for (int i = 0; i < predictions.n_elem; i++)
		sub_predictions[i] = predictions[i];

	int *all_predictions = nullptr;

	if (world_rank == 0)
		all_predictions = new int[world_size * predictions.n_elem];

	MPI_Gather(sub_predictions, predictions.n_elem, MPI_INT, all_predictions, predictions.n_elem, MPI_INT, 0, MPI_COMM_WORLD);

	delete sub_predictions;

	if (world_rank == 0)
	{
		arma::mat meta_X(world_size, predictions.n_elem);

		for (int i = 0; i < predictions.n_elem; i++)
			for (int j = 0; j < world_size; j++)
				meta_X(j, i) = all_predictions[j * predictions.n_elem + i];

		mlpack::regression::SoftmaxRegression meta_classifier;
		mlpack::data::Load(get_meta_classifier_save_file_name(ensemble_object_id), "model", meta_classifier);

		arma::Row<size_t> ensemble_predictions;
		meta_classifier.Classify(meta_X, ensemble_predictions);

		ensemble_predictions.save(get_predictions_file_name(ensemble_object_id).c_str());
		delete all_predictions;
	}

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

const std::string get_model_save_file_name(const int object_id, const int world_rank)
{
	return get_object_data_file_prefix(object_id) + "_Model_" + std::to_string(world_rank) + data_files_extension;
}

const std::string get_meta_classifier_save_file_name(const int object_id)
{
	return get_object_data_file_prefix(object_id) + "_Meta_Classifier" + data_files_extension;
}
