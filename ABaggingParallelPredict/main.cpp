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
const std::string get_model_accuracy_save_file_name(const int object_id, const int world_rank);

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

	size_t num_classes;
	int *accuracy = new int;

	try
	{
		mlpack::data::Load(get_model_save_file_name(ensemble_object_id, world_rank), "decision_tree_model", decision_tree, true);
		num_classes = decision_tree.NumClasses();
		arma::mat accuracy_mat;
		accuracy_mat.load(get_model_accuracy_save_file_name(ensemble_object_id, world_rank));
		*accuracy = accuracy_mat(0, 0);
		decision_tree.Classify(X, predictions);
	}
	catch (...)
	{
		mlpack::data::Load(get_model_save_file_name(ensemble_object_id, world_rank), "softmax_regressor_model", softmax_regressor, true);
		num_classes = softmax_regressor.NumClasses();
		arma::mat accuracy_mat;
		accuracy_mat.load(get_model_accuracy_save_file_name(ensemble_object_id, world_rank));
		*accuracy = accuracy_mat(0, 0);
		softmax_regressor.Classify(X, predictions);
	}

	int *sub_predictions = new int[predictions.n_elem];
	for (int i = 0; i < predictions.n_elem; i++)
		sub_predictions[i] = predictions[i];

	int *all_predictions = nullptr;
	int *all_accuracies = nullptr;

	if (world_rank == 0)
	{
		all_predictions = new int[world_size * predictions.n_elem];
		all_accuracies = new int[world_size];
	}

	MPI_Gather(sub_predictions, predictions.n_elem, MPI_INT, all_predictions, predictions.n_elem, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(accuracy, 1, MPI_INT, all_accuracies, 1, MPI_INT, 0, MPI_COMM_WORLD);

	delete sub_predictions;
	delete accuracy;

	if (world_rank == 0)
	{
		arma::Row<size_t> ensemble_predictions(predictions.n_elem);

		for (int i = 0; i < predictions.n_elem; i++)
		{
			std::vector<double> weight_predictions(num_classes);

			for (int j = 0; j < world_size; j++)
				weight_predictions[all_predictions[j * predictions.n_elem + i]] += all_accuracies[j];

			int max_weight_index = -1, max_weight = -1;
			for (int i = 0; i < num_classes; i++)
				if (weight_predictions[i] > max_weight)
				{
					max_weight = weight_predictions[i];
					max_weight_index = i;
				}
			ensemble_predictions[i] = max_weight_index;
		}
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

const std::string get_model_accuracy_save_file_name(const int object_id, const int world_rank)
{
	return get_object_data_file_prefix(object_id) + "_Model_Accuracy_" + std::to_string(world_rank) + data_files_extension;
}
