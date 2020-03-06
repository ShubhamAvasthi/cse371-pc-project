#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
#include <chrono>
#include <random>
#include <string>

int main(int argc, char **argv)
{
	// Initialize the MPI environment
	MPI_Init(&argc, &argv);

	/*
	for (int i = 0; i < argc; i++)
		printf("%s ", argv[i]);
	printf("\n");
	*/

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const int ensemble_object_id = std::stoi(argv[1]);

	arma::mat X;
	X.load("StackingParallelPredictX_" + std::to_string(ensemble_object_id) + ".bin");

	arma::Row<size_t> predictions;

	int *sub_predictions = nullptr;

	int num_classes;

	if (world_rank == 0)
	{
		mlpack::tree::DecisionTree<> decision_tree;
		mlpack::data::Load(
			"StackingParallelModel_" + std::to_string(ensemble_object_id) + "_" + std::to_string(world_rank) + ".bin",
			"model",
			decision_tree
		);

		decision_tree.Classify(X, predictions);

		sub_predictions = new int[predictions.n_elem];
		for (int i = 0; i < predictions.n_elem; i++)
			sub_predictions[i] = predictions[i];
		num_classes = decision_tree.NumClasses();
	}
	else
	{
		mlpack::regression::SoftmaxRegression softmax_regressor;
		mlpack::data::Load(
			"StackingParallelModel_" + std::to_string(ensemble_object_id) + "_" + std::to_string(world_rank) + ".bin",
			"model",
			softmax_regressor
		);

		softmax_regressor.Classify(X, predictions);

		sub_predictions = new int[predictions.n_elem];
		for (int i = 0; i < predictions.n_elem; i++)
			sub_predictions[i] = predictions[i];
	}

	int *all_predictions = nullptr;

	if (world_rank == 0)
		all_predictions = new int[world_size * predictions.n_elem];

	MPI_Gather(sub_predictions, predictions.n_elem, MPI_INT, all_predictions, predictions.n_elem, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0)
	{
		arma::Row<size_t> ensemble_predictions;
		
		arma::mat meta_X(world_size, predictions.n_elem);

		for (int i = 0; i < predictions.n_elem; i++)
			for (int j = 0; j < world_size; j++)
				meta_X(j, i) = all_predictions[j * predictions.n_elem + i];

		mlpack::regression::SoftmaxRegression meta_classifier;
		mlpack::data::Load(
			"StackingParallelMetaClassifierModel_" + std::to_string(ensemble_object_id) + ".bin",
			"model",
			meta_classifier
		);

		meta_classifier.Classify(meta_X, ensemble_predictions);

		ensemble_predictions.save(("StackingParallelPredictions_" + std::to_string(ensemble_object_id) + ".bin").c_str());
		delete all_predictions;
	}

	delete sub_predictions;

	// Finalize the MPI environment.
	MPI_Finalize();
}
