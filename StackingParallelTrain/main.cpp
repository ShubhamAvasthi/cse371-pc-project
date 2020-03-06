#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
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
	const int num_classes = std::stoi(argv[2]);

	arma::mat train;
	train.load("StackingParallelTrain_" + std::to_string(ensemble_object_id) + ".bin");
	arma::Row<size_t> train_labels;
	train_labels.load("StackingParallelTrainLabels_" + std::to_string(ensemble_object_id) + ".bin");

	arma::Row<size_t> predictions;

	int *sub_predictions = nullptr;

	if (world_rank == 0)
	{
		mlpack::tree::DecisionTree<> decision_tree(train, train_labels, num_classes, 5);

		mlpack::data::Save(
			"StackingParallelModel_" + std::to_string(ensemble_object_id) + "_" + std::to_string(world_rank) + ".bin",
			"model",
			decision_tree
		);

		decision_tree.Classify(train, predictions);

		sub_predictions = new int[predictions.n_elem];
		for (int i = 0; i < predictions.n_elem; i++)
			sub_predictions[i] = predictions[i];
	}
	else
	{
		mlpack::regression::SoftmaxRegression softmax_regressor(train, train_labels, num_classes);

		mlpack::data::Save(
			"StackingParallelModel_" + std::to_string(ensemble_object_id) + "_" + std::to_string(world_rank) + ".bin",
			"model",
			softmax_regressor
		);

		softmax_regressor.Classify(train, predictions);

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
		arma::mat meta_train(world_size, predictions.n_elem);

		for (int i = 0; i < predictions.n_elem; i++)
			for (int j = 0; j < world_size; j++)
				meta_train(j, i) = all_predictions[j * predictions.n_elem + i];

		mlpack::regression::SoftmaxRegression meta_classifier(meta_train, train_labels, num_classes);
		mlpack::data::Save(
			"StackingParallelMetaClassifierModel_" + std::to_string(ensemble_object_id) + ".bin",
			"model",
			meta_classifier
		);
	}

	// Finalize the MPI environment.
	MPI_Finalize();
}
