#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
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

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const int ensemble_object_id = std::stoi(argv[1]);
	const int sample_size = std::stoi(argv[2]);
	const int num_classes = std::stoi(argv[3]);
	const int minimum_leaf_size = std::stoi(argv[4]);

	arma::mat train;
	train.load("BaggingParallelTrain_" + std::to_string(ensemble_object_id) + ".bin");
	arma::Row<size_t> train_labels;
	train_labels.load("BaggingParallelTrainLabels_" + std::to_string(ensemble_object_id) + ".bin");

	std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	const std::uniform_int_distribution<> distribution(0, train.n_cols - 1); // Training examples are present columnwise

	arma::mat sampled_train(train.n_rows, sample_size);
	arma::Row<size_t> sampled_labels(sample_size);
	for (int j = 0; j < sample_size; j++)
	{
		int index = distribution(generator);
		sampled_train.col(j) = train.col(index);
		sampled_labels[j] = train_labels[index];
	}

	mlpack::tree::DecisionTree<> decision_tree(sampled_train, sampled_labels, num_classes, minimum_leaf_size);

	mlpack::data::Save(
		"BaggingParallelModel_" + std::to_string(ensemble_object_id) + "_" + std::to_string(world_rank) + ".bin",
		"model",
		decision_tree
	);

	// Finalize the MPI environment.
	MPI_Finalize();
}
