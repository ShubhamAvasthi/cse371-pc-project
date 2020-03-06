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

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const int ensemble_object_id = std::stoi(argv[1]);

	arma::mat X;
	X.load("BaggingParallelPredictX_" + std::to_string(ensemble_object_id) + ".bin");

	mlpack::tree::DecisionTree<> decision_tree;
	mlpack::data::Load(
		"BaggingParallelModel_" + std::to_string(ensemble_object_id) + "_" + std::to_string(world_rank) + ".bin",
		"model",
		decision_tree
	);

	arma::Row<size_t> predictions;
	decision_tree.Classify(X, predictions);

	int *sub_predictions = new int[predictions.n_elem];
	for (int i = 0; i < predictions.n_elem; i++)
		sub_predictions[i] = predictions[i];

	int *all_predictions = nullptr;
	
	if (world_rank == 0)
		all_predictions = new int[world_size * predictions.n_elem];

	MPI_Gather(sub_predictions, predictions.n_elem, MPI_INT, all_predictions, predictions.n_elem, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0)
	{
		arma::Row<size_t> ensemble_predictions(predictions.n_elem);

		for (int i = 0; i < predictions.n_elem; i++)
		{
			std::vector<int> num_predictions(decision_tree.NumClasses());

			for (int j = 0; j < world_size; j++)
				num_predictions[all_predictions[j * predictions.n_elem + i]]++;

			int max_freq_index = -1, max_freq = -1;
			for (int i = 0; i < decision_tree.NumClasses(); i++)
				if (num_predictions[i] > max_freq)
				{
					max_freq = num_predictions[i];
					max_freq_index = i;
				}
			ensemble_predictions[i] = max_freq_index;
		}
		ensemble_predictions.save(("BaggingParallelPredictions_" + std::to_string(ensemble_object_id) + ".bin").c_str());
		delete all_predictions;
	}

	delete sub_predictions;

	// Finalize the MPI environment.
	MPI_Finalize();
}
