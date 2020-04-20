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
const std::string get_model_type_save_file_name(const int object_id, const int classifier_num);
const std::string get_model_accuracy_save_file_name(const int object_id, const int classifier_num);

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
	double *accuracy = nullptr;
	int predictions_size;
	int num_classes;
	int seg_size, seg_size_2;

	for (int classifier_num = world_rank; classifier_num < num_classifiers; classifier_num += world_size)
	{
		arma::Row<size_t> type;
		type.load(get_model_type_save_file_name(ensemble_object_id, classifier_num));
		arma::Row<size_t> predictions;

		if (type(0) == 0)
		{
			mlpack::tree::DecisionTree<> decision_tree;
			mlpack::data::Load(get_model_save_file_name(ensemble_object_id, classifier_num), "model", decision_tree, true);
			decision_tree.Classify(X, predictions);

			if (world_rank == 0 and classifier_num == world_rank)
				num_classes = decision_tree.NumClasses();
		}
		else
		{
			mlpack::regression::SoftmaxRegression softmax_regressor;
			mlpack::data::Load(get_model_save_file_name(ensemble_object_id, classifier_num), "model", softmax_regressor, true);
			softmax_regressor.Classify(X, predictions);

			if (world_rank == 0 and classifier_num == world_rank)
				num_classes = softmax_regressor.NumClasses();
		}

		if (classifier_num == world_rank)
		{
			predictions_size = predictions.n_elem;
			seg_size = predictions_size * ((num_classifiers + world_size - 1) / world_size);
			seg_size_2 = (num_classifiers + world_size - 1) / world_size;
			sub_predictions = new int[seg_size];
			accuracy = new double[seg_size_2];
		}

		for (int i = 0; i < predictions_size; i++)
			sub_predictions[predictions_size * (classifier_num / world_size) + i] = predictions[i];

		arma::mat accuracy_mat;
		accuracy_mat.load(get_model_accuracy_save_file_name(ensemble_object_id, classifier_num));
		accuracy[classifier_num / world_size] = accuracy_mat(0, 0);
	}

	int *all_predictions = nullptr;
	double *all_accuracies = nullptr;

	if (world_rank == 0)
	{
		all_predictions = new int[world_size * seg_size];
		all_accuracies = new double[world_size * seg_size_2];
	}

	MPI_Gather(sub_predictions, seg_size, MPI_INT, all_predictions, seg_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(accuracy, seg_size_2, MPI_DOUBLE, all_accuracies, seg_size_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete sub_predictions;
	delete accuracy;

	if (world_rank == 0)
	{
		arma::Row<size_t> ensemble_predictions(predictions_size);

		for (int i = 0; i < predictions_size; i++)
		{
			std::vector<double> weight_predictions(num_classes);

			for (int j = 0; j < num_classifiers; j++)
				weight_predictions[all_predictions[seg_size * (j % world_size) + predictions_size * (j / world_size) + i]] += all_accuracies[seg_size_2 * (j % world_size) + (j / world_size)];

			int max_weight_index = -1;
			double max_weight = -1;
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

const std::string get_model_save_file_name(const int object_id, const int classifier_num)
{
	return get_object_data_file_prefix(object_id) + "_Model_" + std::to_string(classifier_num) + data_files_extension;
}

const std::string get_model_type_save_file_name(const int object_id, const int classifier_num)
{
	return get_object_data_file_prefix(object_id) + "_Model_Type_" + std::to_string(classifier_num) + data_files_extension;
}

const std::string get_model_accuracy_save_file_name(const int object_id, const int classifier_num)
{
	return get_object_data_file_prefix(object_id) + "_Model_Accuracy_" + std::to_string(classifier_num) + data_files_extension;
}
