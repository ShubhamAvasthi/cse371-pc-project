#include "mpi.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/kmeans/kmeans.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include <string>
#include <vector>

const std::string data_files_extension = ".bin";
const std::string get_object_data_file_prefix(const int object_id);
const std::string get_training_dataset_file_name(const int object_id);
const std::string get_training_labels_file_name(const int object_id);
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

	const int num_classes = std::stoi(argv[1]);
	const int ensemble_object_id = std::stoi(argv[2]);

	// if (world_rank == 0)
	// {
	arma::mat train;
	train.load(get_training_dataset_file_name(ensemble_object_id));
	arma::Row<size_t> train_labels;
	train_labels.load(get_training_labels_file_name(ensemble_object_id));

	arma::Row<size_t> assignments;
	mlpack::kmeans::KMeans<> k_means;
	k_means.Cluster(train, world_size, assignments);
	// }

	std::vector<arma::mat> clustered_train(world_size);
	std::vector<arma::Row<size_t>> clustered_train_labels(world_size);
	for (int j = 0; j < assignments.n_elem; j++)
	{
		arma::mat& train_mat = clustered_train[assignments(j)];
		train_mat.insert_cols(train_mat.n_cols, train.col(j));
		arma::Row<size_t>& train_labels_row = clustered_train_labels[assignments(j)];
		train_labels_row.insert_cols(train_labels_row.n_elem, train_labels.col(j));
	}

	mlpack::cv::KFoldCV<mlpack::tree::DecisionTree<>, mlpack::cv::Accuracy> cv_dt(10, clustered_train[world_rank], clustered_train_labels[world_rank], num_classes, true);
	mlpack::cv::KFoldCV<mlpack::regression::SoftmaxRegression, mlpack::cv::Accuracy> cv_sr(10, clustered_train[world_rank], clustered_train_labels[world_rank], num_classes, true);

	double cv_dt_score = cv_dt.Evaluate(5);
	double cv_sr_score = cv_sr.Evaluate();
	if (cv_dt_score > cv_sr_score)
	{
		mlpack::tree::DecisionTree<> decision_tree(clustered_train[world_rank], clustered_train_labels[world_rank], num_classes, 5);
		mlpack::data::Save(get_model_save_file_name(ensemble_object_id, world_rank), "decision_tree_model", decision_tree);
		arma::mat accuracy(1, 1);
		accuracy(0, 0) = cv_dt_score;
		accuracy.save(get_model_accuracy_save_file_name(ensemble_object_id, world_rank));
	}
	else
	{
		mlpack::regression::SoftmaxRegression softmax_regressor(clustered_train[world_rank], clustered_train_labels[world_rank], num_classes);
		mlpack::data::Save(get_model_save_file_name(ensemble_object_id, world_rank), "softmax_regressor_model", softmax_regressor);
		arma::mat accuracy(1, 1);
		accuracy(0, 0) = cv_sr_score;
		accuracy.save(get_model_accuracy_save_file_name(ensemble_object_id, world_rank));
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

const std::string get_model_save_file_name(const int object_id, const int world_rank)
{
	return get_object_data_file_prefix(object_id) + "_Model_" + std::to_string(world_rank) + data_files_extension;
}

const std::string get_model_accuracy_save_file_name(const int object_id, const int world_rank)
{
	return get_object_data_file_prefix(object_id) + "_Model_Accuracy_" + std::to_string(world_rank) + data_files_extension;
}
