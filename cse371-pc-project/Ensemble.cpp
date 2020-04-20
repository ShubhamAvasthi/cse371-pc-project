#include "Ensemble.h"
#include <chrono>

int Ensemble::object_id_cnt = 0;

const std::string Ensemble::get_test_file_name()
{
	return get_object_data_file_prefix() + "_Test" + data_files_extension;
}

const std::string Ensemble::get_predictions_file_name()
{
	return get_object_data_file_prefix() + "_Predictions" + data_files_extension;
}

const std::string Ensemble::get_predict_command()
{
	return "mpiexec -n " + std::to_string(num_processes) + ' ' + get_parallel_predict_executable_name() + ' ' + std::to_string(num_classifiers) + ' ' + std::to_string(object_id);
}

const std::string Ensemble::data_files_extension = ".bin";

const std::string Ensemble::get_object_data_file_prefix()
{
	return "Ensemble_Object" + std::to_string(object_id);
}

const std::string Ensemble::get_training_dataset_file_name()
{
	return get_object_data_file_prefix() + "_Training_Dataset" + data_files_extension;
}

const std::string Ensemble::get_training_labels_file_name()
{
	return get_object_data_file_prefix() + "_Training_Labels" + data_files_extension;
}

const std::string Ensemble::get_train_command_prefix()
{
	return "mpiexec -n " + std::to_string(num_processes) + ' ' + get_parallel_train_executable_name() + ' ' + std::to_string(num_classifiers) + ' ' + std::to_string(num_classes) + ' ' + std::to_string(object_id);
}

void Ensemble::train(
	const arma::mat& train_dataset,
	const arma::Row<size_t>& train_labels,
	const std::string& train_command
)
{
	train_dataset.save(get_training_dataset_file_name());
	train_labels.save(get_training_labels_file_name());

	auto start_time = std::chrono::high_resolution_clock::now();

	std::system(train_command.c_str());

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration time_duration = end_time - start_time;

	auto h = std::chrono::duration_cast<std::chrono::hours>(time_duration);        time_duration -= h;
	auto m = std::chrono::duration_cast<std::chrono::minutes>(time_duration);      time_duration -= m;
	auto s = std::chrono::duration_cast<std::chrono::seconds>(time_duration);      time_duration -= s;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_duration); time_duration -= ms;
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(time_duration); time_duration -= us;
	auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_duration);

	std::cerr << "Execution Time: "
		<< h.count() << " hours "
		<< m.count() << " minutes "
		<< s.count() << " seconds "
		<< ms.count() << " milliseconds "
		<< us.count() << " microseconds "
		<< ns.count() << " nanoseconds\n";

	std::remove(get_training_dataset_file_name().c_str());
	std::remove(get_training_labels_file_name().c_str());
}

Ensemble::Ensemble(
	const int _num_classifiers,
	const int _num_classes,
	const int _num_processes
) : object_id(object_id_cnt++),
	num_classifiers(_num_classifiers),
	num_classes(_num_classes),
	num_processes(_num_processes == -1 ? _num_classifiers : _num_processes)
{}

void Ensemble::predict(const arma::mat& X, arma::Row<size_t>& ensemble_predictions)
{
	X.save(get_test_file_name());
	auto start_time = std::chrono::high_resolution_clock::now();

	std::system(get_predict_command().c_str());

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration time_duration = end_time - start_time;

	auto h = std::chrono::duration_cast<std::chrono::hours>(time_duration);        time_duration -= h;
	auto m = std::chrono::duration_cast<std::chrono::minutes>(time_duration);      time_duration -= m;
	auto s = std::chrono::duration_cast<std::chrono::seconds>(time_duration);      time_duration -= s;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_duration); time_duration -= ms;
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(time_duration); time_duration -= us;
	auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_duration);

	std::cerr << "Execution Time: "
		<< h.count() << " hours "
		<< m.count() << " minutes "
		<< s.count() << " seconds "
		<< ms.count() << " milliseconds "
		<< us.count() << " microseconds "
		<< ns.count() << " nanoseconds\n";

	std::remove(get_test_file_name().c_str());
	ensemble_predictions.load(get_predictions_file_name().c_str());
	std::remove(get_predictions_file_name().c_str());
}

double Ensemble::get_score(const arma::mat& test, const arma::Row<size_t>& labels)
{
	arma::Row<size_t> predictions;
	predict(test, predictions);
	const size_t correct = arma::accu(predictions == labels);
	return correct / double(labels.n_elem);
}
