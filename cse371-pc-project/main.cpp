#include "mlpack/core.hpp"
#include "mlpack/core/data/split_data.hpp"
#include <array>
#include <cstdlib>
#include <iostream>

#include "Bagging.h"
#include "Stacking.h"

int main()
{	
	std::cout <<
		"Enter the ensemble to test: \n" \
		"1. Bagging\n" \
		"2. Stacking\n"
		"\n";

	int choice;
	std::cin >> choice;

	if (choice != 1 and choice !=  2)
	{
		std::cout << "You entered the wrong choice. Exiting.\n";
		std::system("pause");
		return 0;
	}

	std::array<std::string, 4> training_datasets = {
		"Test_Datasets\\11_Bio_Cat2_Avg_training.csv",
		"Test_Datasets\\11_Bio_Cat2_Max_training.csv",
		"Test_Datasets\\11_Dig_Cat2_Avg_training.csv",
		"Test_Datasets\\11_Dig_Cat2_Max_training.csv"
	};

	std::array<std::string, 4> test_datasets = {
		"Test_Datasets\\11_Bio_Cat2_Avg_evaluation.csv",
		"Test_Datasets\\11_Bio_Cat2_Max_evaluation.csv",
		"Test_Datasets\\11_Dig_Cat2_Avg_evaluation.csv",
		"Test_Datasets\\11_Dig_Cat2_Max_evaluation.csv"
	};

	std::array<bool, 4> labels_first = { false, true, true, true };

	for (int i = 0; i < 4; i++)
	{
		arma::mat train, test;
		mlpack::data::Load(training_datasets[i], train);
		mlpack::data::Load(test_datasets[i], test);

		// Ignore the first row of the input (the first column of the matrix)
		train.shed_col(0);
		test.shed_col(0);

		int labels_row = (labels_first[i] ? 0 : train.n_rows - 1);
		const arma::Row<size_t> train_labels(arma::conv_to<arma::Row<size_t>>::from(train.row(labels_row)));
		train.shed_row(labels_row);

		labels_row = (labels_first[i] ? 0 : test.n_rows - 1);
		const arma::Row<size_t> test_labels(arma::conv_to<arma::Row<size_t>>::from(test.row(labels_row)));
		test.shed_row(labels_row);

		if (choice == 1) // Bagging
		{
			Bagging bagging(train, train_labels, 4, 400, 2);
			std::cout << "Training Accuracy: " << bagging.get_score(train, train_labels) << '\n';
			std::cout << "Test Accuracy: " << bagging.get_score(test, test_labels) << '\n';
		}
		else if (choice == 2) // Stacking
		{
			Stacking stacking(train, train_labels, 2, 400, 2);
			std::cout << "Training Accuracy: " << stacking.get_score(train, train_labels) << '\n';
			std::cout << "Test Accuracy: " << stacking.get_score(test, test_labels) << '\n';
		}
	}
	std::system("pause");
}
