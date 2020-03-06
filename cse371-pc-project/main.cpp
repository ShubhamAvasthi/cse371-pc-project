#include "mlpack/core.hpp"
#include "mlpack/core/data/split_data.hpp"
#include <cstdlib>
#include <iostream>

#include "Bagging.h"

using arma::mat;
using mlpack::data::Load;
using mlpack::data::Split;

int main()
{
	mat dataset, train, test;
	Load("german.csv", dataset);
	Split(dataset, train, test, 0.3);

	const arma::Row<size_t> train_labels(arma::conv_to<arma::Row<size_t>>::from(train.row(train.n_rows - 1)));
	train.shed_row(train.n_rows - 1);
	const arma::Row<size_t> test_labels(arma::conv_to<arma::Row<size_t>>::from(test.row(test.n_rows - 1)));
	test.shed_row(test.n_rows - 1);
	
	std::cout <<
		"Enter the ensemble to test: \n" \
		"1. Bagging\n" \
		"\n";
	
	int choice;
	std::cin >> choice;

	if (choice == 1) // Bagging
	{
		Bagging bagging(train, train_labels, 4, 100, 2);
		std::cout << "Training Accuracy: " << bagging.get_score(train, train_labels) << '\n';
		std::cout << "Test Accuracy: "     << bagging.get_score(test, test_labels)   << '\n';
	}
	else
		std::cout << "You entered the wrong choice. Exiting.\n";
	std::system("pause");
}
