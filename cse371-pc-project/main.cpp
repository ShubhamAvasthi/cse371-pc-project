#include "mlpack/core.hpp"
#include "mlpack/methods/decision_tree/decision_tree.hpp"
#include "mlpack/methods/softmax_regression/softmax_regression.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/F1.hpp"
#include "mlpack/core/data/split_data.hpp"
#include <random>
#include <chrono>
#include <vector>
#include <array>

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;

std::default_random_engine generator;

const size_t numClasses = 2;

void bagging(int num_classifiers, int sample_size, mat train, mat test)
{
	// Training examples are present columnwise
	std::uniform_int_distribution<int> distribution(0, train.n_cols - 1);

	Row<size_t> train_labels = conv_to<Row<size_t>>::from(train.row(train.n_rows - 1));
	train.shed_row(train.n_rows - 1);
	Row<size_t> test_labels = conv_to<Row<size_t>>::from(test.row(test.n_rows - 1));
	test.shed_row(test.n_rows - 1);

	std::vector<Row<size_t>> all_test_predictions;

	// Loop over all num_classifiers classifiers
	for (int i = 0; i < num_classifiers; i++)
	{
		mat sampled_train(train.n_rows, sample_size);
		Row<size_t> sampled_labels(sample_size);
		//cout << sampled_train.n_rows << ' ' << sampled_train.n_cols << ' ' << dataset.n_rows << ' ' << dataset.n_cols << '\n';
		for (int j = 0; j < sample_size; j++)
		{
			int index = distribution(generator);
			sampled_train.col(j) = train.col(index);
			sampled_labels[j] = train_labels[index];
		}

		const size_t minimumLeafSize = 5;
		DecisionTree<> dt(sampled_train, sampled_labels, numClasses, minimumLeafSize);
		
		Row<size_t> predictions;
		dt.Classify(sampled_train, predictions);
		const size_t correct = arma::accu(predictions == sampled_labels);
		cout << "\nTraining Accuracy: " << (double(correct) / double(sampled_labels.n_elem)) << '\n';

		const size_t k = 10;
		KFoldCV<DecisionTree<>, Accuracy> cv(k, train, train_labels, numClasses);
		double cvAcc = cv.Evaluate(minimumLeafSize);
		cout << "KFoldCV Accuracy: " << cvAcc << '\n';

		double cvPrecision = Precision<Binary>::Evaluate(dt, sampled_train, sampled_labels);
		cout << "Precision: " << cvPrecision << '\n';
		double cvRecall = Recall<Binary>::Evaluate(dt, sampled_train, sampled_labels);
		cout << "Recall: " << cvRecall << '\n';
		double cvF1 = F1<Binary>::Evaluate(dt, sampled_train, sampled_labels);
		cout << "F1: " << cvF1 << '\n';

		Row<size_t> test_predictions;
		dt.Classify(test, test_predictions);
		const size_t test_correct = arma::accu(test_predictions == test_labels);
		cout << "Test Accuracy: " << (double(test_correct) / double(test_labels.n_elem)) << '\n';

		all_test_predictions.push_back(test_predictions);
	}

	Row<size_t> final_test_predictions(all_test_predictions[0].size());
	for (int i = 0; i < all_test_predictions[0].size(); i++)
	{
		std::array<int, numClasses> num_predictions;
		for (int i = 0; i < numClasses; i++)
			num_predictions[i] = 0;
		
		for (int j = 0; j < all_test_predictions.size(); j++)
			num_predictions[all_test_predictions[j][i]]++;
		
		int max_freq_index = -1, max_freq = -1;
		for(int i = 0; i < numClasses; i++)
			if (num_predictions[i] > max_freq)
			{
				max_freq = num_predictions[i];
				max_freq_index = i;
			}
		final_test_predictions[i] = max_freq_index;
	}

	const size_t test_correct = arma::accu(final_test_predictions == test_labels);
	cout << "\nFinal Test Accuracy for the Ensemble: " << (double(test_correct) / double(test_labels.n_elem)) << '\n';
}

void stacking(mat train, mat test)
{
	// Training examples are present columnwise

	Row<size_t> train_labels = conv_to<Row<size_t>>::from(train.row(train.n_rows - 1));
	train.shed_row(train.n_rows - 1);
	Row<size_t> test_labels = conv_to<Row<size_t>>::from(test.row(test.n_rows - 1));
	test.shed_row(test.n_rows - 1);

	std::vector<Row<size_t>> all_test_predictions;

	// Decision Tree Classifier

	const size_t minimumLeafSize = 5;
	DecisionTree<> dt(train, train_labels, numClasses, minimumLeafSize);

	Row<size_t> dt_predictions;
	dt.Classify(train, dt_predictions);
	const size_t dt_correct = arma::accu(dt_predictions == train_labels);
	cout << "\nTraining Accuracy: " << (double(dt_correct) / double(train_labels.n_elem)) << '\n';

	KFoldCV<DecisionTree<>, Accuracy> dt_cv(10, train, train_labels, numClasses);
	double dt_cvAcc = dt_cv.Evaluate(minimumLeafSize);
	cout << "KFoldCV Accuracy: " << dt_cvAcc << '\n';

	double dt_cvPrecision = Precision<Binary>::Evaluate(dt, train, train_labels);
	cout << "Precision: " << dt_cvPrecision << '\n';
	double dt_cvRecall = Recall<Binary>::Evaluate(dt, train, train_labels);
	cout << "Recall: " << dt_cvRecall << '\n';
	double dt_cvF1 = F1<Binary>::Evaluate(dt, train, train_labels);
	cout << "F1: " << dt_cvF1 << '\n';

	Row<size_t> dt_test_predictions;
	dt.Classify(test, dt_test_predictions);
	const size_t dt_test_correct = arma::accu(dt_test_predictions == test_labels);
	cout << "Test Accuracy: " << (double(dt_test_correct) / double(test_labels.n_elem)) << '\n';

	all_test_predictions.push_back(dt_test_predictions);

	// Softmax Regression Classifier

	regression::SoftmaxRegression sr(train, train_labels, numClasses);

	Row<size_t> sr_predictions;
	sr.Classify(train, sr_predictions);
	const size_t sr_correct = arma::accu(sr_predictions == train_labels);
	cout << "\nTraining Accuracy: " << (double(sr_correct) / double(train_labels.n_elem)) << '\n';

	KFoldCV<DecisionTree<>, Accuracy> sr_cv(10, train, train_labels, numClasses);
	double sr_cvAcc = sr_cv.Evaluate(minimumLeafSize);
	cout << "KFoldCV Accuracy: " <<sr_cvAcc << '\n';

	double sr_cvPrecision = Precision<Binary>::Evaluate(sr, train, train_labels);
	cout << "Precision: " << sr_cvPrecision << '\n';
	double sr_cvRecall = Recall<Binary>::Evaluate(sr, train, train_labels);
	cout << "Recall: " << sr_cvRecall << '\n';
	double sr_cvF1 = F1<Binary>::Evaluate(sr, train, train_labels);
	cout << "F1: " << sr_cvF1 << '\n';

	Row<size_t> sr_test_predictions;
	sr.Classify(test, sr_test_predictions);
	const size_t sr_test_correct = arma::accu(sr_test_predictions == test_labels);
	cout << "Test Accuracy: " << (double(sr_test_correct) / double(test_labels.n_elem)) << '\n';

	all_test_predictions.push_back(sr_test_predictions);

	// Final testing

	Row<size_t> final_test_predictions(all_test_predictions[0].size());
	for (int i = 0; i < all_test_predictions[0].size(); i++)
	{
		std::array<int, numClasses> num_predictions;
		for (int i = 0; i < numClasses; i++)
			num_predictions[i] = 0;

		for (int j = 0; j < all_test_predictions.size(); j++)
			num_predictions[all_test_predictions[j][i]]++;

		int max_freq_index = -1, max_freq = -1;
		for (int i = 0; i < numClasses; i++)
			if (num_predictions[i] > max_freq)
			{
				max_freq = num_predictions[i];
				max_freq_index = i;
			}
		final_test_predictions[i] = max_freq_index;
	}

	const size_t test_correct = arma::accu(final_test_predictions == test_labels);
	cout << "\nFinal Test Accuracy for the Ensemble: " << (double(test_correct) / double(test_labels.n_elem)) << '\n';
}

int main()
{
	generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	mat dataset, train, test;
	mlpack::data::Load("german.csv", dataset);
	mlpack::data::Split(dataset, train, test, 0.3);

	bagging(5, 100, train, test);
	// stacking(train, test);

	system("pause");
}
