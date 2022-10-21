import joblib
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from Task3 import Task3


def write_to_performance_file(filename, model, classifier_task, c_matrix, c_report, best_params):
    with open(filename, 'a', encoding='UTF-8') as file:
        file.write("\n\n")
        file.write(model + "\n")
        file.write("Hyper parameters used:" + best_params + "\n")
        file.write("Classification task: " + classifier_task)
        file.write("\n")
        file.write("\nConfusion Matrix\n")
        file.write(str(c_matrix))
        file.write("\n\nClassification Report\n")
        file.write(str(c_report))


# Initialize class Task3 to do all the steps up to 3.5
task3 = Task3("word2vec-google-news-300")
# debugging_nr_posts = 3
task3.tokenize_posts(task3.get_posts())
task3.display_nr_tokens()
task3.compute_embeddings_and_hit_rates()
# print(task3.get_hit_rates())
# task3.display_embeddings_test()

# define GridSearch parameters
gridsearch_parameters = {'activation': ('softmax', 'relu'),
              'hidden_layer_sizes': [(30, 50)],
              'solver': ['sgd']}

# Initialize and train Top-MLP model for emotions classifier
x_train_emotions, x_test_emotions, y_train_emotions, y_test_emotions = task3.get_train_test_data("emotions", 0.2)

MLP_emotions_model = MLPClassifier(verbose=True)
optimized_model = GridSearchCV(MLP_emotions_model, gridsearch_parameters, scoring='f1_weighted')
optimized_model.fit(x_train_emotions, y_train_emotions)
joblib.dump(optimized_model, "Top_MLP_emotions_trained.joblib")
print("Top-MLP emotions model trained and saved to disk")
print("Best Parameters: ", optimized_model.best_params_)

y_emotions_predictions = optimized_model.predict(x_test_emotions)

# Create confusion matrix and classification report and write to performance file
confusion_matrix = metrics.confusion_matrix(y_test_emotions, y_emotions_predictions)
cl_report = metrics.classification_report(y_test_emotions, y_emotions_predictions)

write_to_performance_file("performance.txt", "Top Multi Layer Perceptron", "emotion", confusion_matrix, cl_report, optimized_model.best_params_)

# Initialize and train Top-MLP model for sentiments classifier
x_train_sentiments, x_test_sentiments, y_train_sentiments, y_test_sentiments = task3.get_train_test_data("sentiments",
                                                                                                         0.2)
MLP_sentiments_model = MLPClassifier(verbose=True)
optimized_model = GridSearchCV(MLP_emotions_model, gridsearch_parameters, scoring='f1_weighted')
optimized_model.fit(x_train_sentiments, y_train_sentiments)
joblib.dump(optimized_model, "Top_MLP_sentiments_trained.joblib")
print("Top-MLP sentiments model trained and saved to disk")
print("Best Parameters: ", optimized_model.best_params_)

y_sentiments_predictions = optimized_model.predict(x_test_sentiments)

# Create confusion matrix and classification report and write to performance file
confusion_matrix = metrics.confusion_matrix(y_test_sentiments, y_sentiments_predictions)
cl_report = metrics.classification_report(y_test_sentiments, y_sentiments_predictions)

write_to_performance_file("performance.txt", "Top Multi Layer Perceptron", "sentiment", confusion_matrix, cl_report, optimized_model.best_params_)
