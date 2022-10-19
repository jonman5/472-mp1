import joblib
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

import Task3


def write_to_performance_file(filename, model, classifier_task, c_matrix, c_report):
    with open(filename, 'a', encoding='UTF-8') as file:
        file.write("\n\n")
        file.write(model)
        file.write("\n")
        file.write("Classification task: " + classifier_task)
        file.write("\n")
        file.write("\nConfusion Matrix\n")
        file.write(str(c_matrix))
        file.write("\n\nClassification Report\n")
        file.write(str(c_report))


# Initialize class Task3 to do all the steps up to 3.5
task3 = Task3()
debugging_nr_posts = 3
# task3.tokenize_posts(task3.get_posts())
task3.tokenize_posts_partial_for_debugging(task3.get_posts(), debugging_nr_posts)
task3.display_nr_tokens()
task3.compute_embeddings_and_hit_rates_partial_for_debugging(debugging_nr_posts)
# print(task3.get_embedding_scores())
print(task3.get_hit_rates())
# task3.display_embeddings_test()

# Initialize and train base-MLP model for emotions classifier
x_train_emotions, x_test_emotions, y_train_emotions, y_test_emotions = task3.get_train_test_data("emotions")
MLP_emotions_model = MLPClassifier(verbose=True)
MLP_emotions_model.fit(x_train_emotions, y_train_emotions)
joblib.dump(MLP_emotions_model, "trained/MLP_emotions_trained.joblib")
print("MLP emotions model trained and saved to disk")

y_emotions_predictions = MLP_emotions_model.predict(x_test_emotions)

# Create confusion matrix and classification report and write to performance file
confusion_matrix = metrics.confusion_matrix(y_test_emotions, y_emotions_predictions)
cl_report = metrics.classification_report(y_test_emotions, y_emotions_predictions)

write_to_performance_file("performance.txt", "Multi Layer Perceptron", "emotion", confusion_matrix, cl_report)

# Initialize and train base-MLP model for sentiments classifier
x_train_sentiments, x_test_sentiments, y_train_sentiments, y_test_sentiments = task3.get_train_test_data("sentiments")
MLP_sentiments_model = MLPClassifier(verbose=True)
MLP_sentiments_model.fit(x_train_sentiments, y_train_sentiments)
joblib.dump(MLP_sentiments_model, "trained/MLP_sentiments_trained.joblib")
print("MLP sentiments model trained and saved to disk")

y_sentiments_predictions = MLP_sentiments_model.predict(x_test_sentiments)

# Create confusion matrix and classification report and write to performance file
confusion_matrix = metrics.confusion_matrix(y_test_sentiments, y_sentiments_predictions)
cl_report = metrics.classification_report(y_test_sentiments, y_sentiments_predictions)

write_to_performance_file("performance.txt", "Multi Layer Perceptron", "sentiment", confusion_matrix, cl_report)
