import joblib
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from Task3.Task3Helper import Helper as Task3Helper


def write_to_performance_file(filename, model, classifier_task, c_matrix, c_report, best_params):
    with open(filename, 'a', encoding='UTF-8') as file:
        file.write("\n\n")
        file.write(model)
        file.write("\n")
        file.write("Hyper parameters used:" + str(best_params) + "\n")
        file.write("Classification task: " + classifier_task)
        file.write("\n")
        file.write("\nConfusion Matrix\n")
        file.write(str(c_matrix))
        file.write("\n\nClassification Report\n")
        file.write(str(c_report))


# Define which pretrained model to use from gensim
pretrained_model_name = "fasttext-wiki-news-subwords-300"

# Initialize class Helper to do all the steps up to 3.5
task3 = Task3Helper.Helper(pretrained_model_name)
task3.tokenize_posts(task3.get_posts())
task3.display_nr_tokens()
task3.compute_embeddings_and_hit_rates()

# Initialize and train MLP model for emotions classifier using the optimal hyperparameters found
x_train_emotions, x_test_emotions, y_train_emotions, y_test_emotions = task3.get_train_test_data("emotions", 0.2)
MLP_emotions_model = MLPClassifier(verbose=True, activation='identity', hidden_layer_sizes=(30, 50), solver='adam')
MLP_emotions_model.fit(x_train_emotions, y_train_emotions)
joblib.dump(MLP_emotions_model, "../Exploration_2_MLP_emotions_trained.joblib")
print("MLP emotions model trained and saved to disk")

y_emotions_predictions = MLP_emotions_model.predict(x_test_emotions)

# Create confusion matrix and classification report and write to performance file
confusion_matrix = metrics.confusion_matrix(y_test_emotions, y_emotions_predictions)
cl_report = metrics.classification_report(y_test_emotions, y_emotions_predictions)

write_to_performance_file("../performance.txt", "Multi Layer Perceptron using " + pretrained_model_name, "emotion", confusion_matrix, cl_report, MLP_emotions_model.get_params())

# Initialize and train MLP model for sentiments classifier  using the optimal hyperparameters found
x_train_sentiments, x_test_sentiments, y_train_sentiments, y_test_sentiments = task3.get_train_test_data("sentiments",
                                                                                                         0.2)
MLP_sentiments_model = MLPClassifier(verbose=True, activation='logistic', hidden_layer_sizes=(30, 50), solver='adam')
MLP_sentiments_model.fit(x_train_sentiments, y_train_sentiments)
joblib.dump(MLP_sentiments_model, "../Exploration_2_MLP_sentiments_trained.joblib")
print("MLP sentiments model trained and saved to disk")

y_sentiments_predictions = MLP_sentiments_model.predict(x_test_sentiments)

# Create confusion matrix and classification report and write to performance file
confusion_matrix = metrics.confusion_matrix(y_test_sentiments, y_sentiments_predictions)
cl_report = metrics.classification_report(y_test_sentiments, y_sentiments_predictions)

write_to_performance_file("../performance.txt", "Multi Layer Perceptron using " + pretrained_model_name, "sentiment", confusion_matrix, cl_report, MLP_emotions_model.get_params())
