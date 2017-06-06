
import helper_quora as quora
trainDF, valDF = quora.load_train(train_file='downloads/train.csv', test_size=.2)
train_question1, train_question2, val_question1, val_question2 = quora.clean_up(trainDF=trainDF, valDF=valDF)
BagOfWordsExtractor, X_train, X_val, y_train, y_val = quora.BOW(train_data = trainDF, val_data=valDF, train_question1 = train_question1, train_question2 = train_question2, val_question1 = val_question1, val_question2 = val_question2, maxNumFeatures=10, ngrams=(1,2))
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
graph = quora.tensor(X_train=X_train, y_train = y_train, X_val = X_val, y_val = y_val, n_epochs=5, n_inputs=10,length=len(X_train))
testDF = quora.load_test(test_file='downloads/train.csv')
test = quora.process_test(BagOfWordsExtractor=BagOfWordsExtractor, testDF=testDF)
predictions = quora.tensor_load(length=len(test), test=test, prediction_length=10000, n_inputs=10)
output = quora.convert_to_output(predictions, len(test),to_csv=0)
