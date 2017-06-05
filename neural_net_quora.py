trainDF, valDF = load_train(train_file='../downloads/train.csv')
train_question1, train_question2, val_question1, val_question2 = clean_up(trainDF=trainDF, valDF=valDF)
BagOfWordsExtractor, X_train, x_val, y_train, y_val = BOW()
graph = tensor(n_epochs=25, length=len(X_train))
testDF = load_test(BagofWordsExtractor=BagofWordsExtractor,test_file='../downloads/test.csv', test_size=0.2)
test = process_test
predictions = tensor_load(length=len(test), prediction_length=10000)
output = convert_to_output(predictions, len(test),to_csv=1)
