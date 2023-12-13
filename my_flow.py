"""

This script is a Metaflow-based refactoring of the text classification pipeline in the notebook. Its purpose is to
show a more realistic DAG as an example for the final project: note that we skip some steps for brevity, so refer 
to the project checklist from week 11 (https://github.com/jacopotagliabue/MLSys-NYU-2023/blob/main/weeks/11/Project_checklist.pdf) 
for a more complete set of features/requirements.

This script has been created for pedagogical purposes, and it does NOT necessarely reflect all best practices.

"""


from metaflow import FlowSpec, step, current, Parameter
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier




class LoanDefaultFlow(FlowSpec):
    """
    This is the metaflow for training different models to predict whether the loan will default or not
    and get the best model and improves it
    """
    model_name = Parameter('model_name',
                      help='dict of models name and its method',
                      default=[RandomForestClassifier(random_state=42),
                          XGBClassifier(random_state=42),
                          AdaBoostClassifier(random_state=42),
                          GradientBoostingClassifier(random_state=42),
                          LogisticRegression(random_state=42)
                          ])
    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: {}".format(current.flow_name))
        print("run id: {}".format(current.run_id))
        print("username: {}".format(current.username))

        self.next(self.load_data)

    @step
    def load_data(self): 
        
        from flow_utils import data
        # load data
        self.df = pd.read_csv('/Users/ganyuke/Desktop/7773 Final Project/train.csv')
        
        # count the categorical columns and numerical columns
        # Get all categorical, non-categorical columns
        self.categorical_columns, self.non_categorical_columns = data(self.df)
        
        print("Total Categorical Columns:", len(self.categorical_columns))
        print("Total Non-Categorical Columns:", len(self.non_categorical_columns))

        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check if any feature contain unique value, drop that feature, replace with new df and cat columns
        """
        from flow_utils import drop_unique
        self.categorical_columns, self.df = drop_unique(self.categorical_columns, self.df)

        # if data is all good, let's go to training
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from flow_utils import imbalance, get_X_y, cat_convert_num
        
        # Convert categorical columns to numeric labels
        self.label_list, self.df = cat_convert_num(self.categorical_columns, self.df)
        
        """prepare X and y for training and testing"""
        X, y = get_X_y(self.df)
        """check imbalancy of target value df['Loan status']"""
        # Calculate percentage for each 'Loan Status' category
        status_percentage = (y.value_counts() / len(y)) * 100
        print("Current percentage for target value 1 is:", status_percentage)
        # balance data
        self.X, self.y = imbalance(status_percentage, X, y)
        # check new percentage
        status_percentage = (self.y.value_counts() / len(self.y))
        print("The new percentage for target value 1 is:", status_percentage)
        
        """ Train / test split """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.TEST_SPLIT, 
            random_state=42, 
            stratify=self.y)

        # debug / info
        print("# train sentences: {},  # test: {}".format(len(self.X_train), len(self.X_test)))
        
        # standardize the features as some are not normal distributed
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # next step to train different models
        self.next(self.foreach)

    @step
    def foreach(self):
        self.next(self.train_model, foreach='model_name')

    @step
    def train_model(self):
        """
        train 5 models together
        """
        model = self.input
        model.fit(self.X_train, self.y_train)
        self.trained_model = model
        # go to the testing phase
        self.next(self.test_model)

    @step 
    def test_model(self, inputs):
        """
        Test the model on the held out sample and compare matrices

        TODO: add confusion matrix and a plot!

        TODO: sends a summary of these metrics to Comet to make sure we track them!
        """
        from flow_utils import select_model
        models = [input.trained_model for input in inputs]
        self.merge_artifacts(inputs, include = ['X_train','X_test', 'y_train', 'y_test', 'X', 'y', 'label_list', 'scaler'])
        self.good_model, self.model_report = select_model(models, self.X_test, self.y_test)
        self.good_model.feature_names = self.X.columns.tolist()
        # print out the report
        print("!!!!! Classification Report !!!!!")
        print(self.model_report)
        # all is done go to the end
        self.next(self.check_feature_importance)

    @step
    def check_feature_importance(self):
        """
        Since we have 32 features in X, we want to make sure no overfitting
        """
        from flow_utils import new_features, new_feature_model
        self.X_newtrain, self.X_newtest, self.new_X = new_features(self.good_model, self.X, self.X_train, self.X_test)
        self.feature_model, self.feature_report = new_feature_model(self.X_newtrain,self.y_train,self.X_newtest,self.y_test, self.new_X)
        
        # compare models
        print(f"Original Precision: {self.model_report['1']['precision']}")
        print(f"Improved Precision: {self.feature_report['1']['precision']}")
        if self.feature_report['1']['precision'] < self.model_report['1']['precision']:
            self.X_newtrain, self.X_newtest, self.new_X = self.X_train, self.X_test, self.X
            self.feature_model, self.feature_report = self.good_model, self.model_report
            print('The original one is better')
        else:
            print('The improved model is better')

        # next, improve the model
        self.next(self.improve_model)

    @step
    def improve_model(self):
        from flow_utils import improved_model
        self.best_model, self.best_report = improved_model(self.X_newtrain, self.y_train, self.X_newtest, self.y_test, self.new_X)
        print("!!!!! Classification Report !!!!!")
        print(self.best_report)

        # compare the improved report with original report by precision score on target value 1
        print("Compare the precision on target value 1:")
        print(f"Original Precision: {self.feature_report['1']['precision']}")
        print(f"Improved Precision: {self.best_report['1']['precision']}")
        if self.feature_report['1']['precision'] > self.best_report['1']['precision']:
            self.best_model = self.feature_model
            print('The original one is better')
        else:
            print('The improved model is better')
        
        # all is done, dump the model
        self.next(self.dump_for_serving)

    @step
    def dump_for_serving(self):
        """
        Make sure we pickled the artifacts necessary for the Flask app to work. Note that 
        we just dump the model and the vectorizer in the current dir.

        Hint: is there a better way of doing this than pickling feature prep and model in two files? ;-)
        """
        import pickle

        pickle.dump(self.label_list, open('label_encoder_list.pkl', 'wb+'))
        pickle.dump(self.best_model, open('model.pkl', 'wb+'))
        pickle.dump(self.scaler, open('scaler.pkl', 'wb+'))
        # go to the end
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))

if __name__ == '__main__':
    LoanDefaultFlow()