import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
import xgboost as xgb

# Read dataset
df = pd.read_csv("/Users/steam/Desktop/code/datacamp/cr_loan2.csv")

# Plot settings
def setup_plot():
    plt.style.use("dark_background")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['tab:blue'])
    plt.rcParams["figure.dpi"] = 500
setup_plot()

#XPLORATORY ANALYSIS-----------------------

# 0 = non-default
# 1 = default

def loan_dist():
    plt.hist(x=df['loan_amnt'], bins='auto')
    plt.xlabel("Loan Amount")
    plt.show()
    
# loan_dist()

def age_income():
    plt.scatter(df['person_income'], df['person_age'], alpha=0.3)
    plt.xlabel('Personal Income')
    plt.ylabel('Person Age')
    plt.show()

# age_income()
    
def box():
    df.boxplot(column = ['loan_percent_income'], by = 'loan_status')
    plt.title('Average Percent Income by Loan Status')
    plt.suptitle('')
    plt.show()

# box()

def crosstable1():
    ct1=pd.crosstab(df['loan_intent'], df['loan_status'], margins = True)
    return ct1

def crosstable2():
    ct2=pd.crosstab(df['person_home_ownership'],[df['loan_status'],
    df['loan_grade']])
    return ct2

def crosstable3():
    ct3=pd.crosstab(df['person_home_ownership'],df['loan_status'],
    values=df['loan_percent_income'],aggfunc='mean')
    return ct3

ct1 = crosstable1()
ct2 = crosstable2()
ct3 = crosstable3()

#OUTLIER AND MISSING DATA-----------------------


#OUTLIER DETECTION-----------------------

def age_emp(query):
    if query:
        plt.scatter(df['person_age'], df['person_emp_length'], alpha=0.5)
        plt.xlabel("Person Age")
        plt.ylabel("Employment length")
        plt.show()
    else:
        plt.scatter(clean_df['person_age'], clean_df['person_emp_length'],c="green", alpha=0.5)
        plt.xlabel("Person Age")
        plt.ylabel("Employment length")
        plt.show()

# age_emp(query=False)
clean_df = df.query("person_age <= 100 and person_emp_length <= 60")


# Print an array of columns with null values
# print(clean_df.columns[clean_df.isnull().any()])
# print(clean_df[clean_df['loan_int_rate'].isnull()])

#1 Replace the null values with the median value for all interest rates
clean_df.loc[:, 'loan_int_rate'] = clean_df['loan_int_rate'].fillna(clean_df['loan_int_rate'].median())
# print(clean_df.columns[clean_df.isnull().any()])

#2
# Store the array on indices
# indices = clean_df[clean_df['loan_int_rate'].isnull()].index

# Save the new data without missing data
# clean_df = clean_df.drop(indices)


#LOGISTIC REGRESSION-----------------------

# Create two data sets for numeric and non-numeric data
cred_num = clean_df.select_dtypes(exclude=['object'])
cred_str = clean_df.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# X=cr_loan_prep
X = cr_loan_prep.drop(columns=['loan_status'])
y = clean_df[['loan_status']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)


clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

preds = clf_logistic.predict_proba(X_test)

#AUC:
prob_default = preds[:, 1]
auc = roc_auc_score(y_test, prob_default)

# fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
# plt.plot(fallout, sensitivity, color='darkorange', label= f'AUC: {auc}')
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.show()

#classification:

#Precision : proportion of correctly predicted defaults out of all predicted defaults
#Recall : proportion of correctly predicted defaults out of all actual defaults.

# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.275 else 0)

# Print the classification report
target_names = ['Non-Default', 'Default']
# print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

cm=confusion_matrix(y_test,preds_df['loan_status'])
cm_df = pd.DataFrame(cm, index=['Actual Default', 'Actual Non-Default'], columns=['Predicted Default', 'Predicted Non-Default'])
# print(cm_df)

#impact:
avg_loan_amnt=clean_df["loan_amnt"].mean()

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['loan_status'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]

# Calculate the estimated impact of the new default recall rate
impact = round(num_defaults * avg_loan_amnt * (1 - default_recall), 2)
# print(f"IMPACT: {impact:,}")

#treshold:
thresh=[0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65]
def_recalls=[0.7981438515081206, 0.7583139984532096, 0.7157772621809745, 0.6759474091260634, 0.6349574632637278, 0.594354215003867, 0.5467904098994586, 0.5054137664346481, 0.46403712296983757, 0.39984532095901004, 0.32211910286156226, 0.2354988399071926, 0.16782675947409126, 0.1148491879350348, 0.07733952049497293, 0.05529775715390565, 0.03750966744006187, 0.026295436968290797, 0.017788089713843776]
nondef_recalls=[0.5342465753424658, 0.5973037616873234, 0.6552511415525114, 0.708306153511633, 0.756468797564688, 0.8052837573385518, 0.8482278756251359, 0.8864970645792564, 0.9215046749293324, 0.9492280930637095, 0.9646662317895195, 0.9733637747336378, 0.9809741248097412, 0.9857577734290063, 0.9902152641878669, 0.992280930637095, 0.9948901935203305, 0.9966297021091541, 0.997499456403566]
accs=[0.5921588594704684, 0.6326374745417516, 0.6685336048879837, 0.7012050237610319, 0.7298031228784793, 0.7589952477936185, 0.7820773930753564, 0.8028682959945689, 0.8211133740665308, 0.8286659877800407, 0.8236591989137814, 0.811439239646979, 0.8025288526816021, 0.7946367956551256, 0.7898845892735913, 0.7866598778004074, 0.7847929395790902, 0.7836897488119484, 0.7825016972165648]
ticks=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
# plt.plot(thresh,def_recalls)
# plt.plot(thresh,nondef_recalls)
# plt.plot(thresh,accs)
# plt.xlabel("Probability Threshold")
# plt.xticks(ticks)
# plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
# plt.show()



#Gradient boosted trees with XGBoost -----------------------

clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

gbt_preds = clf_gbt.predict_proba(X_test)

# gbt_preds = clf_gbt.predict(X_test)

target_names = ['Non-Default', 'Default']
# print(classification_report(y_test, gbt_preds, target_names=target_names))
# print(clf_gbt.get_booster().get_score(importance_type = 'weight'))

# Cross Validation
gbt = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7)
cv_scores = cross_val_score(gbt, X_train, np.ravel(y_train), cv = 4)
# print(cv_scores)
# print("Average accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(),cv_scores.std() * 2))


#Under-sampling 

X_y_train = pd.concat([X_train.reset_index(drop = True),
y_train.reset_index(drop = True)], axis = 1)

count_nondefault, count_default = X_y_train['loan_status'].value_counts()

# Create data sets for <defaults and non-defaults
nondefaults = X_y_train[X_y_train['loan_status'] == 0]
defaults = X_y_train[X_y_train['loan_status'] == 1]

# Undersample the non-defaults
nondefaults_under = nondefaults.sample(count_default, random_state=123)

# Concatenate the undersampled nondefaults with defaults
X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True),defaults.reset_index(drop = True)], axis = 0)

# Print the value counts for loan status
# print(X_y_train_under['loan_status'].value_counts())


#Model Evaluation

prob_default = preds[:, 1]

preds_df_gbt = pd.DataFrame(gbt_preds[:,1], columns = ['prob_default'])

preds_df_gbt['loan_status'] = preds_df_gbt['prob_default'].apply(lambda x: 1 if x > 0.275 else 0)

# print("LOGISTIC CLASSIFICATION REPORT",classification_report(y_test, preds_df['loan_status'], target_names=target_names))

# print("GBT CLASSIFICATION REPORT",classification_report(y_test, preds_df_gbt['loan_status'], target_names=target_names))


# ROC chart
# fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(y_test, preds_df["prob_default"])
# fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test, preds_df_gbt["prob_default"])
# plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
# plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
# plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
# plt.title("ROC Chart for LR and GBT on the Probability of Default")
# plt.xlabel('Fall-out')
# plt.ylabel('Sensitivity')
# plt.legend()
# plt.show()


#Calibration curve
# frac_of_pos_lr = calibration_curve(y_test, preds_df['prob_default'], n_bins=20)[0]
# mean_pred_val_lr = calibration_curve(y_test, preds_df['prob_default'], n_bins=20)[1]
# frac_of_pos_gbt = calibration_curve(y_test, preds_df_gbt['prob_default'], n_bins=20)[0]
# mean_pred_val_gbt = calibration_curve(y_test, preds_df_gbt['prob_default'], n_bins=20)[1]
# plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated',c="white")    
# plt.plot(mean_pred_val_lr, frac_of_pos_lr, 's-', label='%s' % 'Logistic Regression')
# plt.plot(frac_of_pos_gbt, mean_pred_val_gbt, 's-', label='%s' % 'Gradient Boosted tree',c="red")
# plt.ylabel('Fraction of positives')
# plt.xlabel('Average Predicted Probability')
# plt.legend()
# plt.title('Calibration Curve')
# plt.show()


#Acceptance rates

# Check the statistics of the probabilities of default
# print(preds_df_gbt['prob_default'].describe())
# Calculate the threshold for a 85% acceptance rate
threshold_85 = np.quantile(preds_df_gbt['prob_default'], 0.85)
# Apply acceptance rate threshold
preds_df_gbt['pred_loan_status'] = preds_df_gbt['prob_default'].apply(lambda x: 1 if x > threshold_85 else 0)
# Print the counts of loan status after the threshold
preds_df_gbt['pred_loan_status'].value_counts()


# Plot the predicted probabilities of default
# plt.hist(preds_df_gbt['prob_default'], color = 'blue', bins = 40)
# threshold = np.quantile(preds_df_gbt['prob_default'], 0.85)
# plt.axvline(x = threshold, color = 'red')
# plt.show()


#Bad rate
preds_df_gbt = preds_df_gbt.drop(columns=['loan_status'])
test_pred_df = pd.concat([preds_df_gbt.reset_index(drop = True), y_test['loan_status'].reset_index(drop = True)], axis = 1)
test_pred_df = test_pred_df.rename(columns={"loan_status":"true_loan_status"})
# print(test_pred_df.head())
# Create a subset of only accepted loans
accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]
# Calculate the bad rate
# print(np.sum(accepted_loans['true_loan_status']) / accepted_loans['true_loan_status'].count())


#Acceptance rate impact 
test_pred_df = pd.concat([test_pred_df.reset_index(drop = True), X_test['loan_amnt'].reset_index(drop = True)], axis = 1)
# print(test_pred_df['loan_amnt'].describe())
avg_loan = np.mean(test_pred_df['loan_amnt'])
pd.options.display.float_format = '${:,.2f}'.format
# print(pd.crosstab(test_pred_df['true_loan_status'], test_pred_df['pred_loan_status']).apply(lambda x: x * avg_loan, axis = 0))


#Strategy table

accept_rates = [round(0.05 * i, 2) for i in range(20, 0, -1)]
# print(accept_rates)

thresholds = []
bad_rates = []
num_accepted_loans = []
preds_df_gbt = test_pred_df

# Populate the arrays for the strategy table with a for loop
for rate in accept_rates:
    
    # Calculate and append the threshold for the acceptance rate
    thresh = np.quantile(preds_df_gbt['prob_default'], rate)
    thresholds.append(thresh.round(3))
    
    # Reassign the loan_status value using the threshold
    test_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > thresh else 0)
    
    # Create a set of accepted loans using this acceptance rate
    accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]
    num_accepted_loans.append(len(accepted_loans))
    
    # Calculate and append the bad rate using the acceptance rate
    bad_rates.append(np.sum((accepted_loans['true_loan_status']) / len(accepted_loans['true_loan_status'])).round(3))
    
# Create a data frame of the strategy table
strat_df = pd.DataFrame(zip(accept_rates, thresholds, bad_rates, num_accepted_loans), columns = ['Acceptance Rate','Threshold','Bad Rate', 'Num Accepted Loans'])

# Visualize the distributions in the strategy table with a boxplot
strat_df[['Acceptance Rate','Threshold','Bad Rate']].boxplot()
plt.show()

# Add average loan amount
strat_df['Avg Loan Amnt'] = np.mean(test_pred_df['loan_amnt']).round(2)

# Add estimated portfolio value
strat_df['Estimated Value'] = ((strat_df['Num Accepted Loans'] * (1 - strat_df['Bad Rate'])) * strat_df['Avg Loan Amnt']) \
                              - (strat_df['Num Accepted Loans'] * strat_df['Bad Rate'] * strat_df['Avg Loan Amnt'])

# Print the entire table
pd.options.display.float_format = None
strat_df.style.format({"Avg Loan Amnt": "${:,.2f}", "Estimated Value": "${:,.2f}"})


# Plot the strategy curve
plt.plot(strat_df['Acceptance Rate'], strat_df['Bad Rate'])
plt.xlabel('Acceptance Rate')
plt.ylabel('Bad Rate')
plt.title('Acceptance and Bad Rates')
plt.grid()
plt.show()


#Line plot of estimated value
plt.plot(strat_df['Acceptance Rate'], strat_df['Estimated Value'])
plt.title('Estimated Value by Acceptance Rate')
plt.xlabel('Acceptance Rate')
plt.ylabel('Estimated Value')
plt.grid()
plt.show()

# Print the row with the max estimated value
max_est_row = strat_df.loc[strat_df['Estimated Value'] == np.max(strat_df['Estimated Value'])].style.format({"Avg Loan Amnt": "${:,.2f}", "Estimated Value": "${:,.2f}"}).to_string()
# print(max_est_row)


#Max Est Loss

# Calculate the bank's expected loss and assign it to a new column
preds_df_gbt['expected_loss'] = preds_df_gbt['prob_default'] * 1 * preds_df_gbt['loan_amnt']

# Calculate the total expected loss to two decimal places
tot_exp_loss = round(np.sum(preds_df_gbt['expected_loss']),2)

# Print the total expected loss
print('Total expected loss: ', '${:,.2f}'.format(tot_exp_loss))





