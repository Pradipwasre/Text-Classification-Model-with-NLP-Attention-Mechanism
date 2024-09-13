lr = 0.001
vec_len = 50
seq_len = 20
num_epochs = 50
label_col = "Product"
tokens_path = "Output/tokens.pkl"
labels_path = "Output/labels.pkl"
data_path = "Input/complaints.csv"
model_path = "Output/attention.pth"
vocabulary_path = "Output/vocabulary.pkl"
embeddings_path = "Output/embeddings.pkl"
glove_vector_path = "Input/glove.6B.50d.txt"
text_col_name = "Consumer complaint narrative"
label_encoder_path = "Output/label_encoder.pkl"
product_map = {'Vehicle loan or lease': 'vehicle_loan',
               'Credit reporting, credit repair services, or other personal consumer reports': 'credit_report',
               'Credit card or prepaid card': 'card',
               'Money transfer, virtual currency, or money service': 'money_transfer',
               'virtual currency': 'money_transfer',
               'Mortgage': 'mortgage',
               'Payday loan, title loan, or personal loan': 'loan',
               'Debt collection': 'debt_collection',
               'Checking or savings account': 'savings_account',
               'Credit card': 'card',
               'Bank account or service': 'savings_account',
               'Credit reporting': 'credit_report',
               'Prepaid card': 'card',
               'Payday loan': 'loan',
               'Other financial service': 'others',
               'Virtual currency': 'money_transfer',
               'Student loan': 'loan',
               'Consumer Loan': 'loan',
               'Money transfers': 'money_transfer'}
